"""CNN view of every candidate: predicts its expected matched IoU (all filaments / Left-Right only).

Cross-fitted over the segmentation folds; the out-of-fold predictions become two extra descriptors
for the gradient-boosted scorer.

    python scripts/cnn_scorer.py --cand v1all
"""
import sys, math, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from torch.utils.data import Dataset, DataLoader
from fil import paths
from fil.dataset import IMG_MEAN, IMG_STD


def to_input(c: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 5, h, w) -> normalised float input."""
    x = c.float() / 255.0
    x[:, 0] = (x[:, 0] - IMG_MEAN) / IMG_STD
    return x


class CropSet(Dataset):
    def __init__(self, path, idx, y=None, train=False):
        self.path, self.idx, self.y, self.train = path, idx, y, train
        self.mm = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        if self.mm is None:
            self.mm = np.load(self.path, mmap_mode="r")
        c = np.array(self.mm[self.idx[k]])
        if self.train:
            if np.random.rand() < 0.5: c = c[:, :, ::-1]
            if np.random.rand() < 0.5: c = c[:, ::-1]
            if np.random.rand() < 0.5: c = c.transpose(0, 2, 1)
        c = torch.from_numpy(np.ascontiguousarray(c))
        return (c, torch.from_numpy(self.y[k])) if self.y is not None else c


def build(arch="resnet34.a1_in1k", pretrained=True):
    return timm.create_model(arch, pretrained=pretrained, in_chans=5, num_classes=2)


@torch.no_grad()
def predict(models, path, idx, bs=512, workers=8, tta=True):
    dl = DataLoader(CropSet(path, idx), batch_size=bs, num_workers=workers)
    out = []
    for c in dl:
        x = to_input(c.cuda(non_blocking=True))
        views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), x.transpose(2, 3)] if tta else [x]
        acc = 0
        for m in models:
            for v in views:
                with torch.autocast("cuda", dtype=torch.float16):
                    acc = acc + torch.sigmoid(m(v).float())
        out.append((acc / (len(models) * len(views))).cpu().numpy())
    return np.concatenate(out)


def targets(df):
    mi = df[["miou0", "miou1", "miou2"]].values
    y_off = np.nanmean(mi, 1)
    lr = np.stack([np.where((df[f"cat{q}"].fillna(0).values >= 1) & (df[f"cat{q}"].fillna(0).values <= 2), mi[:, q], np.where(np.isnan(mi[:, q]), np.nan, 0.0)) for q in range(3)], 1)
    return np.stack([y_off, np.nanmean(lr, 1)], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--arch", default="resnet34.a1_in1k")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    df = pd.read_pickle(paths.RUNS / "cand" / f"{a.cand}.pkl")
    path = paths.RUNS / "cand" / f"{a.cand}_crops.npy"
    Y = targets(df); fold = df.fold.values; w = df.nrec.values.astype(np.float64)
    oof = np.zeros((len(df), 2), np.float32)
    for f in sorted(np.unique(fold)):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
        sampler = torch.utils.data.WeightedRandomSampler(w[tr] / w[tr].sum(), num_samples=len(tr), replacement=True)
        dl = DataLoader(CropSet(path, tr, Y[tr], train=True), batch_size=a.bs, sampler=sampler, num_workers=12, drop_last=True, pin_memory=True)
        model = build(a.arch).cuda()
        total = a.epochs * len(dl)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(it, total) / total)))
        scaler = torch.amp.GradScaler("cuda"); t0 = time.time()
        for ep in range(a.epochs):
            model.train(); run = 0.0
            for c, y in dl:
                x = to_input(c.cuda(non_blocking=True)); y = y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                loss = F.binary_cross_entropy_with_logits(out.float(), y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                run += loss.item()
            print(f"fold {f} epoch {ep} loss {run/len(dl):.4f} ({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        oof[te] = predict([model], path, te)
        torch.save({"model": model.state_dict(), "arch": a.arch}, paths.RUNS / "cand" / f"{a.cand}_cnn_fold{f}.pt")
        print(f"fold {f}: corr official {np.corrcoef(oof[te, 0], Y[te, 0])[0, 1]:.4f}  corr L/R {np.corrcoef(oof[te, 1], Y[te, 1])[0, 1]:.4f}", flush=True)
    np.save(paths.RUNS / "cand" / f"{a.cand}_cnn.npy", oof)
    print(f"all: corr official {np.corrcoef(oof[:, 0], Y[:, 0])[0, 1]:.4f}  corr L/R {np.corrcoef(oof[:, 1], Y[:, 1])[0, 1]:.4f}  rmse {np.sqrt(((oof - Y) ** 2).mean(0))}")


if __name__ == "__main__":
    main()
