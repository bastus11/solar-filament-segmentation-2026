"""Train the stage-2 refiner for one fold (on candidates of the other folds) and measure its effect
on the held-out fold with the plain threshold pipeline.

    python scripts/refine_train.py --samples v1 --exp v1_d4 --fold 0
"""
import sys, json, math, time, argparse, pickle
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fil import data, paths, metric, postproc, refine, candidates


class RefineSet(Dataset):
    def __init__(self, samples, exp, length, seed=0):
        self.s, self.exp, self.length, self.seed = samples, exp, length, seed
        self._img = self._oof = self._gt = None

    def _open(self):
        if self._img is None:
            self._img = np.load(paths.CACHE / "img.npy", mmap_mode="r")
            self._oof = {f: np.load(paths.RUNS / self.exp / f"oof_fold{f}.npy", mmap_mode="r") for f in range(5)
                         if (paths.RUNS / self.exp / f"oof_fold{f}.npy").exists()}
            self._gt = metric.gt_pixel_cache()

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        self._open()
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, k, info.id if info else 0, int(torch.randint(0, 2**31 - 1, (1,)))])
        s = self.s[int(rng.integers(len(self.s)))]
        m = s["m"]; W = refine.WIN
        ys, xs = np.nonzero(m)
        q = int(rng.integers(len(xs)))
        wx = s["x0"] + int(xs[q]) - W // 2 + int(rng.integers(-56, 57))
        wy = s["y0"] + int(ys[q]) - W // 2 + int(rng.integers(-56, 57))
        img = refine.window(self._img[s["img"]], wx, wy, W)
        P = refine.window(self._oof[s["fold"]][s["j"], 0], wx, wy, W)
        S = refine.window(self._oof[s["fold"]][s["j"], 1], wx, wy, W)
        cw = np.zeros((W, W), np.uint8)
        ox, oy = s["x0"] - wx, s["y0"] - wy
        bh, bw = m.shape
        xa, ya = max(ox, 0), max(oy, 0); xb, yb = min(ox + bw, W), min(oy + bh, W)
        if xb > xa and yb > ya:
            cw[ya:yb, xa:xb] = m[ya - oy:yb - oy, xa - ox:xb - ox]
        tgt = np.zeros((W, W), np.float32)
        for rid, gid in zip(s["rec"], s["gid"]):
            pix = self._gt[rid][gid - 1]
            py, px = pix // paths.W - wy, pix % paths.W - wx
            ok = (py >= 0) & (py < W) & (px >= 0) & (px < W)
            t = np.zeros((W, W), np.float32); t[py[ok], px[ok]] = 1.0
            tgt += t
        tgt /= len(s["rec"])
        imgf = img.astype(np.float32) / 255.0
        if rng.random() < 0.7:
            imgf = (imgf - 0.35) * math.exp(rng.uniform(-0.15, 0.15)) + 0.35 + rng.uniform(-0.03, 0.03)
        x = np.stack([(imgf - refine.IMG_MEAN) / refine.IMG_STD, cw.astype(np.float32), P.astype(np.float32) / 255.0, S.astype(np.float32) / 255.0])
        y = tgt[None]
        if rng.random() < 0.5: x, y = x[:, :, ::-1], y[:, :, ::-1]
        if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5: x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def loss_fn(logit, tgt):
    bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p = torch.sigmoid(logit)
    inter = (p * tgt).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + tgt.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(models, exp, fold, thr=0.45, min_area=400, rthr=0.5):
    stems = json.load(open(paths.CACHE / "stems.json")); idx = {s: i for i, s in enumerate(stems)}
    vst = json.load(open(paths.RUNS / exp / f"oof_fold{fold}_stems.json"))
    oof = np.load(paths.RUNS / exp / f"oof_fold{fold}.npy", mmap_mode="r")
    img_mm = np.load(paths.CACHE / "img.npy", mmap_mode="r")
    by = data.records_by_stem(data.load_records()); gp = metric.gt_pixel_cache()
    a, b = metric.PQ(), metric.PQ()
    for j, stem in enumerate(vst):
        P = np.ascontiguousarray(oof[j, 0]); S = np.ascontiguousarray(oof[j, 1]); img = np.ascontiguousarray(img_mm[idx[stem]])
        lab, areas, stats = postproc.cc_instances(P, int(round(thr * 255)), min_area)
        masks = [(int(x), int(y), (lab[y:y + h, x:x + w] == i + 1).astype(np.uint8)) for i, (x, y, w, h, _) in enumerate(stats)]
        new = refine.refine_masks(models, img, P, S, masks, thr=rthr) if masks else []
        order = np.argsort([-int(m.sum()) for _, _, m in new]) if new else []
        lab2, _ = candidates.paint(new, [int(o) for o in order if new[o][2].sum() >= 50], P.shape)
        ar2 = np.bincount(lab2.reshape(-1), minlength=int(lab2.max()) + 1); ar2[0] = 0
        for r in by[stem]:
            a = a + metric.pq_from_labels(lab, areas, gp[r.record_id]); b = b + metric.pq_from_labels(lab2, ar2, gp[r.record_id])
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True); ap.add_argument("--exp", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--name", default=None); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    name = a.name or a.samples
    D = pickle.load(open(paths.RUNS / "refine" / f"{a.samples}_samples.pkl", "rb"))
    tr = [s for s in D["samples"] if s["fold"] != a.fold]
    print(f"refiner {name} fold {a.fold}: train samples {len(tr)}", flush=True)
    dl = DataLoader(RefineSet(tr, a.exp, a.iters * a.bs), batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True)
    model = refine.build_refiner(a.encoder).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda"); t0 = time.time(); run = 0.0
    model.train()
    for it, (x, y) in enumerate(dl, 1):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        loss = loss_fn(out.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
        if it % 250 == 0:
            print(f"it {it}/{a.iters} loss {run/250:.4f} {it/(time.time()-t0):.1f} it/s", flush=True); run = 0.0
    model.eval()
    torch.save({"model": model.state_dict(), "encoder": a.encoder}, paths.RUNS / "refine" / f"{name}_fold{a.fold}.pt")
    for rthr in (0.4, 0.5, 0.6):
        base, ref = evaluate([model], a.exp, a.fold, rthr=rthr)
        print(f"[fold {a.fold}] refine_thr={rthr}  plain: {base}\n{'':26s}refined: {ref}", flush=True)
    print("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
