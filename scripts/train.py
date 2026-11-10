"""Train one segmentation network (one CV fold or all data) and dump out-of-fold probability maps.

Example:
    python scripts/train.py --exp v1_effv2s --fold 0
"""
import sys, json, time, math, argparse, copy
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from fil import data, paths, metric, postproc, infer
from fil.dataset import CropDataset
from fil.model import build_model


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--fold", type=int, default=0, help="validation fold; -1 = train on everything")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="tu-tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--fg_prob", type=float, default=0.7)
    ap.add_argument("--scale_min", type=float, default=0.75)
    ap.add_argument("--scale_max", type=float, default=1.33)
    ap.add_argument("--spine_w", type=float, default=0.5)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--tta", default="flip4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_oof", action="store_true")
    ap.add_argument("--infer_scales", type=float, nargs="+", default=[1.0], help="frame scale(s) used at inference")
    ap.add_argument("--native_prob", type=float, default=0.0, help="share of crops taken on the native pixel grid")
    return ap.parse_args()


def seg_loss(logit, target, dice_w=1.0):
    """BCE on soft (multi-annotator) targets + batch-level soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * target).sum() + 1.0) / (p.sum() + target.sum() + 1.0)
    return bce + dice_w * dice


@torch.no_grad()
def evaluate(model, val_idx, stems, by, gt_pix, img_mm, tta="none", thrs=(0.3, 0.4, 0.5, 0.6), min_area=300, dump=None, scales=(1.0,)):
    """PQ of threshold+connected-components instances on the validation images."""
    model.eval()
    tot = {t: metric.PQ() for t in thrs}
    for j, i in enumerate(val_idx):
        prob = infer.predict_full(model, np.ascontiguousarray(img_mm[i]), tta=tta, scales=scales)
        if dump is not None:
            dump[j] = np.clip(np.round(prob * 255), 0, 255).astype(np.uint8)
        for t in thrs:
            lab, areas, _ = postproc.cc_instances(prob[0], t, min_area)
            for r in by[stems[i]]:
                tot[t] = tot[t] + metric.pq_from_labels(lab, areas, gt_pix[r.record_id])
    return tot


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = paths.RUNS / a.exp; out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / f"args_fold{a.fold}.json", "w"), indent=1)
    log = open(out / f"log_fold{a.fold}.txt", "a")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    stems = json.load(open(paths.CACHE / "stems.json"))
    folds = data.make_folds(stems, a.folds)
    val_idx = [i for i, s in enumerate(stems) if folds[s] == a.fold]
    trn_idx = [i for i, s in enumerate(stems) if folds[s] != a.fold]
    recs = data.load_records(); by = data.records_by_stem(recs)
    gt_pix = metric.gt_pixel_cache()
    img_mm = np.load(paths.CACHE / "img.npy", mmap_mode="r")
    P(f"exp={a.exp} fold={a.fold} train={len(trn_idx)} val={len(val_idx)} args={vars(a)}")

    ds = CropDataset(trn_idx, crop=a.crop, length=a.iters * a.bs, fg_prob=a.fg_prob,
                     scale=(a.scale_min, a.scale_max), seed=a.seed, native_prob=a.native_prob)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=False, prefetch_factor=4)
    n_out = 2                                    # filament mask, spine
    model = build_model(a.arch, a.encoder, out_ch=n_out).cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    P("params (M): %.1f" % (sum(p.numel() for p in model.parameters()) / 1e6))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time(); run = {"loss": 0.0, "m": 0.0, "s": 0.0, "n": 0}
    model.train()
    for it, (x, y) in enumerate(dl, 1):
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out_ = model(x)
        out_ = out_.float()
        lm = seg_loss(out_[:, 0], y[:, 0], a.dice_w); ls = seg_loss(out_[:, 1], y[:, 1], a.dice_w)
        loss = lm + a.spine_w * ls
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run["loss"] += loss.item(); run["m"] += lm.item(); run["s"] += ls.item(); run["n"] += 1
        if it % 100 == 0:
            el = time.time() - t0
            P(f"it {it}/{a.iters} loss {run['loss']/run['n']:.4f} mask {run['m']/run['n']:.4f} spine {run['s']/run['n']:.4f} "
              f"lr {sched.get_last_lr()[0]:.2e} {it/el:.2f} it/s eta {(a.iters-it)/(it/el)/60:.1f} min")
            run = {"loss": 0.0, "m": 0.0, "s": 0.0, "n": 0}
        if a.fold >= 0 and it % a.eval_every == 0 and it < a.iters:
            res = evaluate(ema, val_idx, stems, by, gt_pix, img_mm, scales=tuple(a.infer_scales))
            P(f"[eval it {it}] " + " | ".join(f"t={t}: {r}" for t, r in res.items()))
            model.train()
        if it >= a.iters:
            break

    torch.save({"model": ema.state_dict(), "args": vars(a), "out_ch": n_out}, out / f"fold{a.fold}.pt")
    if a.fold >= 0:
        dump = None
        if not a.no_oof:
            dump = np.lib.format.open_memmap(out / f"oof_fold{a.fold}.npy", "w+", np.uint8, (len(val_idx), n_out, paths.H, paths.W))
            json.dump([stems[i] for i in val_idx], open(out / f"oof_fold{a.fold}_stems.json", "w"))
        res = evaluate(ema, val_idx, stems, by, gt_pix, img_mm, tta=a.tta, dump=dump, scales=tuple(a.infer_scales))
        if dump is not None: dump.flush()
        P(f"[final tta={a.tta}] " + " | ".join(f"t={t}: {r}" for t, r in res.items()))
    P("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
