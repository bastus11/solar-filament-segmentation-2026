"""Average the out-of-fold maps of several experiments into a new pseudo-experiment (first 2 channels).

    python scripts/avg_oof.py --exps v1_effv2s_d4 v5_regnety040_d4 --out ens_v1v5 [--folds 0]
"""
import sys, json, argparse, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from fil import paths

ap = argparse.ArgumentParser()
ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--weights", type=float, nargs="*", default=None)
a = ap.parse_args()
w = np.array(a.weights if a.weights else [1.0] * len(a.exps)); w = w / w.sum()
folds = a.folds if a.folds is not None else [f for f in range(5) if all((paths.RUNS / e / f"oof_fold{f}.npy").exists() for e in a.exps)]
out = paths.RUNS / a.out; out.mkdir(exist_ok=True, parents=True)
for f in folds:
    srcs = [np.load(paths.RUNS / e / f"oof_fold{f}.npy", mmap_mode="r") for e in a.exps]
    n = srcs[0].shape[0]
    dst = np.lib.format.open_memmap(out / f"oof_fold{f}.npy", "w+", np.uint8, (n, 2, paths.H, paths.W))
    for j in range(n):
        acc = sum(wi * s[j, :2].astype(np.float32) for wi, s in zip(w, srcs))
        dst[j] = np.round(acc).astype(np.uint8)
    dst.flush()
    shutil.copy(paths.RUNS / a.exps[0] / f"oof_fold{f}_stems.json", out / f"oof_fold{f}_stems.json")
    print(f"{a.out} fold {f}: averaged {len(a.exps)} experiments, {n} images", flush=True)
