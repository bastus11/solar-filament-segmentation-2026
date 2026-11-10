"""Does a validation image score higher when an image of (almost) the same day was in the training folds?
Timestamps from file names are used for this ANALYSIS only - never as a model input."""
import sys, json, datetime
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from fil import paths, candidates, data
name = sys.argv[1]
df = pd.read_pickle(paths.RUNS / "cand" / f"{name}.pkl"); meta = pd.read_pickle(paths.RUNS / "cand" / f"{name}_meta.pkl")
oof = np.load(paths.RUNS / "cand" / f"{name}_oof.npy")
stems = json.load(open(paths.CACHE / "stems.json")); folds = data.make_folds(stems, 5)
ts = np.array([datetime.datetime.strptime(s[:14], "%Y%m%d%H%M%S").timestamp() for s in stems]); fd = np.array([folds[s] for s in stems])
gap = np.array([np.abs(ts[fd != fd[i]] - ts[i]).min() / 3600 for i in range(len(stems))])
ch = np.zeros(len(df), bool)
for _, idx in df.groupby("img").indices.items():
    ch[idx] = candidates.select(df.node.values[idx], df.parent.values[idx], df.k.values[idx], oof[idx], 0.52)
c = df[ch]
print("gap to nearest image of another fold (hours): quartiles", np.percentile(gap, [10, 25, 50, 75, 90]).round(1))
for lo, hi in [(0, 6), (6, 24), (24, 48), (48, 96), (96, 240), (240, 1e9)]:
    imgs = np.flatnonzero((gap >= lo) & (gap < hi))
    cc = c[c.img.isin(imgs)]; mm = meta[meta.img.isin(imgs)]
    S = TP = FP = 0.
    for q in range(3):
        has = cc.nrec.values > q; mi = cc[f"miou{q}"].values[has]; S += mi.sum(); tp = (mi > 0).sum(); TP += tp; FP += has.sum() - tp
    FN = sum(mm[f"ngt{q}"].sum() for q in range(3)) - TP
    print(f"  gap [{lo:>4.0f},{hi:>6.0f}) h: images={len(imgs):3d}  PQ={S/(TP+.5*FP+.5*FN):.4f} SQ={S/max(TP,1):.4f} RQ={TP/(TP+.5*FP+.5*FN):.4f}")
