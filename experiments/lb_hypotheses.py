"""Decision matrix: selection strategies (scorer target x lambda) vs. candidate definitions of the hidden leaderboard metric."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from fil import paths, candidates
name = sys.argv[1]
df = pd.read_pickle(paths.RUNS / "cand" / f"{name}.pkl"); meta = pd.read_pickle(paths.RUNS / "cand" / f"{name}_meta.pkl")
B = df[["biou0", "biou1", "biou2"]].values                  # best IoU per record (nan if record absent)
C = np.stack([df.get(f"cat{q}", pd.Series(np.zeros(len(df)))).fillna(0).values for q in range(3)], 1)
has = ~np.isnan(B)
Bz = np.nan_to_num(B)
NG = np.array([meta[f"ngt{q}"].sum() for q in range(3)]).sum(); NLR = np.array([meta[f"nlr{q}"].sum() for q in range(3)]).sum()
M50 = np.where(Bz > 0.5, Bz, 0); M60 = np.where(Bz > 0.6, Bz, 0); MLR = np.where((Bz > 0.5) & (C >= 1) & (C <= 2), Bz, 0)
def pq_rec(ch, M, n_gt):      # per-record pooling (official structure) with matched-IoU matrix M
    m = M[ch]; h = has[ch]; S = m.sum(); TP = (m > 0).sum(); FP = h.sum() - TP
    return S / (TP + 0.5 * FP + 0.5 * (n_gt - TP))
def pq_pool(ch):              # annotators pooled into one GT set per image, one-to-one
    b = M50[ch].max(1); TP = (b > 0).sum(); return b.sum() / (TP + 0.5 * (ch.sum() - TP) + 0.5 * (NG - TP))
METRICS = {"official": lambda ch: pq_rec(ch, M50, NG), "pooled1-1": pq_pool, "IoU>0.6": lambda ch: pq_rec(ch, M60, NG), "L/R only": lambda ch: pq_rec(ch, MLR, NLR)}
TARGETS = {"official": np.nanmean(np.where(has, M50, np.nan), 1), "pooled1-1": M50.max(1), "IoU>0.6": np.nanmean(np.where(has, M60, np.nan), 1), "L/R only": np.nanmean(np.where(has, MLR, np.nan), 1)}
X = df[candidates.FEATS].values.astype(np.float32); w = df.nrec.values.astype(np.float32)
P = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
def oof(y):
    o = np.zeros(len(df))
    for f in sorted(df.fold.unique()):
        tr = df.fold.values != f
        o[~tr] = lgb.train(P, lgb.Dataset(X[tr], y[tr], weight=w[tr]), 500).predict(X[~tr])
    return o
def sel(score, lam):
    ch = np.zeros(len(df), bool)
    for _, idx in df.groupby("img").indices.items():
        ch[idx] = candidates.select(df.node.values[idx], df.parent.values[idx], df.k.values[idx], score[idx], lam)
    return ch
print("rows: selection strategy (scorer target, lambda)   cols: PQ under each candidate leaderboard metric")
print(f"{'strategy':34s} n_sel   " + "  ".join(f"{k:>9s}" for k in METRICS))
for tn, y in TARGETS.items():
    s = oof(y)
    for lam in (0.30, 0.38, 0.47, 0.60, 0.72, 0.85, 1.00):
        ch = sel(s, lam)
        print(f"target={tn:10s} lambda={lam:.2f}       {int(ch.sum()):5d}   " + "  ".join(f"{fn(ch):9.4f}" for fn in METRICS.values()), flush=True)
