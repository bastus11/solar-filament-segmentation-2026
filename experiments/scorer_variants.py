"""Quick scorer ablations on a candidate table (fold-wise CV)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from fil import paths, candidates
name = sys.argv[1]
df0 = pd.read_pickle(paths.RUNS / "cand" / f"{name}.pkl"); meta = pd.read_pickle(paths.RUNS / "cand" / f"{name}_meta.pkl")
NG = sum(meta[f"ngt{q}"].sum() for q in range(3))
def H0(df, ch):
    c = df[ch]; S = TP = FP = 0.
    for q in range(3):
        has = c.nrec.values > q; m_ = c[f"miou{q}"].values[has]; S += m_.sum(); tp = (m_ > 0).sum(); TP += tp; FP += has.sum() - tp
    return S / (TP + .5 * FP + .5 * (NG - TP))
def sel(df, score, lam):
    ch = np.zeros(len(df), bool)
    for _, idx in df.groupby("img").indices.items():
        ch[idx] = candidates.select(df.node.values[idx], df.parent.values[idx], df.k.values[idx], score[idx], lam)
    return ch
def run(df, feats, tag, params=None, rounds=600, target="mean", weight=True):
    mi = df[["miou0", "miou1", "miou2"]].values
    y = np.nanmean(mi, 1) if target == "mean" else np.nanmax(mi, 1)
    X = df[feats].values.astype(np.float32); w = df.nrec.values.astype(np.float32) if weight else np.ones(len(df), np.float32)
    p = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
    if params: p.update(params)
    oof = np.zeros(len(df))
    for f in sorted(df.fold.unique()):
        tr = df.fold.values != f
        oof[~tr] = lgb.train(p, lgb.Dataset(X[tr], y[tr], weight=w[tr]), rounds).predict(X[~tr])
    lam = 0.45
    for _ in range(3):
        r = H0(df, sel(df, oof, lam)); lam = r
    best = max(H0(df, sel(df, oof, l)) for l in (lam - 0.06, lam - 0.03, lam, lam + 0.03, lam + 0.06))
    print(f"{tag:46s} rows={len(df):6d} PQ(fixed point)={r:.4f}  best over lambda={best:.4f}", flush=True)
    return oof
F = candidates.FEATS
run(df0, F, "baseline (all candidates, all features)")
plain = df0[df0.is_var == 0].reset_index(drop=True)
run(plain, F, "no grouped candidates")
run(plain, [f for f in F if not f.startswith("img_")], "no grouped, no image-level features")
run(plain, [f for f in F if f not in ("cx", "cy")], "no grouped, no cx/cy")
run(plain, F, "no grouped, leaves 15 / 400 rounds", dict(num_leaves=15), 400)
run(plain, F, "no grouped, leaves 63, min_leaf 100", dict(num_leaves=63, min_data_in_leaf=100))
run(plain, F, "no grouped, lr 0.01 x 1500", dict(learning_rate=0.01), 1500)
run(plain, F, "no grouped, xentropy objective", dict(objective="cross_entropy"))
run(plain, F, "no grouped, huber", dict(objective="huber", alpha=0.3))
run(plain, F, "no grouped, unweighted", weight=False)
# coarser ladder: keep every second level
keep = plain[np.isin(np.round(plain.lvl, 2), [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])].reset_index(drop=True)
