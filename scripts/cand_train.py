"""Pass 2: train the candidate scorer (expected matched IoU) with cross-validation and report the
out-of-fold PQ of the PQ-optimal selection, next to plain threshold baselines.

    python scripts/cand_train.py --cand v1
"""
import sys, argparse, pickle, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from fil import paths, candidates


def pq_of(df, meta, chosen):
    """PQ of the chosen rows from the stored per-record matched IoUs."""
    c = df[chosen]
    S = TP = FP = 0.0
    for q in range(3):
        has = c.nrec.values > q
        mi = c[f"miou{q}"].values[has]
        S += mi.sum(); tp = (mi > 0).sum(); TP += tp; FP += has.sum() - tp
    FN = sum(meta[f"ngt{q}"].sum() for q in range(3)) - TP
    den = TP + 0.5 * FP + 0.5 * FN
    return S / den, S / max(TP, 1), TP / den, int(TP), int(FP), int(FN)


def run_select(df, score, lam):
    chosen = np.zeros(len(df), bool)
    for _, idx in df.groupby("img").indices.items():
        chosen[idx] = candidates.select(df.node.values[idx], df.parent.values[idx], df.k.values[idx], score[idx], lam)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--obj", default="regression")
    ap.add_argument("--leaves", type=int, default=15); ap.add_argument("--save", default=None)
    ap.add_argument("--cnn", action="store_true", help="add the cross-fitted CNN scorer outputs as descriptors")
    ap.add_argument("--target", default="official", choices=["official", "lr"],
                    help="official: matched IoU with any annotated filament; lr: only filaments with a Left/Right chirality label")
    a = ap.parse_args()
    df = pd.read_pickle(paths.RUNS / "cand" / f"{a.cand}.pkl"); meta = pd.read_pickle(paths.RUNS / "cand" / f"{a.cand}_meta.pkl")
    if a.target == "lr":      # matched IoU counted only for chirality-identified (Left/Right) ground truth
        for q in range(3):
            c = df[f"cat{q}"].fillna(0).values if f"cat{q}" in df else np.zeros(len(df))
            df[f"miou{q}"] = np.where(np.isnan(df[f"miou{q}"].values), np.nan, np.where((c >= 1) & (c <= 2), df[f"miou{q}"].values, 0.0))
            meta[f"ngt{q}"] = meta[f"nlr{q}"]
    y = np.nanmean(df[["miou0", "miou1", "miou2"]].values, 1)
    feats = list(candidates.FEATS) + [f for f in candidates.EXTRA_FEATS if f in df.columns]
    if a.cnn:
        cnn = np.load(paths.RUNS / "cand" / f"{a.cand}_cnn.npy")
        df["cnn_off"], df["cnn_lr"] = cnn[:, 0], cnn[:, 1]
        feats += ["cnn_off", "cnn_lr"]
    X = df[feats].values.astype(np.float32)
    w = df.nrec.values.astype(np.float32)
    print(f"candidates={len(df)} images={len(meta)} mean target={y.mean():.4f} positives={(y>0).mean():.3f}")

    # --- baselines straight from the table: one fixed level + minimum area
    print("baseline (single threshold, min area):")
    for lvl in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        line = []
        for ma in (0, 200, 300, 400, 600):
            ch = (df.is_var.values == 0) & np.isclose(df.lvl.values, lvl) & (df.area.values >= ma)
            line.append(f"{pq_of(df, meta, ch)[0]:.4f}")
        print(f"  t={lvl:.2f} min_area[0,200,300,400,600]: " + " ".join(line))

    params = dict(objective=a.obj, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=24)
    oof = np.zeros(len(df)); models = []
    folds = sorted(df.fold.unique())
    t = time.time()
    if len(folds) > 1:
        for f in folds:
            tr = df.fold.values != f
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr]), a.rounds)
            oof[~tr] = m.predict(X[~tr]); models.append(m)
    else:                                   # single segmentation fold available: inner 5-fold over images
        imgs = np.array(sorted(df.img.unique())); rng = np.random.RandomState(0); rng.shuffle(imgs)
        for f in range(5):
            te = np.isin(df.img.values, imgs[f::5])
            m = lgb.train(params, lgb.Dataset(X[~te], y[~te], weight=w[~te]), a.rounds)
            oof[te] = m.predict(X[te]); models.append(m)
    print(f"scorer trained in {time.time()-t:.0f}s; OOF rmse={np.sqrt(np.mean((oof-y)**2)):.4f} corr={np.corrcoef(oof,y)[0,1]:.4f}")
    lam = 0.45
    for it in range(4):
        ch = run_select(df, oof, lam)
        r = pq_of(df, meta, ch)
        print(f"  lambda={lam:.4f} -> PQ={r[0]:.4f} SQ={r[1]:.4f} RQ={r[2]:.4f} TP={r[3]} FP={r[4]} FN={r[5]} n_sel={int(ch.sum())} ")
        lam = r[0]
    for lam2 in (0.30, 0.38, 0.52, 0.60):
        print(f"  [sensitivity] lambda={lam2:.2f} -> PQ={pq_of(df, meta, run_select(df, oof, lam2))[0]:.4f}")
    # oracle upper bound of the candidate set (true targets as scores)
    ch = run_select(df, y, lam)
    print(f"  [oracle scores] PQ={pq_of(df, meta, ch)[0]:.4f}")
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in models], 0), index=feats).sort_values(ascending=False)
    print("top features by gain:", ", ".join(f"{k}:{v/imp.sum():.3f}" for k, v in imp.head(15).items()))
    if a.save:
        cf = paths.RUNS / "cand" / f"{a.cand}_cfg.json"
        cfg = json.load(open(cf)) if cf.exists() else {}
        cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()}
        full = lgb.train(params, lgb.Dataset(X, y, weight=w), a.rounds)
        pickle.dump(dict(models=[m.model_to_string() for m in models], full=full.model_to_string(), lam=lam, feats=feats,
                         cfg=cfg, target=a.target),
                    open(paths.RUNS / "cand" / f"{a.save}_scorer.pkl", "wb"))
        np.save(paths.RUNS / "cand" / f"{a.save}_oof.npy", oof)
        print("saved scorer", a.save)


if __name__ == "__main__":
    main()
