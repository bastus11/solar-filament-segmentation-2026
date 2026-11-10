"""Turn test probability maps into a submission file.

Simple mode (threshold + connected components):
    python scripts/make_sub.py --prob v1 --out v1_simple --thr 0.5 --min_area 300
Scored mode (candidate tree + learned scorer + PQ-optimal selection):
    python scripts/make_sub.py --prob v1 --out v1_scored --scorer v1 [--lam 0.47]
"""
import sys, json, argparse, pickle, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from fil import paths, postproc, rle, candidates

G = {}


def work(i):
    pr = np.ascontiguousarray(np.load(paths.RUNS / "test" / f"{G['prob']}.npy", mmap_mode="r")[i])
    a = G["args"]
    if a.scorer is None:
        tl = a.thr_low if a.thr_low is not None else a.thr
        n, lab = cv2.connectedComponents((pr[0] >= int(round(tl * 255))).astype(np.uint8), connectivity=8)
        if tl < a.thr:
            lab = lab * (pr[0] >= int(round(a.thr * 255)))
        areas = np.bincount(lab.reshape(-1), minlength=n); areas[0] = 0
        keep = np.flatnonzero(areas >= max(a.min_area, 1))
        remap = np.zeros(n, np.int32); remap[keep] = np.arange(1, len(keep) + 1)
        lab = remap[lab]
        return rle.labels_to_rles(lab)
    import lightgbm as lgb
    img = np.ascontiguousarray(np.load(paths.CACHE / "test_img.npy", mmap_mode="r")[i])
    rows, masks = candidates.extract(pr[0], pr[1], img, candidates.Cfg(**G["scorer"].get("cfg", {})), E=pr[2] if pr.shape[0] > 2 else None)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if "boosters" not in G:
        G["boosters"] = [lgb.Booster(model_str=s) for s in G["scorer"]["models"]]
    X = df[G["scorer"]["feats"]].values.astype(np.float32)
    score = np.mean([b.predict(X) for b in G["boosters"]], 0)
    ch = candidates.select(df.node.values, df.parent.values, df.k.values, score, G["lam"])
    idx = np.flatnonzero(ch)
    idx = idx[np.argsort(-score[idx])]
    lab, stats = candidates.paint(masks, idx, pr[0].shape)
    return rle.labels_to_rles(lab, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--thr_low", type=float, default=None)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--scorer", default=None); ap.add_argument("--lam", type=float, default=None)
    a = ap.parse_args()
    G["prob"] = a.prob; G["args"] = a
    if a.scorer:
        G["scorer"] = pickle.load(open(paths.RUNS / "cand" / f"{a.scorer}_scorer.pkl", "rb"))
        G["lam"] = a.lam if a.lam is not None else G["scorer"]["lam"]
        print("scorer", a.scorer, "lambda", G["lam"])
    tstems = json.load(open(paths.CACHE / "test_stems.json"))
    t = time.time()
    with Pool(28) as p:
        res = p.map(work, range(len(tstems)), chunksize=1)
    pred = {s: r for s, r in zip(tstems, res) if r}
    paths.SUBS.mkdir(exist_ok=True)
    f = paths.SUBS / f"{a.out}.csv"
    rle.write_submission(pred, f)
    info = rle.check_submission(f, set(tstems))
    print(f"wrote {f}: {info} (images without predictions: {len(tstems) - len(pred)}) in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
