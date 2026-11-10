"""Pass 1 of the candidate pipeline: build the candidate table (descriptors + IoU targets) from
out-of-fold probability maps.

    python scripts/cand_build.py --exps v1_effv2s --out v1
Several experiments are averaged (they must share the fold split).
"""
import sys, json, argparse, time, dataclasses
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from multiprocessing import Pool
from fil import data, paths, metric, candidates

G = {}


def load_prob(exps, f, j):
    acc = None
    for e in exps:
        a = np.load(paths.RUNS / e / f"oof_fold{f}.npy", mmap_mode="r")[j].astype(np.float32)
        acc = a if acc is None else acc + a
    return np.round(acc / len(exps)).astype(np.uint8)


def work(job):
    f, j, stem = job
    i = G["idx"][stem]
    pr = load_prob(G["exps"], f, j)
    img = np.ascontiguousarray(np.load(paths.CACHE / "img.npy", mmap_mode="r")[i])
    recs = G["by"][stem]
    labs, areas = [], []
    for r in recs:
        gl = np.zeros(paths.H * paths.W, np.int32); ar = [0]
        for q, pix in enumerate(G["gt_pix"][r.record_id], 1):
            gl[pix] = q; ar.append(pix.size)
        labs.append(gl.reshape(paths.H, paths.W)); areas.append(np.asarray(ar))
    rows, masks = candidates.extract(pr[0], pr[1], img, G["cfg"], labs, areas, E=pr[2] if pr.shape[0] > 2 else None)
    df = pd.DataFrame(rows)
    df["img"] = i; df["fold"] = f; df["nrec"] = len(recs)
    # chirality class of the best-overlapping annotated instance (analysis only, never a model input)
    cats = [np.array([0] + [a["category_id"] for a in r.anns]) for r in recs]
    for q in range(len(recs)):
        if len(df):
            df[f"cat{q}"] = cats[q][df[f"gid{q}"].values]
    for q in range(3):
        df[f"ngt{q}"] = len(G["gt_pix"][recs[q].record_id]) if q < len(recs) else 0
        if q >= len(recs) and len(df):
            df[f"biou{q}"] = np.nan; df[f"miou{q}"] = np.nan; df[f"gid{q}"] = 0
    meta = dict(img=i, fold=f, nrec=len(recs), **{f"ngt{q}": (len(G["gt_pix"][recs[q].record_id]) if q < len(recs) else 0) for q in range(3)},
                **{f"nlr{q}": (int((cats[q][1:] <= 2).sum()) if q < len(recs) else 0) for q in range(3)})
    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--no_fill", action="store_true")
    ap.add_argument("--levels", type=float, nargs="*", default=None, help="threshold ladder (default: candidates.LEVELS)")
    a = ap.parse_args()
    stems = json.load(open(paths.CACHE / "stems.json"))
    G.update(exps=a.exps, idx={s: i for i, s in enumerate(stems)}, by=data.records_by_stem(data.load_records()),
             gt_pix=metric.gt_pixel_cache(), cfg=candidates.Cfg(fill_holes=not a.no_fill, **({"levels": tuple(a.levels)} if a.levels else {})))
    folds = a.folds if a.folds is not None else [f for f in range(5) if all((paths.RUNS / e / f"oof_fold{f}.npy").exists() for e in a.exps)]
    jobs = [(f, j, s) for f in folds for j, s in enumerate(json.load(open(paths.RUNS / a.exps[0] / f"oof_fold{f}_stems.json")))]
    t = time.time()
    with Pool(28) as p:
        res = p.map(work, jobs, chunksize=1)
    df = pd.concat([r[0] for r in res], ignore_index=True)
    meta = pd.DataFrame([r[1] for r in res])
    out = paths.RUNS / "cand"; out.mkdir(exist_ok=True)
    df.to_pickle(out / f"{a.out}.pkl"); meta.to_pickle(out / f"{a.out}_meta.pkl")
    json.dump(dataclasses.asdict(G["cfg"]), open(out / f"{a.out}_cfg.json", "w"))
    print(f"images={len(jobs)} candidates={len(df)} ({len(df)/len(jobs):.0f}/img) in {time.time()-t:.0f}s -> {out / (a.out + '.pkl')}")


if __name__ == "__main__":
    main()
