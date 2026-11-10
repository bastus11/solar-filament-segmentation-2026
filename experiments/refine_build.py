"""Collect training samples for the stage-2 refiner from out-of-fold candidates.

    python scripts/refine_build.py --exps v1_d4 --out v1
Each sample = one tree candidate that overlaps an annotated instance (best IoU >= 0.3) plus the ids
of the annotated instances it should be refined towards.
"""
import sys, json, argparse, pickle, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool
from fil import data, paths, metric, candidates

G = {}
KEEP = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
MIN_IOU, MIN_AREA = 0.3, 100


def work(job):
    f, j, stem = job
    i = G["idx"][stem]
    acc = None
    for e in G["exps"]:
        a = np.load(paths.RUNS / e / f"oof_fold{f}.npy", mmap_mode="r")[j].astype(np.float32)
        acc = a if acc is None else acc + a
    pr = np.round(acc / len(G["exps"])).astype(np.uint8)
    img = np.ascontiguousarray(np.load(paths.CACHE / "img.npy", mmap_mode="r")[i])
    recs = G["by"][stem]
    labs, areas = [], []
    for r in recs:
        gl = np.zeros(paths.H * paths.W, np.int32); ar = [0]
        for q, pix in enumerate(G["gt_pix"][r.record_id], 1):
            gl[pix] = q; ar.append(pix.size)
        labs.append(gl.reshape(paths.H, paths.W)); areas.append(np.asarray(ar))
    rows, masks = candidates.extract(pr[0], pr[1], img, candidates.Cfg(var_levels=()), labs, areas)
    out = []
    for row, (x0, y0, m) in zip(rows, masks):
        if round(row["lvl"], 2) not in KEEP or row["area"] < MIN_AREA:
            continue
        b = [row[f"biou{q}"] for q in range(len(recs))]
        if max(b) < MIN_IOU:
            continue
        out.append(dict(img=i, fold=f, j=j, x0=x0, y0=y0, m=m, lvl=row["lvl"],
                        rec=[recs[q].record_id for q in range(len(recs)) if b[q] >= MIN_IOU],
                        gid=[row[f"gid{q}"] for q in range(len(recs)) if b[q] >= MIN_IOU]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stems = json.load(open(paths.CACHE / "stems.json"))
    G.update(exps=a.exps, idx={s: i for i, s in enumerate(stems)}, by=data.records_by_stem(data.load_records()), gt_pix=metric.gt_pixel_cache())
    folds = [f for f in range(5) if all((paths.RUNS / e / f"oof_fold{f}.npy").exists() for e in a.exps)]
    jobs = [(f, j, s) for f in folds for j, s in enumerate(json.load(open(paths.RUNS / a.exps[0] / f"oof_fold{f}_stems.json")))]
    t = time.time()
    with Pool(24) as p:
        res = p.map(work, jobs, chunksize=2)
    samples = [s for r in res for s in r]
    out = paths.RUNS / "refine"; out.mkdir(exist_ok=True)
    pickle.dump(dict(samples=samples, exps=a.exps, folds=folds), open(out / f"{a.out}_samples.pkl", "wb"))
    print(f"folds={folds} images={len(jobs)} samples={len(samples)} ({len(samples)/len(jobs):.1f}/img) in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
