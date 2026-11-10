"""Is there a systematic sub-pixel offset between annotated polygons and the predicted masks?"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from multiprocessing import Pool
from fil import data, paths, metric, postproc
EXP = sys.argv[1]; THR = float(sys.argv[2]); MINA = int(sys.argv[3])
by = data.records_by_stem(data.load_records()); gp = metric.gt_pixel_cache()
SHIFTS = [(0, 0), (0, -0.25), (0, -0.5), (0, 0.25), (0, 0.5), (-0.25, 0), (0.25, 0), (-0.5, 0), (0.5, 0), (-0.25, -0.25), (0.25, -0.25), (0, -0.35), (0, -0.15)]
def work(job):
    f, j, stem = job
    prob = np.ascontiguousarray(np.load(paths.RUNS / EXP / f"oof_fold{f}.npy", mmap_mode="r")[j, 0])
    out = []
    for dx, dy in SHIFTS:
        p = prob if (dx, dy) == (0, 0) else cv2.warpAffine(prob, np.float32([[1, 0, dx], [0, 1, dy]]), prob.shape[::-1], flags=cv2.INTER_LINEAR)
        lab, areas, _ = postproc.cc_instances(p, int(round(THR * 255)), MINA)
        t = metric.PQ()
        for r in by[stem]: t = t + metric.pq_from_labels(lab, areas, gp[r.record_id])
        out.append(t)
    return out
if __name__ == "__main__":
    folds = [f for f in range(5) if (paths.RUNS / EXP / f"oof_fold{f}.npy").exists()]
    jobs = [(f, j, s) for f in folds for j, s in enumerate(json.load(open(paths.RUNS / EXP / f"oof_fold{f}_stems.json")))]
    with Pool(20) as p: res = p.map(work, jobs, chunksize=2)
    print(f"{EXP} folds={folds} images={len(jobs)}  (dx, dy) = shift applied to the predicted map in pixels; dy<0 = up")
    base = None
    for k, sh in enumerate(SHIFTS):
        t = metric.PQ()
        for r in res: t = t + r[k]
        base = base or t.pq
        print(f"  shift {str(sh):16s} {t}   delta={t.pq-base:+.4f}")
