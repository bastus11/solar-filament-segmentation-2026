"""Grid-search simple post-processing (hysteresis thresholds + minimum area) on out-of-fold maps.

    python scripts/sweep_pp.py --exp v1_effv2s [--folds 0 1 2]
"""
import sys, json, argparse, itertools
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from multiprocessing import Pool
from fil import data, paths, metric

T_HIGH = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_LOW = [0.0, 0.1, 0.2, 0.3]                       # t_low = t_high - d
MIN_AREA = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1500]
G = {}


def match(lab, areas, gt_pix):
    """For each GT instance: (id of the IoU>0.5 prediction or 0, IoU)."""
    flat = lab.reshape(-1); out = []
    for pix in gt_pix:
        l = flat[pix]; l = l[l > 0]
        pid, best = 0, 0.0
        if l.size:
            ids, inter = np.unique(l, return_counts=True)
            iou = inter / (pix.size + areas[ids] - inter)
            k = int(np.argmax(iou))
            if iou[k] > 0.5: pid, best = int(ids[k]), float(iou[k])
        out.append((pid, best))
    return out


def work(job):
    f, j, stem = job
    prob = np.load(paths.RUNS / G["exp"] / f"oof_fold{f}.npy", mmap_mode="r")[j, 0]
    prob = np.ascontiguousarray(prob)
    recs = G["by"][stem]
    res = np.zeros((len(T_HIGH), len(D_LOW), len(MIN_AREA), 4))
    for a, th in enumerate(T_HIGH):
        hi = prob >= int(round(th * 255))
        for b, d in enumerate(D_LOW):
            tl = th - d
            if tl < 0.05: res[a, b] = np.nan; continue
            n, lab = cv2.connectedComponents((prob >= int(round(tl * 255))).astype(np.uint8), connectivity=8)
            if d > 0: lab = lab * hi
            areas = np.bincount(lab.reshape(-1), minlength=n); areas[0] = 0
            for r in recs:
                m = match(lab, areas, G["gt_pix"][r.record_id])
                pid = np.array([x[0] for x in m], np.int64); iou = np.array([x[1] for x in m])
                parea = areas[pid]
                for c, ma in enumerate(MIN_AREA):
                    ok = (pid > 0) & (parea >= ma)
                    n_pred = int(((areas >= max(ma, 1))).sum())
                    tp = int(ok.sum())
                    res[a, b, c] += (iou[ok].sum(), tp, n_pred - len(set(pid[ok].tolist())), len(m) - tp)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--folds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    G["exp"] = a.exp
    G["by"] = data.records_by_stem(data.load_records()); G["gt_pix"] = metric.gt_pixel_cache()
    folds = a.folds if a.folds is not None else [f for f in range(5) if (paths.RUNS / a.exp / f"oof_fold{f}.npy").exists()]
    jobs = [(f, j, s) for f in folds for j, s in enumerate(json.load(open(paths.RUNS / a.exp / f"oof_fold{f}_stems.json")))]
    with Pool(28) as p:
        res = np.nansum(np.stack(p.map(work, jobs, chunksize=2)), 0)
    pq = res[..., 0] / (res[..., 1] + 0.5 * res[..., 2] + 0.5 * res[..., 3])
    print(f"exp={a.exp} folds={folds} images={len(jobs)}")
    for b, d in enumerate(D_LOW):
        print(f"\n--- t_low = t_high - {d}   (rows: t_high, cols: min_area {MIN_AREA})")
        for i, th in enumerate(T_HIGH):
            print(f"  {th:.2f} " + " ".join(f"{pq[i, b, c]:.4f}" for c in range(len(MIN_AREA))))
    order = np.dstack(np.unravel_index(np.argsort(-pq, axis=None), pq.shape))[0][:8]
    print("\nbest configs:")
    for i, b, c in order:
        r = res[i, b, c]
        print(f"  t_high={T_HIGH[i]:.2f} t_low={T_HIGH[i]-D_LOW[b]:.2f} min_area={MIN_AREA[c]:4d}  PQ={pq[i,b,c]:.4f} "
              f"SQ={r[0]/r[1]:.4f} RQ={r[1]/(r[1]+.5*r[2]+.5*r[3]):.4f} TP={int(r[1])} FP={int(r[2])} FN={int(r[3])}")


if __name__ == "__main__":
    main()
