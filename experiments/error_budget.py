"""Where does PQ get lost?  Categorise GT instances and predictions of a simple threshold pipeline.

    python scripts/error_budget.py --exp v1_effv2s --thr 0.5 --min_area 300
"""
import sys, json, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from multiprocessing import Pool
from fil import data, paths, metric, postproc

G = {}
BINS = [0, 300, 600, 1200, 2500, 5000, 1e9]


def work(job):
    f, j, stem = job
    prob = np.ascontiguousarray(np.load(paths.RUNS / G["exp"] / f"oof_fold{f}.npy", mmap_mode="r")[j, 0])
    lab, areas, _ = postproc.cc_instances(prob, int(round(G["thr"] * 255)), G["min_area"])
    flat = lab.reshape(-1)
    gt_rows, pr_rows = [], []
    recs = G["by"][stem]
    for r in recs:
        best_pred = np.zeros(len(areas))
        for pix in G["gt_pix"][r.record_id]:
            l = flat[pix]; l = l[l > 0]
            bi, n_ov, frac_cov = 0.0, 0, 0.0
            if l.size:
                ids, inter = np.unique(l, return_counts=True)
                iou = inter / (pix.size + areas[ids] - inter)
                bi = float(iou.max()); n_ov = int((inter >= 0.1 * pix.size).sum()); frac_cov = float(inter.sum() / pix.size)
                np.maximum.at(best_pred, ids, iou)
            cx = (pix % paths.W).mean() - 1023.5; cy = (pix // paths.W).mean() - 1023.5
            gt_rows.append((pix.size, bi, n_ov, frac_cov, np.hypot(cx, cy) / 900.0, len(recs)))
        for a, b in zip(areas[1:], best_pred[1:]):
            pr_rows.append((a, b, len(recs)))
    return gt_rows, pr_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--min_area", type=int, default=300)
    a = ap.parse_args()
    G.update(exp=a.exp, thr=a.thr, min_area=a.min_area, by=data.records_by_stem(data.load_records()), gt_pix=metric.gt_pixel_cache())
    folds = [f for f in range(5) if (paths.RUNS / a.exp / f"oof_fold{f}.npy").exists()]
    jobs = [(f, j, s) for f in folds for j, s in enumerate(json.load(open(paths.RUNS / a.exp / f"oof_fold{f}_stems.json")))]
    with Pool(28) as p:
        res = p.map(work, jobs, chunksize=2)
    gt = np.array([x for r in res for x in r[0]]); pr = np.array([x for r in res for x in r[1]])
    tp = (gt[:, 1] > 0.5).sum(); S = gt[gt[:, 1] > 0.5, 1].sum()
    print(f"GT={len(gt)} pred(record-level)={len(pr)}  PQ={S/(tp+0.5*(len(pr)-tp)+0.5*(len(gt)-tp)):.4f} SQ={S/tp:.4f}")
    print("GT instances: matched %.3f | overlapped but IoU<=0.5 %.3f | untouched (IoU<0.05) %.3f" % (
        (gt[:, 1] > 0.5).mean(), ((gt[:, 1] <= 0.5) & (gt[:, 1] >= 0.05)).mean(), (gt[:, 1] < 0.05).mean()))
    part = gt[(gt[:, 1] <= 0.5) & (gt[:, 1] >= 0.05)]
    print("  of the overlapped-unmatched GT: covered by >=2 predictions %.3f | mean coverage of GT by predictions %.3f | IoU in (0.4,0.5] %.3f" % (
        (part[:, 2] >= 2).mean(), part[:, 3].mean(), (part[:, 1] > 0.4).mean()))
    print("predictions: matched %.3f | overlapped but IoU<=0.5 %.3f | no GT (IoU<0.05) %.3f" % (
        (pr[:, 1] > 0.5).mean(), ((pr[:, 1] <= 0.5) & (pr[:, 1] >= 0.05)).mean(), (pr[:, 1] < 0.05).mean()))
    print("by GT area bin: n, matched, overlapped-unmatched, untouched, mean matched IoU")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (gt[:, 0] >= lo) & (gt[:, 0] < hi)
        g = gt[s]
        print(f"  [{lo:>5.0f},{hi:>10.0f}) n={len(g):5d} {(g[:,1]>0.5).mean():.3f} {((g[:,1]<=0.5)&(g[:,1]>=0.05)).mean():.3f} {(g[:,1]<0.05).mean():.3f} {g[g[:,1]>0.5,1].mean():.3f}")
    print("by pred area bin: n, matched, overlapped-unmatched, spurious")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (pr[:, 0] >= lo) & (pr[:, 0] < hi)
        g = pr[s]
        if len(g): print(f"  [{lo:>5.0f},{hi:>10.0f}) n={len(g):5d} {(g[:,1]>0.5).mean():.3f} {((g[:,1]<=0.5)&(g[:,1]>=0.05)).mean():.3f} {(g[:,1]<0.05).mean():.3f}")
    print("by r/R ring (GT): n, matched")
    for lo, hi in [(0, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.1)]:
        s = (gt[:, 4] >= lo) & (gt[:, 4] < hi)
        print(f"  [{lo},{hi}) n={int(s.sum()):5d} {(gt[s,1]>0.5).mean():.3f}")
    print("by #annotators of the image (GT matched rate / pred matched rate):")
    for nr in (1, 2, 3):
        g = gt[gt[:, 5] == nr]; p_ = pr[pr[:, 2] == nr]
        tp = (g[:, 1] > 0.5).sum()
        print(f"  nrec={nr}: GT n={len(g)} matched={(g[:,1]>0.5).mean():.3f} | pred n={len(p_)} matched={(p_[:,1]>0.5).mean():.3f} | PQ={g[g[:,1]>0.5,1].sum()/(tp+0.5*(len(p_)-tp)+0.5*(len(g)-tp)):.4f}")


if __name__ == "__main__":
    main()
