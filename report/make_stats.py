"""Numbers quoted in the report: out-of-fold evaluation of the final system (lambda = 0.60).

Re-builds the selected instances of every training image from the out-of-fold maps and the
cross-fitted scorer predictions (both produced by scripts/run_ensemble.sh), then collects
per-record statistics, bootstrap intervals and the one-to-many / many-to-one relations.
Writes report/stats.pkl.
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from multiprocessing import Pool
from fil import data, paths, metric, candidates

EXP, TAB, LAM = "v1_effv2s_d4", "v1e", 0.60
G = {}


def work(job):
    f, j, stem, i, rows_idx = job
    pr = np.ascontiguousarray(np.load(paths.RUNS / EXP / f"oof_fold{f}.npy", mmap_mode="r")[j])
    img = np.ascontiguousarray(np.load(paths.CACHE / "img.npy", mmap_mode="r")[i])
    rows, masks = candidates.extract(pr[0], pr[1], img)
    assert len(rows) == len(rows_idx), (stem, len(rows), len(rows_idx))
    if rows:
        df = pd.DataFrame(rows)
        score = G["oof"][rows_idx]
        ch = candidates.select(df.node.values, df.parent.values, df.k.values, score, LAM)
        idx = np.flatnonzero(ch); idx = idx[np.argsort(-score[idx])]
        lab, stats = candidates.paint(masks, idx, img.shape)
    else:
        lab, stats = np.zeros(img.shape, np.int32), np.zeros((0, 4), np.int64)
    n = len(stats); area = np.bincount(lab.reshape(-1), minlength=n + 1); area[0] = 0
    flat = lab.reshape(-1)
    out = dict(stem=stem, fold=f, nrec=len(G["by"][stem]), n_pred=n, pred_area=area[1:].tolist(), recs=[])
    for r in G["by"][stem]:
        gp = G["gt_pix"][r.record_id]
        best = np.zeros(len(gp)); novl = np.zeros(len(gp), int); garea = np.array([p.size for p in gp])
        pbest = np.zeros(n + 1); pn = np.zeros(n + 1, int)
        for q, pix in enumerate(gp):
            l = flat[pix]; l = l[l > 0]
            if l.size:
                ids, inter = np.unique(l, return_counts=True)
                iou = inter / (pix.size + area[ids] - inter)
                best[q] = iou.max(); novl[q] = len(ids)
                np.maximum.at(pbest, ids, iou); pn[ids] += 1
        cx = np.array([(p % paths.W).mean() - 1023.5 for p in gp]); cy = np.array([(p // paths.W).mean() - 1023.5 for p in gp])
        out["recs"].append(dict(annotator=r.annotator, gt_area=garea, gt_best=best, gt_novl=novl, gt_r=np.hypot(cx, cy) / 900.0,
                                cat=np.array([a["category_id"] for a in r.anns]), pred_best=pbest[1:], pred_novl=pn[1:]))
    return out


def pq_from(res):
    S = TP = FP = FN = 0.0
    for r in res:
        for rec in r["recs"]:
            m = rec["gt_best"] > 0.5
            S += rec["gt_best"][m].sum(); TP += m.sum(); FN += (~m).sum(); FP += (rec["pred_best"] <= 0.5).sum()
    return S / (TP + 0.5 * FP + 0.5 * FN), S / TP, TP / (TP + 0.5 * FP + 0.5 * FN), int(TP), int(FP), int(FN)


def main():
    stems = json.load(open(paths.CACHE / "stems.json")); idx = {s: i for i, s in enumerate(stems)}
    df = pd.read_pickle(paths.RUNS / "cand" / f"{TAB}.pkl")
    G.update(oof=np.load(paths.RUNS / "cand" / f"{TAB}_oof.npy"), by=data.records_by_stem(data.load_records()), gt_pix=metric.gt_pixel_cache())
    rows_of = df.groupby("img").indices
    jobs = [(f, j, s, idx[s], rows_of.get(idx[s], np.zeros(0, np.int64))) for f in range(5)
            for j, s in enumerate(json.load(open(paths.RUNS / EXP / f"oof_fold{f}_stems.json")))]
    with Pool(24) as p:
        res = p.map(work, jobs, chunksize=2)
    pickle.dump(res, open(Path(__file__).parent / "stats.pkl", "wb"))
    pq = pq_from(res)
    print("final system, pooled OOF: PQ=%.4f SQ=%.4f RQ=%.4f TP=%d FP=%d FN=%d" % pq)
    rng = np.random.default_rng(0)
    boot = [pq_from([res[k] for k in rng.integers(0, len(res), len(res))])[:3] for _ in range(1000)]
    b = np.array(boot)
    print("bootstrap over images (1000x): PQ sd %.4f [%.4f, %.4f] | SQ sd %.4f | RQ sd %.4f" % (b[:, 0].std(), *np.percentile(b[:, 0], [2.5, 97.5]), b[:, 1].std(), b[:, 2].std()))
    print("per fold PQ:", [round(pq_from([r for r in res if r["fold"] == f])[0], 4) for f in range(5)])
    print("by #annotators PQ:", {n: round(pq_from([r for r in res if r["nrec"] == n])[0], 4) for n in (1, 2, 3)})
    gb = np.concatenate([rec["gt_best"] for r in res for rec in r["recs"]]); gn = np.concatenate([rec["gt_novl"] for r in res for rec in r["recs"]])
    pb = np.concatenate([rec["pred_best"] for r in res for rec in r["recs"]]); pn = np.concatenate([rec["pred_novl"] for r in res for rec in r["recs"]])
    ga = np.concatenate([rec["gt_area"] for r in res for rec in r["recs"]]); gr = np.concatenate([rec["gt_r"] for r in res for rec in r["recs"]])
    print("GT -> #overlapping predictions: 0: %.3f  1: %.3f  >=2: %.3f" % ((gn == 0).mean(), (gn == 1).mean(), (gn >= 2).mean()))
    print("prediction -> #overlapping GT (per record): 0: %.3f  1: %.3f  >=2: %.3f" % ((pn == 0).mean(), (pn == 1).mean(), (pn >= 2).mean()))
    print("GT matched %.3f | overlapped, IoU<=0.5 %.3f | no overlap %.3f ; matched IoU mean %.3f median %.3f; Dice of matched mean %.3f" % (
        (gb > 0.5).mean(), ((gb <= 0.5) & (gn > 0)).mean(), (gn == 0).mean(), gb[gb > 0.5].mean(), np.median(gb[gb > 0.5]), (2 * gb[gb > 0.5] / (1 + gb[gb > 0.5])).mean()))
    print("predictions matched %.3f | overlapping, unmatched %.3f | spurious %.3f" % ((pb > 0.5).mean(), ((pb <= 0.5) & (pn > 0)).mean(), (pn == 0).mean()))
    for lo, hi in [(0, 300), (300, 600), (600, 1200), (1200, 2500), (2500, 5000), (5000, 1e9)]:
        s = (ga >= lo) & (ga < hi); print(f"  GT area [{lo},{hi}): n={int(s.sum())} matched={(gb[s] > 0.5).mean():.3f}")
    for lo, hi in [(0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.2)]:
        s = (gr >= lo) & (gr < hi); print(f"  r/R [{lo},{hi}): n={int(s.sum())} matched={(gb[s] > 0.5).mean():.3f}")
    print("predicted filaments per image: mean %.2f ; GT per record: mean %.2f" % (np.mean([r["n_pred"] for r in res]), len(gb) / sum(r["nrec"] for r in res)))


if __name__ == "__main__":
    main()
