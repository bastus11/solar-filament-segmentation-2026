"""Label-noise ceiling analysis: how well do annotators agree under the competition metric?"""
import sys, itertools, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from pycocotools import mask as mu
from fil import data, metric, paths

recs = data.load_records()
gt = data.cached_gt_rles(recs)
by = data.records_by_stem(recs)
print("records", len(recs), "images", len(by))

# 1) within-record overlaps between instances
n_pairs = n_ov = 0; ov_frac = []
for rid, rl in gt.items():
    if len(rl) < 2: continue
    iou = np.asarray(mu.iou(rl, rl, [0]*len(rl)))
    area = mu.area(rl)
    for i, j in itertools.combinations(range(len(rl)), 2):
        n_pairs += 1
        if iou[i, j] > 0:
            n_ov += 1
            inter = iou[i, j] * (area[i] + area[j]) / (1 + iou[i, j])
            ov_frac.append(inter / min(area[i], area[j]))
ov_frac = np.array(ov_frac)
print(f"within-record overlapping instance pairs: {n_ov} of {n_pairs}; "
      f"overlap/min-area pct: {np.percentile(ov_frac,[50,90,99]).round(3) if len(ov_frac) else None}; "
      f">20%: {(ov_frac>0.2).sum()}")

# 2) inter-annotator PQ (A predicts B), pooled over ordered pairs
tot = metric.PQ()
for stem, rs in by.items():
    for a, b in itertools.permutations(rs, 2):
        tot = tot + metric.pq_record(gt[b.record_id], gt[a.record_id])
print("inter-annotator (A as prediction for B):", tot)

# 3) 'copy one annotator' score against ALL records of the image (what a label leak would give)
for pick in ["first", "largest"]:
    tot = metric.PQ()
    for stem, rs in by.items():
        p = rs[0] if pick == "first" else max(rs, key=lambda r: len(r.anns))
        for b in rs:
            tot = tot + metric.pq_record(gt[b.record_id], gt[p.record_id])
    print(f"copy one annotator ({pick}) vs all records of all images:", tot)

# 4) oracle: union of one record -> connected components, scored against the same record
tot = metric.PQ(); tot_all = metric.PQ()
for stem, rs in by.items():
    for r in rs:
        m = data.decode(gt[r.record_id]).max(0) if gt[r.record_id] else np.zeros((2048,2048),np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        pr = [mu.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in range(1, n)]
        tot = tot + metric.pq_record(gt[r.record_id], pr)
print("oracle: own union mask -> connected components vs own record:", tot)

# 5) consensus oracles on multi-annotator images: mean of union masks thresholded -> CC
for thr_name, thr in [("any(union)", 0.01), ("majority(>=0.5)", 0.5), ("all", 0.99)]:
    tot = metric.PQ()
    for stem, rs in by.items():
        if len(rs) < 2: continue
        ms = np.stack([data.decode(gt[r.record_id]).max(0) for r in rs]).astype(np.float32).mean(0)
        m = (ms >= thr).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        pr = [mu.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in range(1, n)]
        for r in rs:
            tot = tot + metric.pq_record(gt[r.record_id], pr)
    print(f"consensus oracle [{thr_name}] -> CC vs each record (multi-annot images only):", tot)

# 6) inter-annotator recall by area bin
bins = [0, 300, 500, 800, 1200, 2000, 4000, 8000, 1e9]
hit = collections.Counter(); cnt = collections.Counter(); ious = collections.defaultdict(list)
for stem, rs in by.items():
    for a, b in itertools.permutations(rs, 2):
        ga, gb = gt[a.record_id], gt[b.record_id]
        if not ga: continue
        iou = metric.iou_matrix(ga, gb)   # rows a
        area = mu.area(ga)
        best = iou.max(1) if iou.shape[1] else np.zeros(len(ga))
        for ar, bi in zip(area, best):
            k = np.searchsorted(bins, ar, side="right") - 1
            cnt[k] += 1; hit[k] += bi > 0.5
            if bi > 0.5: ious[k].append(bi)
print("inter-annotator match rate by area bin (IoU>0.5) and mean matched IoU:")
for k in range(len(bins)-1):
    if cnt[k]:
        print(f"  area [{bins[k]:>6.0f},{bins[k+1]:>10.0f}): n={cnt[k]:5d} matched={hit[k]/cnt[k]:.3f} meanIoU={np.mean(ious[k]) if ious[k] else 0:.3f}")

# 7) instance count disagreement
diffs = []
for stem, rs in by.items():
    if len(rs) >= 2:
        c = [len(r.anns) for r in rs]; diffs.append((min(c), max(c)))
diffs = np.array(diffs)
print("multi-annot images: mean min-count %.2f mean max-count %.2f" % (diffs[:,0].mean(), diffs[:,1].mean()))
