"""Qualitative check: raw crop | annotator outlines | predicted instances (green = matched by some annotator, red = not)."""
import sys, json, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from fil import data, paths, metric, postproc

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="v1_effv2s"); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--thr", type=float, default=0.45); ap.add_argument("--min_area", type=int, default=400)
ap.add_argument("--n", type=int, default=4); ap.add_argument("--size", type=int, default=640)
a = ap.parse_args()
out = paths.ROOT / "viz"; out.mkdir(exist_ok=True)
stems_all = json.load(open(paths.CACHE / "stems.json")); idx = {s: i for i, s in enumerate(stems_all)}
by = data.records_by_stem(data.load_records()); gp = metric.gt_pixel_cache()
vst = json.load(open(paths.RUNS / a.exp / f"oof_fold{a.fold}_stems.json"))
oof = np.load(paths.RUNS / a.exp / f"oof_fold{a.fold}.npy", mmap_mode="r")
img_mm = np.load(paths.CACHE / "img.npy", mmap_mode="r")
COL = [(0, 255, 255), (255, 0, 255), (0, 200, 0)]
rng = np.random.RandomState(1)
cand = [j for j, s in enumerate(vst) if len(by[s]) == 3]
for j in rng.choice(cand, a.n, replace=False):
    stem = vst[j]; g = np.ascontiguousarray(img_mm[idx[stem]]); prob = np.ascontiguousarray(oof[j, 0])
    lab, areas, stats = postproc.cc_instances(prob, int(round(a.thr * 255)), a.min_area)
    matched = np.zeros(len(areas), bool)
    for r in by[stem]:
        for pix in gp[r.record_id]:
            l = lab.reshape(-1)[pix]; l = l[l > 0]
            if l.size:
                ids, inter = np.unique(l, return_counts=True)
                iou = inter / (pix.size + areas[ids] - inter)
                matched[ids[iou > 0.5]] = True
    S = a.size
    cx = np.array([[b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2] for r in by[stem] for b in r.anns])
    best, bxy = -1, (0, 0)
    for x0 in range(0, 2048 - S + 1, 64):
        for y0 in range(0, 2048 - S + 1, 64):
            c = ((cx[:, 0] >= x0) & (cx[:, 0] < x0 + S) & (cx[:, 1] >= y0) & (cx[:, 1] < y0 + S)).sum()
            if c > best: best, bxy = c, (x0, y0)
    x0, y0 = bxy
    raw = cv2.cvtColor(g[y0:y0 + S, x0:x0 + S], cv2.COLOR_GRAY2BGR)
    raw = cv2.convertScaleAbs(raw, alpha=1.6, beta=-60)                      # contrast stretch for viewing
    pa = raw.copy(); pb = raw.copy()
    for k, r in enumerate(by[stem]):
        for b in r.anns:
            p = np.array(b["segmentation"][0], np.float32).reshape(-1, 2) - [x0, y0]
            cv2.polylines(pa, [np.round(p).astype(np.int32)], True, COL[k], 1, cv2.LINE_AA)
    sub = lab[y0:y0 + S, x0:x0 + S]
    for i in np.unique(sub):
        if i == 0: continue
        cs, _ = cv2.findContours((sub == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(pb, cs, -1, (0, 255, 0) if matched[i] else (0, 0, 255), 1)
    heat = cv2.applyColorMap(prob[y0:y0 + S, x0:x0 + S], cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / f"pred_{stem}.jpg"), np.concatenate([np.concatenate([raw, pa], 1), np.concatenate([pb, heat], 1)], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(stem, "crop", x0, y0, "instances", len(areas) - 1, "matched", int(matched.sum()))
