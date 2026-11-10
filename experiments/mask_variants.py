"""Does a global morphological tweak of the predicted masks improve PQ? (fold-0 OOF, simple pipeline)"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from multiprocessing import Pool
from fil import data, paths, metric, postproc, candidates
EXP, THR, MINA = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
by = data.records_by_stem(data.load_records()); gp = metric.gt_pixel_cache()
E3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); E5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); E7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)); E11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
def hull(m):
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    o = np.zeros_like(m); cv2.fillConvexPoly(o, cv2.convexHull(np.concatenate(c)), 1); return o
def poly(m, eps):
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    o = np.zeros_like(m)
    for cc in c: cv2.fillPoly(o, [cv2.approxPolyDP(cc, eps, True)], 1)
    return o
VAR = {
 "orig": lambda m: m, "fill": candidates.fill_holes,
 "dil3": lambda m: cv2.dilate(m, E3), "ero3": lambda m: cv2.erode(m, E3),
 "close5": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, E5), "close7": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, E7),
 "close11": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, E11), "open3": lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, E3),
 "open5": lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, E5),
 "poly1.0": lambda m: poly(m, 1.0), "poly2.0": lambda m: poly(m, 2.0),
 "blur": lambda m: (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8),
 "close7+fill": lambda m: candidates.fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, E7)),
}
def work(job):
    j, stem = job
    prob = np.ascontiguousarray(np.load(paths.RUNS / EXP / "oof_fold0.npy", mmap_mode="r")[j, 0])
    lab, areas, stats = postproc.cc_instances(prob, int(round(THR * 255)), MINA)
    out = {}
    P = 14
    for name, fn in VAR.items():
        new = np.zeros_like(lab)
        for i in range(1, len(areas)):
            x, y, w, h = (int(v) for v in stats[i - 1][:4])
            x0, y0, x1, y1 = max(0, x - P), max(0, y - P), min(2048, x + w + P), min(2048, y + h + P)
            m = fn((lab[y0:y1, x0:x1] == i).astype(np.uint8))
            v = new[y0:y1, x0:x1]; v[(m > 0) & (v == 0)] = i
        ar = np.bincount(new.reshape(-1), minlength=len(areas)); ar[0] = 0
        tot = metric.PQ()
        for r in by[stem]: tot = tot + metric.pq_from_labels(new, ar, gp[r.record_id])
        out[name] = tot
    return out
if __name__ == "__main__":
    vst = json.load(open(paths.RUNS / EXP / "oof_fold0_stems.json"))
    with Pool(16) as p: res = p.map(work, list(enumerate(vst)), chunksize=2)
    for name in VAR:
        t = metric.PQ()
        for r in res: t = t + r[name]
        print(f"{name:12s} {t}")
