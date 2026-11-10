"""Model-free check of the annotation offset: where is the image darkest relative to the annotated polygons?

For every annotated filament the mean grey level under its (rasterised) polygon is measured after
moving the polygon by a sub-pixel offset (the image is resampled instead).  Filaments are dark, so
the offset with the lowest mean grey level is where the polygons would sit best.
Result on the training set: +0.29 px in y (all six annotator groups +0.11 ... +0.31), +0.05 px in x.
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from fil import data, paths, metric

stems = json.load(open(paths.CACHE / "stems.json")); img = np.load(paths.CACHE / "img.npy", mmap_mode="r")
by = data.records_by_stem(data.load_records()); gp = metric.gt_pixel_cache()
offs = np.round(np.arange(-1.0, 1.01, 0.25), 2)
acc = {("y", d): [0.0, 0] for d in offs}; acc.update({("x", d): [0.0, 0] for d in offs})
per_group = {}
for i in range(0, len(stems), 2):
    g = img[i].astype(np.float32)
    pix = np.concatenate([p for r in by[stems[i]] for p in gp[r.record_id]])
    for ax in ("y", "x"):
        for d in offs:
            M = np.float32([[1, 0, -d if ax == "x" else 0], [0, 1, -d if ax == "y" else 0]])      # polygon moved by +d  <=>  image moved by -d
            s = cv2.warpAffine(g, M, g.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).reshape(-1)[pix]
            acc[(ax, d)][0] += float(s.sum()); acc[(ax, d)][1] += s.size
    for r in by[stems[i]]:
        k = r.annotator[:2]; pr = np.concatenate(gp[r.record_id]) if gp[r.record_id] else np.zeros(0, np.int64)
        for d in (-0.5, -0.25, 0.0, 0.25, 0.5):
            M = np.float32([[1, 0, 0], [0, 1, -d]])
            s = cv2.warpAffine(g, M, g.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).reshape(-1)[pr]
            a = per_group.setdefault((k, d), [0.0, 0]); a[0] += float(s.sum()); a[1] += s.size
for ax in ("y", "x"):
    m = np.array([acc[(ax, d)][0] / acc[(ax, d)][1] for d in offs])
    c = np.polyfit(offs, m, 2); best = -c[1] / (2 * c[0])
    print(f"polygon shifted along {ax} (down/right positive): mean grey " + " ".join(f"{d:+.2f}:{v:.2f}" for d, v in zip(offs, m)) + f"  -> minimum at {best:+.3f} px")
print("per annotator group (first two digits), minimum of a parabola through dy = -0.5..0.5:")
for k in sorted({k for k, _ in per_group}):
    ds = np.array([-0.5, -0.25, 0.0, 0.25, 0.5]); m = np.array([per_group[(k, d)][0] / per_group[(k, d)][1] for d in ds])
    c = np.polyfit(ds, m, 2); print(f"  group {k}: {-c[1] / (2 * c[0]):+.3f} px")
