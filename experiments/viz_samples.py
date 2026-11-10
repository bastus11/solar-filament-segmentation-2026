"""Render a few annotated samples (full disk + zoomed crops) for visual inspection."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2, os
from fil import data, paths

out = paths.ROOT / "viz"; out.mkdir(exist_ok=True)
recs = data.load_records(); by = data.records_by_stem(recs)
COL = [(0, 255, 255), (255, 0, 255), (0, 255, 0)]   # BGR per annotator
stems3 = [s for s, r in by.items() if len(r) == 3]
stems1 = [s for s, r in by.items() if len(r) == 1]
rng = np.random.RandomState(0)
pick = list(rng.choice(stems3, 3, replace=False)) + list(rng.choice(stems1, 1, replace=False))
for stem in pick:
    img = cv2.imread(str(paths.TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_UNCHANGED)
    print(stem, img.shape, img.dtype, "min/max/mean", img.min(), img.max(), round(float(img.mean()), 1), "n_rec", len(by[stem]))
    g = img if img.ndim == 2 else img[..., 0]
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for k, r in enumerate(by[stem]):
        for a in r.anns:
            p = np.array(a["segmentation"][0], np.float32).reshape(-1, 2)
            cv2.polylines(vis, [np.round(p).astype(np.int32)], True, COL[k], 1, cv2.LINE_AA)
            s = np.array(a["spine"], np.float32).reshape(-1, 2)
            cv2.polylines(vis, [np.round(s).astype(np.int32)], False, tuple(int(c * 0.6) for c in COL[k]), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / f"{stem}_full.jpg"), cv2.resize(vis, (1024, 1024), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # zoom on the densest 512 window (2x upsampled) with and without overlay
    cx = np.array([[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2] for r in by[stem] for a in r.anns])
    best, bxy = -1, (0, 0)
    for x0 in range(0, 2048 - 512 + 1, 128):
        for y0 in range(0, 2048 - 512 + 1, 128):
            c = ((cx[:, 0] >= x0) & (cx[:, 0] < x0 + 512) & (cx[:, 1] >= y0) & (cx[:, 1] < y0 + 512)).sum()
            if c > best: best, bxy = c, (x0, y0)
    x0, y0 = bxy
    raw = cv2.cvtColor(g[y0:y0 + 512, x0:x0 + 512], cv2.COLOR_GRAY2BGR)
    both = np.concatenate([raw, vis[y0:y0 + 512, x0:x0 + 512]], 1)
    cv2.imwrite(str(out / f"{stem}_zoom.jpg"), cv2.resize(both, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(sorted(os.listdir(out)))
