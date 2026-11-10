"""Pre-rasterise images and training targets into flat uint8 memmaps (random crops become free).

Outputs in ``cache/``:
  stems.json                 ordered list of image stems (train) / test_stems.json
  img.u8        (N,H,W)      grayscale images
  union.u8      (N,H,W)      bit k set  <=> pixel inside any polygon of annotator record k
  spine.u8      (N,H,W)      bit k set  <=> pixel on a (3 px wide) spine of annotator record k
  nrec.npy      (N,)         number of annotator records per image
  test_img.u8   (Nt,H,W)
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
from fil import data, paths

H = W = paths.H
SPINE_THICK = 3
MIN_AREA_TRAIN = 30          # polygons smaller than this are annotation slips -> not used as targets


def read_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return im if im.ndim == 2 else im[..., 0]


def main():
    recs = data.load_records(); by = data.records_by_stem(recs); gt = data.cached_gt_rles(recs)
    stems = sorted(by)
    paths.CACHE.mkdir(exist_ok=True, parents=True)
    json.dump(stems, open(paths.CACHE / "stems.json", "w"))
    N = len(stems)
    img = np.lib.format.open_memmap(paths.CACHE / "img.npy", "w+", np.uint8, (N, H, W))
    uni = np.lib.format.open_memmap(paths.CACHE / "union.npy", "w+", np.uint8, (N, H, W))
    spi = np.lib.format.open_memmap(paths.CACHE / "spine.npy", "w+", np.uint8, (N, H, W))
    nrec = np.zeros(N, np.uint8)

    def work(i):
        s = stems[i]
        img[i] = read_gray(paths.TRAIN_IMG_DIR / f"{s}.jpeg")
        u = np.zeros((H, W), np.uint8); sp = np.zeros((H, W), np.uint8)
        for k, r in enumerate(by[s]):
            rl = [x for x, a in zip(gt[r.record_id], r.anns) if a["area"] >= MIN_AREA_TRAIN]
            if rl:
                u |= (data.decode(rl).max(0) << k).astype(np.uint8)
            canvas = np.zeros((H, W), np.uint8)
            for a in r.anns:
                if a["area"] < MIN_AREA_TRAIN or not a["spine"]:
                    continue
                pts = np.round(np.array(a["spine"], np.float32).reshape(-1, 2)).astype(np.int32)
                cv2.polylines(canvas, [pts], False, 1, SPINE_THICK, cv2.LINE_8)
            sp |= (canvas << k).astype(np.uint8)
        uni[i] = u; spi[i] = sp; nrec[i] = len(by[s])
        return i

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(work, range(N)))
    img.flush(); uni.flush(); spi.flush()
    np.save(paths.CACHE / "nrec.npy", nrec)

    tstems = sorted(p.stem for p in paths.TEST_IMG_DIR.glob("*.jpeg"))
    json.dump(tstems, open(paths.CACHE / "test_stems.json", "w"))
    timg = np.lib.format.open_memmap(paths.CACHE / "test_img.npy", "w+", np.uint8, (len(tstems), H, W))
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda i: timg.__setitem__(i, read_gray(paths.TEST_IMG_DIR / f"{tstems[i]}.jpeg")), range(len(tstems))))
    timg.flush()
    print("train", img.shape, "test", timg.shape, "nrec dist", np.bincount(nrec))
    # foreground statistics
    fg = np.mean([(uni[i] > 0).mean() for i in range(0, N, 7)])
    print("mean foreground fraction (any annotator): %.5f" % fg)

    # solar disk geometry: is the disk centred with a constant radius?
    res = []
    for i in list(range(0, N, 25)):
        g = img[i]
        m = (g > 8).astype(np.uint8)          # everything but the black corners
        # limb = bright ring; estimate radius from the radial profile's maximum gradient
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx - (W - 1) / 2, yy - (H - 1) / 2).astype(np.int32)
        prof = np.bincount(r.ravel(), g.ravel().astype(np.float64)) / np.maximum(np.bincount(r.ravel()), 1)
        k = int(np.argmax(prof[700:1000])) + 700
        res.append(k)
    print("limb-peak radius (px) over sampled images: min %d max %d mean %.1f" % (min(res), max(res), np.mean(res)))


if __name__ == "__main__":
    main()
