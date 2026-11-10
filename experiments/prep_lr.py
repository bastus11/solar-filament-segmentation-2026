"""Extra target: union of the chirality-identified (Left/Right) filaments per annotator record.

Writes cache/union_lr.npy (N, H, W) uint8 bit planes, same layout as union.npy.
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fil import data, paths

recs = data.load_records(); by = data.records_by_stem(recs); gt = data.cached_gt_rles(recs)
stems = json.load(open(paths.CACHE / "stems.json"))
out = np.lib.format.open_memmap(paths.CACHE / "union_lr.npy", "w+", np.uint8, (len(stems), paths.H, paths.W))

def work(i):
    u = np.zeros((paths.H, paths.W), np.uint8)
    for k, r in enumerate(by[stems[i]]):
        rl = [x for x, a in zip(gt[r.record_id], r.anns) if a["category_id"] in (1, 2) and a["area"] >= 30]
        if rl:
            u |= (data.decode(rl).max(0) << k).astype(np.uint8)
    out[i] = u

with ThreadPoolExecutor(16) as ex:
    list(ex.map(work, range(len(stems))))
out.flush()
uni = np.load(paths.CACHE / "union.npy", mmap_mode="r")
a = sum(int((out[i] > 0).sum()) for i in range(0, len(stems), 9)); b = sum(int((uni[i] > 0).sum()) for i in range(0, len(stems), 9))
print("union_lr written; L/R share of annotated pixels: %.3f" % (a / b))
