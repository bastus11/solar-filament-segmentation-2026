"""Two checks quoted in the report:
1. our polygon rasterisation reproduces the `area` field of every annotation (8,199 of 8,199);
2. our PQ implementation equals the organisers' self-evaluation notebook (their functions are executed verbatim)
   on a submission file that is written to disk and read back.

    python experiments/verify_metric.py path/to/self-evaluation-notebook.ipynb
needs the out-of-fold artefacts used by report/figlib.py.
"""
import sys, json, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "report"))
import numpy as np, pandas as pd
from pycocotools import mask as mu
from fil import data, paths, metric, rle
import figlib as F

recs = data.load_records(); gt = data.cached_gt_rles(recs); by = data.records_by_stem(recs)
diff = [abs(float(mu.area(r)) - a["area"]) for rid, rc in recs.items() for r, a in zip(gt[rid], rc.anns)]
print(f"1) annotations: {len(diff)}; rasterised area == json area for {int(np.sum(np.array(diff) < 0.5))}; max |difference| = {max(diff):.1f} px")

nb = json.load(open(sys.argv[1] if len(sys.argv) > 1 else paths.ROOT / "ref/self-evaluation-notebook.ipynb"))
ns = {}
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and ("def get_overlap_matrices" in src or "def get_pq_score" in src or src.startswith("import pandas")):
        exec(src, ns)
# held-out predictions of ~60 training observations, regenerated through the same code path as a submission
stems = sorted(by)[::12][:60]
pred = {}
for s in stems:
    img, pr, df, masks, lab, sel = F.held_out(s)
    rl = rle.labels_to_rles(lab)
    if rl: pred[s] = rl
with tempfile.TemporaryDirectory() as d:
    rle.write_submission(pred, f"{d}/sub.csv"); pred_df = pd.read_csv(f"{d}/sub.csv")
    back = rle.read_submission(f"{d}/sub.csv")
rows = [(f"{r.record_id}_{k}", rle.counts_str(x)) for s in stems for r in by[s] for k, x in enumerate(gt[r.record_id], 1)]
gt_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
theirs = ns["get_pq_score"](ns["get_overlap_df"](gt_df, pred_df))
mine = metric.pq_dataset({r.record_id: gt[r.record_id] for s in stems for r in by[s]}, back)
print(f"2) {len(stems)} observations, {len(gt_df)} annotated filaments, {len(pred_df)} predicted: organisers' notebook PQ = {theirs:.6f} | ours: {mine}")
