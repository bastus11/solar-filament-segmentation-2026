"""Collect the final models into ``weights/`` (checkpoints + scorer) for the repository.

Full precision by default so that the notebook reproduces the submitted file; ``--half`` halves the size
(masks then differ by a few boundary pixels).

    python scripts/export_weights.py --exps v1_effv2s --scorer v1e --lam 0.60
"""
import sys, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fil import paths

ap = argparse.ArgumentParser()
ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--scorer", required=True)
ap.add_argument("--half", action="store_true")
ap.add_argument("--lam", type=float, default=0.60); ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "weights"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
for e in a.exps:
    for f in range(5):
        ck = torch.load(paths.RUNS / e / f"fold{f}.pt", map_location="cpu", weights_only=False)
        if a.half:
            ck["model"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
        torch.save(ck, out / f"{e}_fold{f}.pt")
sc = pickle.load(open(paths.RUNS / "cand" / f"{a.scorer}_scorer.pkl", "rb"))
sc["lam_submit"] = a.lam
pickle.dump(sc, open(out / "scorer.pkl", "wb"))
print("exported:", sorted(p.name for p in out.iterdir()), "total MB: %.0f" % (sum(p.stat().st_size for p in out.iterdir()) / 1e6))
