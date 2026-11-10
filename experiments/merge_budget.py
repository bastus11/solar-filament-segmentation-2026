"""How much PQ is locked in merged / fragmented candidates?  (full OOF, selected set of the standard scorer)"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from fil import paths, candidates
name = sys.argv[1]
df = pd.read_pickle(paths.RUNS / "cand" / f"{name}.pkl"); meta = pd.read_pickle(paths.RUNS / "cand" / f"{name}_meta.pkl")
oof = np.load(paths.RUNS / "cand" / f"{name}_oof.npy")
ch = np.zeros(len(df), bool)
for _, idx in df.groupby("img").indices.items():
    ch[idx] = candidates.select(df.node.values[idx], df.parent.values[idx], df.k.values[idx], oof[idx], 0.52)
c = df[ch]
B = np.nan_to_num(c[["biou0", "biou1", "biou2"]].values); has = ~np.isnan(c[["biou0", "biou1", "biou2"]].values)
b = B[has]
print("selected candidates x records:", len(b))
for lo, hi in [(0.5, 1.01), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.05, 0.2), (0, 0.05)]:
    print(f"  best IoU in [{lo},{hi}): {((b >= lo) & (b < hi)).mean():.3f}")
print("matched IoU histogram: " + " ".join(f"{lo:.1f}-{lo+0.1:.1f}:{((b >= lo) & (b < lo + 0.1)).sum() / (b > 0.5).sum():.3f}" for lo in (0.5, 0.6, 0.7, 0.8, 0.9)))
