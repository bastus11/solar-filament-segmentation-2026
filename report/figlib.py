"""Shared helpers for the report figures (held-out predictions of the final system)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, cv2
from fil import data, paths, candidates

EXP, TAB, LAM = "v1_effv2s_d4", "v1e", 0.60
_S = {}


def state():
    if not _S:
        stems = json.load(open(paths.CACHE / "stems.json"))
        df = pd.read_pickle(paths.RUNS / "cand" / f"{TAB}.pkl")
        _S.update(stems=stems, idx={s: i for i, s in enumerate(stems)}, rows_of=df.groupby("img").indices,
                  oof=np.load(paths.RUNS / "cand" / f"{TAB}_oof.npy"), by=data.records_by_stem(data.load_records()),
                  img=np.load(paths.CACHE / "img.npy", mmap_mode="r"),
                  where={s: (f, j) for f in range(5) for j, s in enumerate(json.load(open(paths.RUNS / EXP / f"oof_fold{f}_stems.json")))})
    return _S


def held_out(stem):
    """image, prob maps, candidate table (with scores / chosen flag), masks, label map of the selection."""
    S = state(); f, j = S["where"][stem]; i = S["idx"][stem]
    pr = np.ascontiguousarray(np.load(paths.RUNS / EXP / f"oof_fold{f}.npy", mmap_mode="r")[j])
    img = np.ascontiguousarray(S["img"][i])
    rows, masks = candidates.extract(pr[0], pr[1], img)
    if not rows:                                  # nothing above the lowest threshold in this frame
        return img, pr, pd.DataFrame(), masks, np.zeros(img.shape, np.int32), np.zeros(0, np.int64)
    df = pd.DataFrame(rows); df["score"] = S["oof"][S["rows_of"][i]]
    df["chosen"] = candidates.select(df.node.values, df.parent.values, df.k.values, df.score.values, LAM)
    sel = np.flatnonzero(df.chosen.values); sel = sel[np.argsort(-df.score.values[sel])]
    lab, stats = candidates.paint(masks, sel, img.shape)
    return img, pr, df, masks, lab, sel


def show(img, x0, y0, s, lo=None, hi=None):
    """Contrast-stretched RGB view of a crop."""
    c = img[y0:y0 + s, x0:x0 + s].astype(np.float32)
    lo = np.percentile(c, 1) if lo is None else lo; hi = np.percentile(c, 99.5) if hi is None else hi
    c = np.clip((c - lo) / max(hi - lo, 1), 0, 1)
    return np.repeat((c * 255).astype(np.uint8)[..., None], 3, 2)


def polys(stem):
    return [[np.array(a["segmentation"][0], np.float32).reshape(-1, 2) for a in r.anns] for r in state()["by"][stem]]
