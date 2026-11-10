"""COCO run-length encoding helpers and submission I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from . import paths


def encode_mask(mask: np.ndarray) -> dict:
    """(H, W) binary mask -> compressed COCO RLE (reference implementation, slow for many instances)."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def encode_bbox_mask(sub: np.ndarray, x0: int, y0: int, h: int = paths.H, w: int = paths.W) -> dict:
    """RLE of a full-frame mask that is zero outside the patch ``sub`` placed at (x0, y0).

    Only a (H, patch_width) strip is materialised, which makes encoding ~20x cheaper than building
    the full 2048x2048 mask for every instance.
    """
    ph, pw = sub.shape
    strip = np.zeros((pw, h), np.uint8)            # column-major layout: one row per image column
    strip[:, y0:y0 + ph] = sub.T
    flat = strip.reshape(-1)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        counts = np.concatenate(([0], counts))
    counts[0] += x0 * h
    tail = (w - x0 - pw) * h
    if len(counts) % 2 == 1:
        counts[-1] += tail
    elif tail > 0:
        counts = np.concatenate((counts, [tail]))
    return mask_util.frPyObjects({"size": [h, w], "counts": counts.tolist()}, h, w)


def labels_to_rles(lab: np.ndarray, stats: np.ndarray | None = None) -> list[dict]:
    """Label map (0 = background, 1..n) -> list of n RLEs."""
    n = int(lab.max())
    if n == 0:
        return []
    if stats is None:
        import cv2
        stats = np.zeros((n, 4), np.int64)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            stats[i - 1] = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    out = []
    for i in range(1, n + 1):
        x0, y0, bw, bh = (int(v) for v in stats[i - 1][:4])
        out.append(encode_bbox_mask(lab[y0:y0 + bh, x0:x0 + bw] == i, x0, y0, *lab.shape))
    return out


def counts_str(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def write_submission(pred: dict[str, list[dict]], path) -> pd.DataFrame:
    """``pred``: {image_stem: [rle, ...]} -> csv with columns filament_id, segmentation_rle."""
    rows = [(f"{stem}_{k}", counts_str(r)) for stem in sorted(pred) for k, r in enumerate(pred[stem], 1)]
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False)
    return df


def read_submission(path) -> dict[str, list[dict]]:
    df = pd.read_csv(path)
    out: dict[str, list[dict]] = {}
    for fid, rle in zip(df["filament_id"], df["segmentation_rle"]):
        out.setdefault(fid.rsplit("_", 1)[0], []).append({"size": [paths.H, paths.W], "counts": rle.encode("ascii")})
    return out


def check_submission(path, valid_stems: set[str] | None = None) -> dict:
    """Validate a submission file: decodable RLEs, unique ids, pixel-disjoint masks per image."""
    df = pd.read_csv(path)
    assert list(df.columns) == ["filament_id", "segmentation_rle"], df.columns
    assert df["filament_id"].is_unique, "duplicate filament ids"
    assert not df["segmentation_rle"].str.contains(r'["\',\s]').any(), "illegal characters in RLE"
    pred = read_submission(path)
    n_overlap = 0
    for stem, rles in pred.items():
        if valid_stems is not None:
            assert stem in valid_stems, f"unknown image {stem}"
        m = mask_util.decode(rles)
        assert m.shape[:2] == (paths.H, paths.W)
        assert (m.reshape(-1, m.shape[2]).sum(0) > 0).all(), f"empty mask in {stem}"
        if (m.sum(2) > 1).any():
            n_overlap += 1
    assert n_overlap == 0, f"{n_overlap} images with overlapping masks"
    return {"rows": len(df), "images": len(pred), "mean_per_image": len(df) / max(len(pred), 1)}
