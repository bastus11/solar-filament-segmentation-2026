"""Probability map -> pixel-disjoint filament instances."""
from __future__ import annotations

import cv2
import numpy as np


def cc_instances(prob: np.ndarray, thr: float, min_area: int = 0, connectivity: int = 8):
    """Threshold + connected components.

    Returns ``(label_map int32, areas (n+1,), stats (n, 5))`` with labels 1..n (0 = background).
    """
    m = (prob >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity, ltype=cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= max(min_area, 1)
    keep[0] = False
    remap = np.zeros(n, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[lab], np.concatenate([[0], areas[keep]]), stats[keep]
