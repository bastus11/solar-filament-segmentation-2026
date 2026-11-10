"""Fixed-size views of single candidates for the CNN scorer."""
from __future__ import annotations

import cv2
import numpy as np

CROP = 128
MARGIN = 20
MIN_SIZE = 64


def _window(arr: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    H, W = arr.shape
    out = np.zeros((size, size), arr.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return out


def candidate_crop(img: np.ndarray, P: np.ndarray, S: np.ndarray, x0: int, y0: int, m: np.ndarray) -> np.ndarray:
    """(5, CROP, CROP) uint8: image, candidate mask, filament prob, spine prob, constant scale code.

    The square window is centred on the candidate and sized to its longer side plus a margin, then
    resampled to CROP; the last channel encodes the zoom so that absolute size is not lost.
    """
    bh, bw = m.shape
    size = max(max(bw, bh) + 2 * MARGIN, MIN_SIZE)
    wx = int(round(x0 + bw / 2.0 - size / 2.0)); wy = int(round(y0 + bh / 2.0 - size / 2.0))
    cw = np.zeros((size, size), np.uint8)
    ox, oy = x0 - wx, y0 - wy
    cw[oy:oy + bh, ox:ox + bw] = m * 255
    ch = [_window(img, wx, wy, size), cw, _window(P, wx, wy, size), _window(S, wx, wy, size)]
    if size != CROP:
        interp = cv2.INTER_AREA if size > CROP else cv2.INTER_LINEAR
        ch = [cv2.resize(c, (CROP, CROP), interpolation=interp) for c in ch]
    code = int(np.clip(np.log2(size / MIN_SIZE) / 5.0 * 255, 0, 255))
    return np.stack(ch + [np.full((CROP, CROP), code, np.uint8)])
