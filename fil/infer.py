"""Full-frame inference with test-time augmentation."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import dataset

_COORDS = None

# Annotated polygons sit ~0.25 px above the dark filament body (measured on the training set: the grey
# level inside the polygons is minimal when they are moved 0.22 px down, for every annotator group).
# Rotation / flip augmentation and flip TTA make the network blind to such an offset, so the averaged
# probability map is moved by this amount to follow the annotation convention (+0.0016 PQ on 425
# out-of-fold images; the opposite shift loses the same amount).
LABEL_SHIFT = (0.0, -0.25)       # (dx, dy) in pixels


def _input(img_u8: np.ndarray, device) -> torch.Tensor:
    global _COORDS
    if _COORDS is None or _COORDS.shape[-2:] != img_u8.shape:
        _COORDS = torch.from_numpy(dataset.full_coords(*img_u8.shape))
    x = torch.cat([torch.from_numpy(dataset.normalise_image(img_u8))[None], _COORDS], 0)
    return x[None].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)


TTA = {
    "none": [(False, False, 0)],
    "flip2": [(False, False, 0), (True, False, 0)],
    "flip4": [(False, False, 0), (True, False, 0), (False, True, 0), (True, True, 0)],
    "d4": [(f, False, k) for f in (False, True) for k in range(4)],
}


def align_to_labels(prob: np.ndarray, shift: tuple = LABEL_SHIFT) -> np.ndarray:
    """Translate (C, H, W) probability maps by a sub-pixel offset (bilinear)."""
    if shift == (0.0, 0.0):
        return prob
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return np.stack([cv2.warpAffine(c, M, c.shape[::-1], flags=cv2.INTER_LINEAR) for c in prob])


@torch.no_grad()
def predict_full(model, img_u8: np.ndarray, tta: str = "flip4", device="cuda", amp: bool = True,
                 scales: tuple = (1.0,), shift: tuple = LABEL_SHIFT) -> np.ndarray:
    """(H, W) uint8 image -> (C, H, W) float32 probabilities (average over the TTA views).

    The coordinate channels are transformed together with the image, so every view still carries
    the true position on the solar disk.  ``scales`` runs the network on resized copies of the
    frame (multi-scale TTA, or a network trained at reduced resolution); outputs are resized back.
    """
    x0 = _input(img_u8, device)
    H, W = x0.shape[-2:]
    acc = None
    n = 0
    for sc in scales:
      if sc == 1.0:
        x = x0
      else:
        hs, ws = int(round(H * sc / 32) * 32), int(round(W * sc / 32) * 32)
        x = torch.nn.functional.interpolate(x0, size=(hs, ws), mode="bilinear", antialias=sc < 1, align_corners=False)
      for hf, vf, k in TTA[tta]:
        xt = x
        if hf: xt = torch.flip(xt, dims=[3])
        if vf: xt = torch.flip(xt, dims=[2])
        if k: xt = torch.rot90(xt, k, dims=[2, 3])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            p = torch.sigmoid(model(xt.contiguous(memory_format=torch.channels_last)).float())
        if k: p = torch.rot90(p, -k, dims=[2, 3])
        if vf: p = torch.flip(p, dims=[2])
        if hf: p = torch.flip(p, dims=[3])
        if p.shape[-2:] != (H, W):
            p = torch.nn.functional.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        acc = p if acc is None else acc + p
        n += 1
    return align_to_labels((acc / n)[0].cpu().numpy(), shift)
