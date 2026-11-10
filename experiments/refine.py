"""Stage 2: instance-level mask refinement.

The first-stage network labels pixels independently; it cannot decide where ONE filament ends
(annotators extend a filament across faint gaps, stop at a crossing, keep or drop a barb).  The
refiner sees a single candidate mask together with the image and the stage-1 probability maps and
re-draws the outline of that one filament.  It is trained on out-of-fold candidates against the
annotated instance(s) they overlap, with a Dice term, i.e. it optimises the per-instance overlap
that Panoptic Quality rewards.

Everything stays on the native pixel grid (no resampling): training uses 256x256 windows around a
random pixel of the candidate, inference uses a square window around the whole candidate.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from . import paths
from .dataset import IMG_MEAN, IMG_STD

WIN = 256
BUCKETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
MARGIN = 40


def build_refiner(encoder: str = "resnet34", pretrained: bool = True):
    """U-Net with inputs [image, candidate mask, filament prob, spine prob] -> refined mask logit."""
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=4, classes=1)


def window(arr: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """size x size window with top-left (x0, y0); parts outside the frame are zero."""
    H, W = arr.shape
    out = np.zeros((size, size), arr.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return out


def stack_inputs(img: np.ndarray, cand: np.ndarray, P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """uint8 windows -> float32 (4, h, w) network input."""
    return np.stack([(img.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD, cand.astype(np.float32),
                     P.astype(np.float32) / 255.0, S.astype(np.float32) / 255.0])


def bucket_for(bw: int, bh: int) -> int:
    need = max(bw, bh) + 2 * MARGIN
    for b in BUCKETS:
        if b >= need:
            return b
    return BUCKETS[-1]


def clean(mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Keep the connected component that overlaps the original candidate most and fill its holes."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    if n > 2:
        ov = np.bincount(lab[ref > 0], minlength=n); ov[0] = 0
        k = int(np.argmax(ov)) if ov.max() > 0 else int(np.argmax(np.bincount(lab.reshape(-1))[1:])) + 1
        mask = lab == k
    m = mask.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8); pad[1:-1, 1:-1] = m
    cv2.floodFill(pad, None, (0, 0), 2)
    return (pad[1:-1, 1:-1] != 2).astype(np.uint8)


@torch.no_grad()
def refine_masks(models: list, img: np.ndarray, P: np.ndarray, S: np.ndarray, masks: list, thr: float = 0.5,
                 tta: bool = True, batch: int = 32, return_prob: bool = False) -> list:
    """Refine candidate masks of ONE image.

    masks: list of (x0, y0, uint8 patch).  Returns a list of (x0, y0, uint8 patch) in frame
    coordinates (possibly empty patches when the refiner rejects the candidate).
    """
    H, W = img.shape
    out = [None] * len(masks)
    groups = {}
    for i, (x0, y0, m) in enumerate(masks):
        bh, bw = m.shape
        groups.setdefault(bucket_for(bw, bh), []).append(i)
    for size, idxs in groups.items():
        bs = max(1, int(batch * (256 * 256) / (size * size)))
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            xs, wins = [], []
            for i in chunk:
                x0, y0, m = masks[i]
                bh, bw = m.shape
                wx = int(round(x0 + bw / 2.0 - size / 2.0)); wy = int(round(y0 + bh / 2.0 - size / 2.0))
                cand = np.zeros((H, W), np.uint8) if False else None
                cw = np.zeros((size, size), np.uint8)
                ox, oy = x0 - wx, y0 - wy
                xa, ya = max(ox, 0), max(oy, 0); xb, yb = min(ox + bw, size), min(oy + bh, size)
                cw[ya:yb, xa:xb] = m[ya - oy:yb - oy, xa - ox:xb - ox]
                xs.append(stack_inputs(window(img, wx, wy, size), cw, window(P, wx, wy, size), window(S, wx, wy, size)))
                wins.append((wx, wy, cw))
            x = torch.from_numpy(np.stack(xs)).cuda()
            acc = 0
            views = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
            for hf, vf in views:
                xt = x
                if hf: xt = torch.flip(xt, dims=[3])
                if vf: xt = torch.flip(xt, dims=[2])
                with torch.autocast("cuda", dtype=torch.float16):
                    p = sum(torch.sigmoid(mdl(xt).float()) for mdl in models) / len(models)
                if vf: p = torch.flip(p, dims=[2])
                if hf: p = torch.flip(p, dims=[3])
                acc = acc + p
            prob = (acc / len(views))[:, 0].cpu().numpy()
            for i, pr, (wx, wy, cw) in zip(chunk, prob, wins):
                m = clean(pr >= thr, cw)
                # clip to the frame
                xa, ya = max(-wx, 0), max(-wy, 0); xb, yb = min(W - wx, size), min(H - wy, size)
                m = m[ya:yb, xa:xb]
                ys, xs_ = np.nonzero(m)
                if len(xs_) == 0:
                    out[i] = (wx + xa, wy + ya, np.zeros((1, 1), np.uint8)); continue
                out[i] = (wx + xa + int(xs_.min()), wy + ya + int(ys.min()),
                          np.ascontiguousarray(m[ys.min():ys.max() + 1, xs_.min():xs_.max() + 1]))
    return out
