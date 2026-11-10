"""Random-crop training dataset working on the pre-rasterised memmaps (see scripts/prep_targets.py)."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import data, paths

cv2.setNumThreads(0)

DISK_C = (paths.W - 1) / 2.0     # the solar disk is centred ...
DISK_R = 900.0                   # ... with a constant radius (size-normalised GONG frames)
IMG_MEAN, IMG_STD = 0.35, 0.20

_POP = np.array([bin(i).count("1") for i in range(256)], np.float32)


def bits_to_soft(bits: np.ndarray, nrec: int) -> np.ndarray:
    """bit-plane target (one bit per annotator) -> fraction of annotators that marked the pixel."""
    return _POP[bits] / float(nrec)


def normalise_image(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD


def coord_channels(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Disk-normalised source coordinates (x/R, y/R) -> (2, h, w) float32."""
    return np.stack([(xs - DISK_C) / DISK_R, (ys - DISK_C) / DISK_R]).astype(np.float32)


def full_coords(h: int = paths.H, w: int = paths.W) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return coord_channels(xs, ys)


class CropDataset(Dataset):
    """Yields augmented crops: input (3, S, S) = [image, x/R, y/R], target (2, S, S) = [mask, spine].

    Geometric augmentation is a single affine warp (rotation, isotropic scale, flip) sampled around
    either a random annotated filament (``fg_prob``) or a random on-disk location.
    """

    def __init__(self, img_idx: list[int], crop: int = 1024, length: int = 4000, fg_prob: float = 0.7,
                 scale: tuple[float, float] = (0.75, 1.33), rotate: bool = True, intensity: bool = True,
                 weight_by_nrec: bool = True, seed: int = 0, native_prob: float = 0.0):
        self.img_idx = np.asarray(img_idx)
        self.crop, self.length, self.fg_prob = crop, length, fg_prob
        self.scale, self.rotate, self.intensity = scale, rotate, intensity
        self.seed = seed
        self.native_prob = native_prob     # share of samples drawn on the native pixel grid (no resampling)
        self.stems = json.load(open(paths.CACHE / "stems.json"))
        self.nrec = np.load(paths.CACHE / "nrec.npy")
        p = self.nrec[self.img_idx].astype(np.float64) if weight_by_nrec else np.ones(len(self.img_idx))
        self.p = p / p.sum()
        # filament centres per image (all annotators) for foreground-biased sampling
        recs = data.load_records(); by = data.records_by_stem(recs)
        self.centres = {}
        for i in self.img_idx:
            c = [[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2,
                  max(a["bbox"][2], a["bbox"][3])] for r in by[self.stems[i]] for a in r.anns]
            self.centres[int(i)] = np.asarray(c, np.float32).reshape(-1, 3)
        self._mm = None

    def _open(self):
        if self._mm is None:
            self._mm = tuple(np.load(paths.CACHE / f"{n}.npy", mmap_mode="r") for n in ("img", "union", "spine"))
        return self._mm

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        img_mm, uni_mm, spi_mm = self._open()
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, k, info.id if info else 0, int(torch.randint(0, 2**31 - 1, (1,)))])
        i = int(rng.choice(self.img_idx, p=self.p))
        S = self.crop
        # --- sample the affine: output pixel (u, v) -> source pixel (x, y)
        native = rng.random() < self.native_prob
        if native:      # flips / 90-degree turns / integer shifts only: pixels are copied, never interpolated
            s, th = 1.0, float(rng.integers(4)) * math.pi / 2
        else:
            s = math.exp(rng.uniform(math.log(self.scale[0]), math.log(self.scale[1])))   # zoom factor (>1: magnify)
            th = rng.uniform(0, 2 * math.pi) if self.rotate else 0.0
        flip = rng.random() < 0.5
        cen = self.centres[i]
        if len(cen) and rng.random() < self.fg_prob:
            c = cen[rng.integers(len(cen))]
            jit = 0.35 * S / s
            cx, cy = c[0] + rng.uniform(-jit, jit), c[1] + rng.uniform(-jit, jit)
        else:
            rr, a = DISK_R * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            cx, cy = DISK_C + rr * math.cos(a), DISK_C + rr * math.sin(a)
        ca, sa = math.cos(th) / s, math.sin(th) / s
        if native:
            ca, sa = float(round(ca)), float(round(sa))
            cx, cy = math.floor(cx) + (0.5 if S % 2 == 0 else 0.0), math.floor(cy) + (0.5 if S % 2 == 0 else 0.0)
        fx = -1.0 if flip else 1.0
        # [x, y]^T = A [u - S/2, v - S/2]^T + [cx, cy]^T
        A = np.array([[ca * fx, -sa], [sa * fx, ca]], np.float64)
        t = np.array([cx, cy]) - A @ np.array([(S - 1) / 2.0, (S - 1) / 2.0])
        # --- read only the source window that is needed
        corners = (A @ np.array([[0, S - 1, 0, S - 1], [0, 0, S - 1, S - 1]], np.float64)) + t[:, None]
        x0 = int(max(0, math.floor(corners[0].min()) - 2)); x1 = int(min(paths.W, math.ceil(corners[0].max()) + 3))
        y0 = int(max(0, math.floor(corners[1].min()) - 2)); y1 = int(min(paths.H, math.ceil(corners[1].max()) + 3))
        if x1 <= x0 or y1 <= y0:                       # crop entirely outside the frame (should not happen)
            x0, x1, y0, y1 = 0, 8, 0, 8
        M = np.concatenate([A, (t - np.array([x0, y0]))[:, None]], 1)
        flags = (cv2.INTER_NEAREST if native else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
        nrec = int(self.nrec[i])
        img = cv2.warpAffine(np.ascontiguousarray(img_mm[i, y0:y1, x0:x1]), M, (S, S), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255.0
        msk = cv2.warpAffine(bits_to_soft(uni_mm[i, y0:y1, x0:x1], nrec), M, (S, S), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        spn = cv2.warpAffine(bits_to_soft(spi_mm[i, y0:y1, x0:x1], nrec), M, (S, S), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # --- photometric augmentation (image only)
        if self.intensity:
            if rng.random() < 0.8:
                img = np.power(np.clip(img, 0, 1), math.exp(rng.uniform(-0.25, 0.25)))
            if rng.random() < 0.8:
                img = (img - 0.35) * math.exp(rng.uniform(-0.2, 0.2)) + 0.35 + rng.uniform(-0.04, 0.04)
            if rng.random() < 0.2:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))
            if rng.random() < 0.3:
                img = img + rng.normal(0, rng.uniform(0.004, 0.02), img.shape).astype(np.float32)
        img = (img - IMG_MEAN) / IMG_STD
        u, v = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
        xs = A[0, 0] * u + A[0, 1] * v + t[0]
        ys = A[1, 0] * u + A[1, 1] * v + t[1]
        x = np.concatenate([img[None].astype(np.float32), coord_channels(xs, ys)], 0)
        y = np.stack([msk, spn]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
