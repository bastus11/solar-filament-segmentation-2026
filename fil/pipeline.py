"""End-to-end inference: H-alpha frame -> pixel-disjoint filament instances.

    seg = FilamentSegmenter.from_dir("weights")
    result = seg.segment(image_uint8)            # result.labels, result.scores, result.rles()
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import candidates, infer, rle
from .model import build_model


@dataclass
class Segmentation:
    labels: np.ndarray        # (H, W) int32, 0 = background, 1..n = filaments
    stats: np.ndarray         # (n, 4) bounding boxes x, y, w, h
    scores: np.ndarray        # (n,) predicted expected matched IoU of every instance
    prob: np.ndarray          # (2, H, W) uint8 filament / spine probability maps

    def rles(self) -> list[dict]:
        return rle.labels_to_rles(self.labels, self.stats)


class FilamentSegmenter:
    """Fold/architecture ensemble + candidate tree + learned scorer + PQ-optimal selection."""

    def __init__(self, checkpoints: list, scorer: dict, lam: float | None = None, tta: str = "d4", device: str = "cuda"):
        self.device, self.tta = device, tta
        self.models = []
        for c in checkpoints:
            ck = torch.load(c, map_location="cpu", weights_only=False) if not isinstance(c, dict) else c
            m = build_model(ck["args"]["arch"], ck["args"]["encoder"], out_ch=ck.get("out_ch", 2), pretrained=False)
            m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
            self.models.append(m.to(device).eval().to(memory_format=torch.channels_last))
        self.feats = scorer["feats"]
        self.boosters = [lgb.Booster(model_str=s) for s in scorer["models"]]
        self.cfg = candidates.Cfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in scorer.get("cfg", {}).items()})
        self.lam = float(lam if lam is not None else scorer.get("lam_submit", scorer["lam"]))

    @classmethod
    def from_dir(cls, weights_dir, **kw) -> "FilamentSegmenter":
        d = Path(weights_dir)
        ckpts = sorted(d.glob("*fold*.pt"))
        scorer = pickle.load(open(d / "scorer.pkl", "rb"))
        return cls(ckpts, scorer, **kw)

    def predict_proba(self, img: np.ndarray) -> np.ndarray:
        """(H, W) uint8 image -> (2, H, W) uint8 probability maps (ensemble average, TTA, label alignment)."""
        p = sum(infer.predict_full(m, img, tta=self.tta, device=self.device)[:2] for m in self.models) / len(self.models)
        return np.clip(np.round(p * 255), 0, 255).astype(np.uint8)

    def instances_from_proba(self, prob: np.ndarray, img: np.ndarray) -> Segmentation:
        rows, masks = candidates.extract(prob[0], prob[1], img, self.cfg)
        if not rows:
            return Segmentation(np.zeros(img.shape, np.int32), np.zeros((0, 4), np.int64), np.zeros(0), prob)
        df = pd.DataFrame(rows)
        X = df[self.feats].values.astype(np.float32)
        score = np.mean([b.predict(X) for b in self.boosters], 0)
        chosen = candidates.select(df.node.values, df.parent.values, df.k.values, score, self.lam)
        idx = np.flatnonzero(chosen)
        idx = idx[np.argsort(-score[idx])]
        labels, stats = candidates.paint(masks, idx, img.shape)
        return Segmentation(labels, stats, score[idx][:len(stats)], prob)

    def segment(self, img: np.ndarray) -> Segmentation:
        return self.instances_from_proba(self.predict_proba(img), img)
