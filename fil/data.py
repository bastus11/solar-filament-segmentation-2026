"""Annotation loading and rasterisation helpers.

Terminology
-----------
image   : one H-alpha observation (unique ``file_name`` stem, e.g. ``20140609195854Bh``)
record  : one annotator's pass over one image (``image['id']``, e.g. ``040301-20140609195854Bh``)
instance: one annotated filament (single polygon) inside a record
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

from . import paths


@dataclass
class Record:
    record_id: str            # "<annotator>-<stem>"
    stem: str                 # image stem (file name without extension)
    annotator: str            # 6-digit annotator/batch code
    anns: list = field(default_factory=list)   # raw COCO annotation dicts

    def rles(self, h: int = paths.H, w: int = paths.W) -> list:
        """One COCO RLE per instance, rasterised exactly like the organisers do (pycocotools)."""
        out = []
        for a in self.anns:
            r = mask_util.frPyObjects(a["segmentation"], h, w)
            out.append(mask_util.merge(r) if isinstance(r, list) else r)
        return out


def load_records(json_path=paths.TRAIN_JSON) -> dict[str, Record]:
    """Parse the COCO json into ``{record_id: Record}``."""
    d = json.load(open(json_path))
    recs: dict[str, Record] = {}
    for im in d["images"]:
        rid = im["id"]
        annotator, stem = rid.split("-", 1)
        recs[rid] = Record(rid, stem, annotator)
    for a in d["annotations"]:
        recs[a["image_id"]].anns.append(a)
    return recs


def records_by_stem(recs: dict[str, Record]) -> dict[str, list[Record]]:
    by = defaultdict(list)
    for r in recs.values():
        by[r.stem].append(r)
    return {k: sorted(v, key=lambda r: r.record_id) for k, v in sorted(by.items())}


def cached_gt_rles(recs: dict[str, Record] | None = None) -> dict[str, list]:
    """``{record_id: [rle, ...]}`` cached on disk (rasterising 8k polygons takes a few seconds)."""
    f = paths.CACHE / "gt_rles.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    recs = recs or load_records()
    out = {rid: r.rles() for rid, r in recs.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def decode(rles: list, h: int = paths.H, w: int = paths.W) -> np.ndarray:
    """RLE list -> uint8 array (n, H, W)."""
    if not rles:
        return np.zeros((0, h, w), np.uint8)
    return np.ascontiguousarray(mask_util.decode(rles).transpose(2, 0, 1))


def make_folds(stems: list[str], k: int = 5, seed: int = 42) -> dict[str, int]:
    """Random K-fold over *images* (all annotator records of an image share a fold).

    The hidden test set is a random sample of the same observation pool, so a plain
    random split over images mirrors the train/test relationship.
    """
    rng = np.random.RandomState(seed)
    stems = sorted(stems)
    perm = rng.permutation(len(stems))
    return {stems[j]: int(i % k) for i, j in enumerate(perm)}
