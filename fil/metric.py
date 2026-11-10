"""Panoptic Quality exactly as defined by the organisers' self-evaluation notebook.

For every annotator record of an image the predictions of that image are compared with the
record's ground-truth instances; a (gt, pred) pair with IoU > 0.5 is a TP, predictions without
a hit are FP, ground-truth instances without a hit are FN.  Counts are pooled over all records:

    PQ = sum(IoU of TPs) / (|TP| + 0.5 |FP| + 0.5 |FN|)

IoUs are computed directly on COCO RLEs (no decoding), so a full CV evaluation takes < 1 s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util

IOU_THR = 0.5


@dataclass
class PQ:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, o: "PQ") -> "PQ":
        return PQ(self.iou_sum + o.iou_sum, self.tp + o.tp, self.fp + o.fp, self.fn + o.fn)

    @property
    def pq(self) -> float:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / den if den > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / den if den > 0 else 0.0

    def __str__(self) -> str:
        return (f"PQ={self.pq:.4f} SQ={self.sq:.4f} RQ={self.rq:.4f} "
                f"TP={self.tp} FP={self.fp} FN={self.fn}")


def iou_matrix(gt_rles: list, pred_rles: list) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    return np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))).T.reshape(
        len(gt_rles), len(pred_rles))


def pq_record(gt_rles: list, pred_rles: list) -> PQ:
    """PQ statistics of one annotator record."""
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0:
        return PQ(fp=n_pred)
    if n_pred == 0:
        return PQ(fn=n_gt)
    iou = iou_matrix(gt_rles, pred_rles)
    hit = iou > IOU_THR
    return PQ(float(iou[hit].sum()), int(hit.sum()),
              int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum()))


def pq_dataset(gt: dict[str, list], pred: dict[str, list]) -> PQ:
    """``gt``: {record_id: rles};  ``pred``: {image_stem: rles}.  Records are pooled."""
    tot = PQ()
    for rid, g in gt.items():
        stem = rid.split("-", 1)[1]
        tot = tot + pq_record(g, pred.get(stem, []))
    return tot


# --------------------------------------------------------------------------------------------
# Label-map based evaluation (used for fast post-processing sweeps; identical result to the RLE path
# for pixel-disjoint predictions)
# --------------------------------------------------------------------------------------------
def gt_pixel_cache() -> dict[str, list[np.ndarray]]:
    """``{record_id: [flat pixel indices (row-major) of each GT instance]}`` cached on disk."""
    import pickle
    from . import data, paths
    f = paths.CACHE / "gt_pix.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    gt = data.cached_gt_rles()
    out = {}
    for rid, rl in gt.items():
        out[rid] = [np.flatnonzero(mask_util.decode(r)).astype(np.int32) for r in rl]
    pickle.dump(out, open(f, "wb"))
    return out


def pq_from_labels(lab: np.ndarray, pred_area: np.ndarray, gt_pix: list[np.ndarray]) -> PQ:
    """PQ statistics of one record from a predicted label map.

    lab       : (H, W) int array, 0 = background, 1..n = predicted instances (pixel-disjoint)
    pred_area : (n + 1,) pixel count of every label (index 0 unused)
    gt_pix    : flat pixel indices of each ground-truth instance of the record
    """
    n_pred = len(pred_area) - 1
    if len(gt_pix) == 0:
        return PQ(fp=n_pred)
    if n_pred == 0:
        return PQ(fn=len(gt_pix))
    flat = lab.reshape(-1)
    hit_pred = np.zeros(n_pred + 1, bool)
    iou_sum, tp, fn = 0.0, 0, 0
    for pix in gt_pix:
        l = flat[pix]
        l = l[l > 0]
        ok = False
        if l.size:
            ids, inter = np.unique(l, return_counts=True)
            iou = inter / (pix.size + pred_area[ids] - inter)
            h = iou > IOU_THR
            if h.any():
                ok = True
                tp += int(h.sum()); iou_sum += float(iou[h].sum()); hit_pred[ids[h]] = True
        if not ok:
            fn += 1
    return PQ(iou_sum, tp, n_pred - int(hit_pred[1:].sum()), fn)
