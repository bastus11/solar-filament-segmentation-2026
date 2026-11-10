"""Candidate filament masks from a probability map, their descriptors and the PQ-optimal selection.

Why
---
Panoptic Quality pools TP / FP / FN over annotator records:

    PQ = sum(IoU of matched pairs) / (0.5 * (#predictions + #ground-truth instances))

so every submitted mask adds 0.5 to the denominator of each annotator record of its image and
contributes its matched IoU (0 when IoU <= 0.5) to the numerator.  A mask is therefore worth
submitting iff its *expected matched IoU*  s  exceeds  PQ / 2, and among alternative segmentations
of the same region (tight vs. generous outline, one long filament vs. its fragments) the best
choice maximises  sum(s_i - PQ / 2).

How
---
1. Threshold the probability map at a ladder of levels.  The connected components form a forest
   (component tree): every component at a higher level is nested inside one at the level below.
2. Every tree node is a candidate mask.  Additional "grouped" candidates keep the pixels above a
   higher level j but treat all fragments inside a level-k component as ONE filament (annotators
   draw a fragmented filament as a single polygon).
3. A learned regressor predicts s for every candidate from simple descriptors.
4. Dynamic programming over the tree returns the pixel-disjoint subset maximising
   sum(s_i - lambda / 2), with lambda ~ PQ.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .dataset import DISK_C, DISK_R

cv2.setNumThreads(0)

LEVELS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
VAR_LEVELS = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass
class Cfg:
    levels: tuple = LEVELS
    var_levels: tuple = ()              # mask levels of optional grouped (multi-fragment) candidates;
                                        # disabled: they add nothing even with oracle scores (see README)
    min_root_area: int = 80
    min_node_area: int = 40
    margin: int = 9
    fill_holes: bool = True


FEATS = [
    "lvl", "lvl_mask", "is_var", "n_parts", "area", "area_root_ratio", "area_parent_ratio", "child_ratio",
    "n_child", "peak_minus_lvl", "depth", "p_mean", "p_std", "p_min", "p_q10", "p_q50", "p_q90", "p_max",
    "s_mean", "s_max", "s_frac", "s_sum", "ring_p", "ring_p2", "g_in", "g_std", "g_min", "g_ring", "g_ring2",
    "contrast", "contrast2", "bw", "bh", "major", "minor", "elong", "length", "width", "extent", "perim",
    "compact", "solidity", "hole_frac", "r", "absx", "mu", "radial",
    "img_n_roots", "img_area50", "img_psum", "img_gmean", "img_gstd", "root_area", "root_pmax",
]
EXTRA_FEATS = ["e_mean", "e_q50", "e_max", "e_ratio", "e_ring"]     # only when an extra map is supplied
_K3 = np.ones((3, 3), np.uint8)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill the holes of a small binary uint8 patch (annotated polygons never have holes)."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    cv2.floodFill(pad, None, (0, 0), 2)
    return (pad[1:-1, 1:-1] != 2).astype(np.uint8)


def _descr(m: np.ndarray, x0: int, y0: int, P: np.ndarray, S: np.ndarray, G: np.ndarray, mg: int, E=None) -> dict:
    """Descriptors of the mask patch ``m`` (uint8) whose top-left corner sits at (x0, y0)."""
    H, W = P.shape
    h, w = m.shape
    X0, Y0, X1, Y1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x0 + w + mg), min(H, y0 + h + mg)
    mm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    mm[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w] = m
    p = P[Y0:Y1, X0:X1]; s = S[Y0:Y1, X0:X1]; g = G[Y0:Y1, X0:X1]
    ys, xs = np.nonzero(mm)
    area = len(xs)
    pv = p[ys, xs].astype(np.float32) / 255.0
    sv = s[ys, xs].astype(np.float32) / 255.0
    gv = g[ys, xs].astype(np.float32)
    q10, q50, q90 = np.percentile(pv, [10, 50, 90])
    d1 = cv2.dilate(mm, _K3, iterations=3)
    d2 = cv2.dilate(d1, _K3, iterations=5)
    r1 = (d1 > 0) & (mm == 0); r2 = (d2 > 0) & (d1 == 0)
    ring_p = float(p[r1].mean()) / 255.0 if r1.any() else 0.0
    ring_p2 = float(p[r2].mean()) / 255.0 if r2.any() else 0.0
    g_ring = float(g[r1].mean()) if r1.any() else float(gv.mean())
    g_ring2 = float(g[r2].mean()) if r2.any() else g_ring
    cx, cy = float(xs.mean()), float(ys.mean())
    if area >= 3:
        cov = np.cov(np.stack([xs - cx, ys - cy]).astype(np.float64))
        ev, evec = np.linalg.eigh(cov)
        major, minor = float(np.sqrt(max(ev[1], 1e-3))), float(np.sqrt(max(ev[0], 1e-3)))
        ax = evec[:, 1]
    else:
        major = minor = 0.5; ax = np.array([1.0, 0.0])
    gx, gy = X0 + cx - DISK_C, Y0 + cy - DISK_C
    rr = float(np.hypot(gx, gy))
    radial = float(abs(ax[0] * gx + ax[1] * gy) / (rr + 1e-6))
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    hull = float(cv2.contourArea(cv2.convexHull(np.concatenate(cnts)))) if cnts else 0.0
    filled = int(fill_holes(mm).sum())
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    length = 3.46 * major
    rn = rr / DISK_R
    extra = {}
    if E is not None:
        e = E[Y0:Y1, X0:X1]
        ev = e[ys, xs].astype(np.float32) / 255.0
        extra = dict(e_mean=float(ev.mean()), e_q50=float(np.median(ev)), e_max=float(ev.max()),
                     e_ratio=float(ev.mean() / max(float(pv.mean()), 1e-3)), e_ring=float(e[r1].mean()) / 255.0 if r1.any() else 0.0)
    return dict(**extra, **dict(
        n_parts=len(cnts), area=area, p_mean=float(pv.mean()), p_std=float(pv.std()), p_min=float(pv.min()),
        p_q10=float(q10), p_q50=float(q50), p_q90=float(q90), p_max=float(pv.max()),
        s_mean=float(sv.mean()), s_max=float(sv.max()), s_frac=float((sv > 0.3).mean()), s_sum=float(sv.sum()),
        ring_p=ring_p, ring_p2=ring_p2, g_in=float(gv.mean()), g_std=float(gv.std()), g_min=float(gv.min()),
        g_ring=g_ring, g_ring2=g_ring2, contrast=g_ring - float(gv.mean()), contrast2=g_ring2 - float(gv.mean()),
        bw=bw, bh=bh, major=major, minor=minor, elong=major / max(minor, 1e-3), length=length,
        width=area / max(length, 1.0), extent=area / float(bw * bh), perim=perim,
        compact=4 * np.pi * area / max(perim * perim, 1.0), solidity=area / max(hull, 1.0),
        hole_frac=(filled - area) / float(area), cx=gx / DISK_R, cy=gy / DISK_R, r=rn, absx=abs(gx) / DISK_R,
        mu=float(np.sqrt(max(0.0, 1 - min(rn, 1.0) ** 2))), radial=radial,
    ))


def _iou_targets(m: np.ndarray, x0: int, y0: int, gt_labs: list, gt_areas: list):
    """Per annotator record: (best IoU, matched IoU [0 if <= 0.5], id of the best GT instance)."""
    h, w = m.shape
    sel = m > 0
    area = int(sel.sum())
    out = []
    for gl, ga in zip(gt_labs, gt_areas):
        l = gl[y0:y0 + h, x0:x0 + w][sel]
        l = l[l > 0]
        if l.size == 0:
            out.append((0.0, 0.0, 0)); continue
        ids, inter = np.unique(l, return_counts=True)
        iou = inter / (area + ga[ids] - inter)
        k = int(np.argmax(iou))
        out.append((float(iou[k]), float(iou[k]) if iou[k] > 0.5 else 0.0, int(ids[k])))
    return out


def extract(P: np.ndarray, S: np.ndarray, G: np.ndarray, cfg: Cfg = Cfg(), gt_labs=None, gt_areas=None, E=None):
    """Build the candidate forest of one image.

    P, S, G : uint8 (H, W) filament probability, spine probability, H-alpha image.
    Returns ``(rows, masks)``: ``rows`` is a list of dicts (structure + descriptors [+ targets]),
    ``masks[i] = (x0, y0, uint8 patch)`` is the mask of candidate i.
    """
    lv = [int(round(t * 255)) for t in cfg.levels]
    var_k = [cfg.levels.index(t) for t in cfg.var_levels]
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats((P >= lv[0]).astype(np.uint8), connectivity=8)
    rows, masks = [], []
    disk = P.shape[0] // 2
    img_feats = dict(img_n_roots=int((st0[1:, 4] >= cfg.min_root_area).sum()), img_area50=float((P >= 128).sum()),
                     img_psum=float(P.sum()) / 255.0, img_gmean=float(G[disk - 600:disk + 600:4, disk - 600:disk + 600:4].mean()),
                     img_gstd=float(G[disk - 600:disk + 600:4, disk - 600:disk + 600:4].std()))
    node_id = 0
    for r in range(1, n0):
        rx, ry, rw, rh, rarea = (int(v) for v in st0[r])
        if rarea < cfg.min_root_area:
            continue
        rootm = (lab0[ry:ry + rh, rx:rx + rw] == r)
        pm = np.where(rootm, P[ry:ry + rh, rx:rx + rw], 0)
        root_pmax = float(pm.max()) / 255.0
        nodes = []                               # dicts: k, x0, y0, m, parent, anc(list per level), area
        prev_lab, prev_map = None, None
        for k, L in enumerate(lv):
            if k == 0:
                lab = rootm.astype(np.int32); n = 2; st = np.array([[0, 0, 0, 0, 0], [0, 0, rw, rh, rarea]])
            else:
                b = (pm >= L).astype(np.uint8)
                if not b.any():
                    break
                n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
            cur_map = {}
            for i in range(1, n):
                x, y, w, h, a = (int(v) for v in st[i])
                if a < cfg.min_node_area:
                    continue
                m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
                if k == 0:
                    parent, anc = -1, []
                else:
                    px = x + int(np.argmax(m[0]))
                    pl = int(prev_lab[y, px])
                    if pl not in prev_map:
                        continue
                    parent = prev_map[pl]; anc = nodes[parent]["anc"] + [parent]
                cur_map[i] = len(nodes)
                nodes.append(dict(k=k, x0=rx + x, y0=ry + y, m=m, parent=parent, anc=anc, area=a, children=[]))
                if parent >= 0:
                    nodes[parent]["children"].append(len(nodes) - 1)
            if not cur_map:
                break
            prev_lab, prev_map = lab, cur_map
        if not nodes:
            continue
        # depth = number of further levels the node survives
        max_k = {}
        for i in range(len(nodes) - 1, -1, -1):
            nd = nodes[i]
            nd["maxk"] = max([nd["k"]] + [nodes[c]["maxk"] for c in nd["children"]])
        cands = []                               # (node index, mask patch, x0, y0, k, j, is_var)
        for i, nd in enumerate(nodes):
            m = fill_holes(nd["m"]) if cfg.fill_holes else nd["m"]
            cands.append((i, m, nd["x0"], nd["y0"], nd["k"], nd["k"], 0))
        # grouped candidates: fragments at level j that share the ancestor at level k < j
        by_level = {}
        for i, nd in enumerate(nodes):
            by_level.setdefault(nd["k"], []).append(i)
        for j in var_k:
            if j not in by_level:
                continue
            for k in range(0, j):
                groups = {}
                for i in by_level[j]:
                    groups.setdefault(nodes[i]["anc"][k], []).append(i)
                for a_idx, parts in groups.items():
                    if len(parts) < 2:
                        continue
                    an = nodes[a_idx]
                    um = np.zeros_like(an["m"])
                    for i in parts:
                        nd = nodes[i]
                        yy, xx = nd["y0"] - an["y0"], nd["x0"] - an["x0"]
                        hh, ww = nd["m"].shape
                        um[yy:yy + hh, xx:xx + ww] |= (fill_holes(nd["m"]) if cfg.fill_holes else nd["m"])
                    ys, xs = np.nonzero(um)
                    y_a, y_b, x_a, x_b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    cands.append((a_idx, um[y_a:y_b, x_a:x_b], an["x0"] + int(x_a), an["y0"] + int(y_a), k, j, 1))
        for (i, m, x0, y0, k, j, is_var) in cands:
            nd = nodes[i]
            f = _descr(m, x0, y0, P, S, G, cfg.margin, E)
            ch_area = sum(nodes[c]["area"] for c in nd["children"])
            f.update(img_feats)
            f.update(lvl=cfg.levels[k], lvl_mask=cfg.levels[j], is_var=is_var, area_root_ratio=f["area"] / float(rarea),
                     area_parent_ratio=f["area"] / float(nodes[nd["parent"]]["area"]) if nd["parent"] >= 0 else 1.0,
                     child_ratio=ch_area / float(nd["area"]), n_child=len(nd["children"]),
                     peak_minus_lvl=f["p_max"] - cfg.levels[j], depth=nd["maxk"] - k, root_area=rarea, root_pmax=root_pmax)
            row = dict(root=r, node=node_id + i, parent=(node_id + nd["parent"]) if nd["parent"] >= 0 else -1, k=k, j=j)
            row.update(f)
            if gt_labs is not None:
                for q, (bi, mi, gid) in enumerate(_iou_targets(m, x0, y0, gt_labs, gt_areas)):
                    row[f"biou{q}"] = bi; row[f"miou{q}"] = mi; row[f"gid{q}"] = gid
            rows.append(row); masks.append((x0, y0, m))
        node_id += len(nodes)
    return rows, masks


def select(node: np.ndarray, parent: np.ndarray, k: np.ndarray, score: np.ndarray, lam: float) -> np.ndarray:
    """PQ-optimal pixel-disjoint subset of the candidates of ONE image (boolean mask over rows).

    Rows sharing a ``node`` id are alternative masks of the same tree node (plain / grouped).
    value(node) = max(best alternative - lam/2, sum of the children's values, 0).
    """
    gain = score - lam / 2.0
    order = np.argsort(-gain, kind="stable")
    best_row = {}
    for i in order:                                   # best alternative per node
        best_row.setdefault(int(node[i]), int(i))
    nodes = sorted(best_row, key=lambda n: -k[best_row[n]])    # deepest level first
    par = {int(node[i]): int(parent[i]) for i in best_row.values()}
    child_sum = {n: 0.0 for n in nodes}
    value, take = {}, {}
    children = {n: [] for n in nodes}
    for n in nodes:
        own = float(gain[best_row[n]])
        cs = child_sum[n]
        if own >= cs and own > 0:
            value[n], take[n] = own, True
        else:
            value[n], take[n] = max(cs, 0.0), False
        p = par[n]
        if p >= 0:
            child_sum[p] += value[n]; children[p].append(n)
    chosen = np.zeros(len(node), bool)
    stack = [n for n in nodes if par[n] < 0]
    while stack:
        n = stack.pop()
        if take[n]:
            chosen[best_row[n]] = True
        else:
            stack.extend(children[n])
    return chosen


def paint(masks: list, chosen_idx, shape) -> tuple[np.ndarray, np.ndarray]:
    """Selected candidates -> (label map, stats[n, 4] = x, y, w, h); later masks never overwrite."""
    lab = np.zeros(shape, np.int32)
    stats = []
    for i in chosen_idx:
        x0, y0, m = masks[i]
        h, w = m.shape
        view = lab[y0:y0 + h, x0:x0 + w]
        free = (m > 0) & (view == 0)
        if not free.any():
            continue
        view[free] = len(stats) + 1
        ys, xs = np.nonzero(free)
        stats.append((x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return lab, np.asarray(stats, np.int64).reshape(-1, 4)
