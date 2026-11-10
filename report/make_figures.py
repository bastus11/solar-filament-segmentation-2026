"""Figures of the report (all from held-out predictions of the cross-validation models).

    python report/make_stats.py              # once, writes report/stats.pkl
    python report/make_figures.py            # writes report/figures/*.pdf (+ png previews)
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import figlib as F

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.5})
ANN = ["#ffd000", "#ff40ff", "#30e030"]            # annotator colours
OURS = "#00e5ff"


def contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :] for c in cs if len(c) >= 3]


def draw(ax, pts, **kw):
    p = np.vstack([pts, pts[:1]])
    ax.plot(p[:, 0], p[:, 1], **kw)


def crop_ax(ax, rgb, title=None):
    ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, rgb.shape[1] - 0.5); ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
    if title: ax.set_title(title, fontsize=7, pad=2.5)


def fig_pipeline(stem, x0, y0, s, name="fig_pipeline"):
    img, pr, df, masks, lab, sel = F.held_out(stem)
    rgb = F.show(img, x0, y0, s)
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.02))
    plt.subplots_adjust(left=0.004, right=0.996, top=0.715, bottom=0.008, wspace=0.07)
    # schematic of the whole data flow above the image panels
    boxes = [("training (once)\n707 frames, polygons of 1-3 annotators\n$\\rightarrow$ soft targets (fraction of annotators) + spines", 0.005, 0.262),
             ("5-fold U-Net ensemble (EfficientNetV2-S)\ninput: frame + disk coordinates\n8-view TTA, 0.25 px label alignment", 0.287, 0.232),
             ("candidates: components of 14 thresholds\nLightGBM on out-of-fold candidates\n$\\rightarrow$ expected matched IoU $s$", 0.539, 0.228),
             ("dynamic programming on the tree:\nmax $\\sum (s-\\lambda/2)$, pixel-disjoint\n$\\rightarrow$ COCO RLE per filament", 0.787, 0.208)]
    for k, (txt, x, w) in enumerate(boxes):
        fig.text(x + w / 2, 0.905, txt, ha="center", va="center", fontsize=5.6, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f8" if k else "#f6efe6", ec="#4a6fa5" if k else "#a57c4a", lw=0.6))
        if k:
            px = boxes[k - 1][1] + boxes[k - 1][2]
            fig.add_artist(FancyArrowPatch((px + 0.004, 0.905), (x - 0.004, 0.905), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5, lw=0.6, color="k"))
    crop_ax(axes[0], rgb, "(a) H$\\alpha$ frame")
    P = pr[0][y0:y0 + s, x0:x0 + s].astype(np.float32) / 255
    axes[1].imshow(P, cmap="magma", vmin=0, vmax=1, interpolation="nearest"); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("(b) consensus probability", fontsize=7, pad=2.5)
    crop_ax(axes[2], (rgb * 0.55).astype(np.uint8), "(c) threshold tree")
    lv = [0.15, 0.30, 0.45, 0.60, 0.75]; cm = plt.get_cmap("viridis")
    for k, t in enumerate(lv):
        for c in contours(P >= t):
            draw(axes[2], c, color=cm(0.15 + 0.8 * k / (len(lv) - 1)), lw=0.55)
    for k, t in enumerate(lv):
        axes[2].plot([], [], color=cm(0.15 + 0.8 * k / (len(lv) - 1)), lw=1.0, label=f"{t:.2f}")
    axes[2].legend(loc="lower left", fontsize=4.6, ncol=5, frameon=True, framealpha=0.75, handlelength=0.9, columnspacing=0.6, borderpad=0.25, handletextpad=0.3)
    # (d) score of the best candidate of every rejected root vs. the selected candidates
    crop_ax(axes[3], (rgb * 0.55).astype(np.uint8), "(d) expected matched IoU")
    inside = lambda m: (m[0] + m[2].shape[1] / 2 > x0 + 8) and (m[0] + m[2].shape[1] / 2 < x0 + s - 8) and (m[1] + m[2].shape[0] / 2 > y0 + 8) and (m[1] + m[2].shape[0] / 2 < y0 + s - 8)
    best_rej = {}
    for i in np.flatnonzero(~df.chosen.values):
        r = int(df.root.values[i])
        if r not in set(df.root.values[df.chosen.values]) and inside(masks[i]) and df.area.values[i] >= 150:
            if r not in best_rej or df.score.values[i] > df.score.values[best_rej[r]]: best_rej[r] = i
    for i in best_rej.values():
        mx, my, m = masks[i]
        for c in contours(m): draw(axes[3], c + [mx - x0, my - y0], color="#ff5050", lw=0.6, ls=(0, (2, 1)))
        axes[3].text(mx - x0 + m.shape[1] / 2, my - y0 - 5, f"{df.score.values[i]:.2f}", color="#ff8080", fontsize=5, ha="center", va="bottom")
    for i in sel:
        if not inside(masks[i]): continue
        mx, my, m = masks[i]
        for c in contours(m): draw(axes[3], c + [mx - x0, my - y0], color=OURS, lw=0.7)
        axes[3].text(mx - x0 + m.shape[1] / 2, my - y0 - 5, f"{df.score.values[i]:.2f}", color=OURS, fontsize=5, ha="center", va="bottom")
    crop_ax(axes[4], rgb, "(e) selection vs. annotators")
    for k, ps in enumerate(F.polys(stem)):
        for p in ps: draw(axes[4], p - [x0, y0], color=ANN[k], lw=0.45)
    sub = lab[y0:y0 + s, x0:x0 + s]
    for i in np.unique(sub):
        if i:
            for c in contours(sub == i): draw(axes[4], c, color=OURS, lw=0.8)
    for a, b in zip(axes[:-1], axes[1:]):
        fig.add_artist(FancyArrowPatch((a.get_position().x1 + 0.001, 0.36), (b.get_position().x0 - 0.001, 0.36), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=5, lw=0.6, color="k"))
    fig.savefig(OUT / f"{name}.pdf", dpi=300); fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig)


def fig_examples(items, name="fig_examples"):
    fig, axes = plt.subplots(1, len(items), figsize=(7.0, 7.0 / len(items) + 0.02))
    plt.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005, wspace=0.03)
    for ax, (stem, x0, y0, s) in zip(axes, items):
        img, pr, df, masks, lab, sel = F.held_out(stem)
        crop_ax(ax, F.show(img, x0, y0, s))
        for k, ps in enumerate(F.polys(stem)):
            for p in ps: draw(ax, p - [x0, y0], color=ANN[k], lw=0.5)
        sub = lab[y0:y0 + s, x0:x0 + s]
        for i in np.unique(sub):
            if i:
                for c in contours(sub == i): draw(ax, c, color=OURS, lw=0.9)
        ax.text(0.03, 0.97, f"({'abcdefgh'[list(axes).index(ax)]})", transform=ax.transAxes, color="w", fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.55))
    fig.savefig(OUT / f"{name}.pdf", dpi=300); fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig)


def fig_eval(name="fig_eval"):
    from pycocotools import mask as mu
    from fil import data, metric
    res = pickle.load(open(Path(__file__).parent / "stats.pkl", "rb"))
    bins = [150, 300, 600, 1200, 2500, 5000, 40000]; mid = [np.sqrt(a * b) for a, b in zip(bins[:-1], bins[1:])]
    multi = [r for r in res if r["nrec"] >= 2]
    ga = np.concatenate([rec["gt_area"] for r in multi for rec in r["recs"]]); gb = np.concatenate([rec["gt_best"] for r in multi for rec in r["recs"]])
    ours = [(gb[(ga >= a) & (ga < b)] > 0.5).mean() for a, b in zip(bins[:-1], bins[1:])]
    # the same filaments matched by another annotator of the same image
    recs = data.load_records(); by = data.records_by_stem(recs); gt = data.cached_gt_rles(recs)
    A, Bst = [], []
    for stem, rs in by.items():
        if len(rs) < 2: continue
        for a in rs:
            for b in rs:
                if a is b or not gt[a.record_id]: continue
                iou = metric.iou_matrix(gt[a.record_id], gt[b.record_id])
                A.append(mu.area(gt[a.record_id])); Bst.append(iou.max(1) if iou.shape[1] else np.zeros(len(gt[a.record_id])))
    A, Bst = np.concatenate(A), np.concatenate(Bst)
    other = [(Bst[(A >= a) & (A < b)] > 0.5).mean() for a, b in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(1, 2, figsize=(3.33, 1.55), gridspec_kw={"width_ratios": [1.25, 1]})
    plt.subplots_adjust(left=0.115, right=0.995, top=0.93, bottom=0.235, wspace=0.42)
    ax[0].plot(mid, ours, "o-", color="#0077b6", ms=2.5, lw=0.9, label="this work")
    ax[0].plot(mid, other, "s--", color="#d1495b", ms=2.5, lw=0.9, label="another annotator")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 0.85); ax[0].set_xlabel("annotated filament area (px)", labelpad=1); ax[0].set_ylabel("matched (IoU $>$ 0.5)", labelpad=1)
    ax[0].legend(fontsize=5.5, frameon=False, loc="lower right"); ax[0].grid(alpha=0.25, lw=0.4); ax[0].tick_params(length=2, pad=1.5)

    def pq(rs):
        S = TP = FP = FN = 0.0
        for r in rs:
            for rec in r["recs"]:
                m = rec["gt_best"] > 0.5; S += rec["gt_best"][m].sum(); TP += m.sum(); FN += (~m).sum(); FP += (rec["pred_best"] <= 0.5).sum()
        return S / (TP + 0.5 * FP + 0.5 * FN)
    folds = [pq([r for r in res if r["fold"] == f]) for f in range(5)]
    rng = np.random.default_rng(0); boot = [pq([res[k] for k in rng.integers(0, len(res), len(res))]) for _ in range(400)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ax[1].bar(range(5), folds, color="#0077b6", width=0.62)
    ax[1].axhspan(lo, hi, color="k", alpha=0.13, lw=0); ax[1].axhline(pq(res), color="k", lw=0.7)
    ax[1].axhline(0.340, color="#d1495b", lw=0.8, ls="--"); ax[1].text(4.45, 0.352, "annotator vs. annotator", color="#d1495b", fontsize=5.5, ha="right", va="bottom")
    ax[1].set_ylim(0.30, 0.50); ax[1].set_xticks(range(5)); ax[1].set_xlabel("validation fold", labelpad=1); ax[1].set_ylabel("PQ", labelpad=1)
    ax[1].tick_params(length=2, pad=1.5); ax[1].grid(axis="y", alpha=0.25, lw=0.4)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220); plt.close(fig)
    print("fig_eval: ours", np.round(ours, 3), "other annotator", np.round(other, 3), "folds", np.round(folds, 4), "CI", round(lo, 4), round(hi, 4))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "pipeline"):
        fig_pipeline("20120610175114Mh", 1280, 576, 448)
    if which in ("all", "examples"):
        fig_examples([("20161224024154Uh", 832, 512, 448), ("20121218063534Lh", 640, 576, 448), ("20110923165814Mh", 512, 512, 448), ("20150228082414Th", 1088, 704, 448)])
    if which in ("all", "eval"):
        fig_eval()
