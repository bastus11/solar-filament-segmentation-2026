# Solar filament instance segmentation on GONG H-alpha (MAGFiLO) — Filament Segmentation Challenge 2026

End-to-end pipeline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
a dense network predicts where annotators mark filament material, a threshold component tree turns
the probability map into nested candidate masks, a small learned model estimates the *expected
matched IoU* of every candidate, and dynamic programming returns the pixel-disjoint set of masks
that maximises Panoptic Quality.

Only the competition training file is used as ground truth. The public MAGFiLO release (which
contains the test annotations) was never downloaded or used; no file-name metadata or neighbouring
frames are used at inference.

## Method in one page

1. **Targets.** Polygons are rasterised with pycocotools (identical to the organisers; verified
   against the `area` field of all 8,199 annotations). For an image with several annotators the
   target is the *fraction of annotators* that marked a pixel, so the network learns a consensus
   probability instead of one annotator's opinion. The provided `spine` polylines supervise an
   auxiliary head.
2. **Network.** U-Net with an ImageNet-pretrained encoder on the native 2048x2048 grid (random
   1024x1024 crops, rotation / scale / flip / photometric augmentation). Input = image plus two
   disk-normalised coordinate channels (position on the solar disk matters: annotators skip the
   limb). Loss = BCE + soft Dice on the soft targets; EMA weights; 5-fold cross-validation over
   images.
3. **Inference.** Full-frame, 8-view (D4) test-time augmentation, average over folds/architectures.
   The averaged map is moved 0.25 px up: annotated polygons sit ~0.25 px above the dark filament
   body, an offset a rotation-augmented network cannot represent.
4. **Candidates.** The map is thresholded at 14 levels (0.15 ... 0.80); connected components form
   a tree (each component is nested in one component of the level below). Every node is a
   candidate mask - a generous or a tight outline, a merged structure or its parts.
5. **Scoring.** PQ pools TP/FP/FN over annotator records, hence a submitted mask adds 0.5 to the
   denominator and its matched IoU (0 unless IoU > 0.5) to the numerator: a mask is worth
   submitting iff its expected matched IoU exceeds PQ/2. A LightGBM regressor predicts that
   expectation from ~55 descriptors (probability statistics, spine evidence, shape, contrast,
   position, tree structure), trained on out-of-fold candidates.
6. **Selection.** Exact dynamic programming over the tree picks the pixel-disjoint subset that
   maximises sum(score - lambda/2); `lambda` plays the role of the PQ level.

## Results (5-fold cross-validation over the 707 training images, official metric)

| stage | PQ | SQ | RQ |
|---|---|---|---|
| another annotator of the same image | 0.340 | 0.634 | 0.536 |
| threshold 0.45 + connected components >= 400 px | 0.435 | 0.682 | 0.638 |
| + hysteresis (0.45 / 0.15), tuned | 0.440 | 0.682 | 0.645 |
| component tree + learned scorer + DP selection (submitted) | **0.460 +- 0.004** | 0.685 | 0.672 |
| same candidates with true scores (upper bound) | 0.559 | | |

Folds: 0.479 / 0.458 / 0.453 / 0.459 / 0.447 (the spread of random 141-image subsets is 0.009). The +- is the
standard deviation of 1,000 bootstrap resamples of observations. Public leaderboard: **0.40** with
`submissions/v1e_std_lam0.60.csv`, which `demo.ipynb` reproduces exactly; a single fold network scores 0.38, the
same network with plain thresholding 0.36.

## What did not help (kept in `experiments/` for transparency)

BCE-only loss; sampling crops on the native pixel grid; RegNetY / ConvNeXt encoders and averaging two
encoder families (within +-0.006 on one fold); a 2.5x longer schedule; multi-scale TTA; morphological
clean-up of masks (1 px dilation -0.016, erosion -0.037); finer or coarser threshold ladders; grouping
fragments into one candidate; splitting candidates along separate spine segments; a stage-2
per-instance mask refiner; a CNN candidate scorer; chirality-aware (Left/Right) targets.

## Reproduce

```bash
pip install -r requirements.txt
export FIL_ROOT=/path/to/workdir                 # data in $FIL_ROOT/data/raw/MAGFiLO_1.0_Kaggle_2026
python scripts/prep_targets.py                   # rasterise images / targets into memmaps
for f in 0 1 2 3 4; do python scripts/train.py --exp v1_effv2s --fold $f; done
bash scripts/run_ensemble.sh v1e "0.60" v1_effv2s # OOF maps -> candidates -> scorer -> test maps -> csv
```

`demo.ipynb` runs the inference part from the stored weights and writes `submission.csv` (about 8 minutes for the
180 test frames on one RTX 5090; it expects the competition data under `data/raw/MAGFiLO_1.0_Kaggle_2026`).
`scripts/export_weights.py` collects the checkpoints and the scorer into `weights/`. `report/` holds the technical
report; `report/make_stats.py` and `report/make_figures.py` recompute its numbers and figures from the
out-of-fold maps and the candidate table written by `scripts/run_ensemble.sh`.

## Layout

```
fil/            library: data, dataset, model, inference, metric, candidates (tree + DP), RLE / submission IO
scripts/        prep_targets, train, predict_oof, predict_test, cand_build, cand_train, make_sub, run_ensemble.sh, analysis tools
experiments/    negative results and analysis scripts
report/         technical report (LaTeX + PDF) and the scripts behind its numbers and figures
submissions/    the file submitted to Kaggle
weights/        network checkpoints (5 folds) and the scorer
```

## Licence

MIT, see `LICENSE`. The pretrained encoder weights (timm `tf_efficientnetv2_s.in21k_ft_in1k`) are Apache-2.0.
