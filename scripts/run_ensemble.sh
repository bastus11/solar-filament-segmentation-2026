#!/bin/bash
# End-to-end: D4-TTA out-of-fold maps -> candidate table -> scorers -> test maps -> submission files.
# usage: run_ensemble.sh <tag> "<lambda list for the standard scorer>" <exp1> [<exp2> ...]
set -eo pipefail
cd "$(dirname "$0")/.."
TAG=$1; LAMS=$2; shift 2
EXPS=("$@")
D4=()
for E in "${EXPS[@]}"; do
  for F in 0 1 2 3 4; do
    if [ ! -f ${FIL_ROOT:-/workspace}/runs/${E}_d4/oof_fold${F}.npy ]; then
      python -u scripts/predict_oof.py --exp $E --fold $F --tta d4 --out ${E}_d4 2>&1 | grep -v -i warn | tail -1
    fi
  done
  D4+=("${E}_d4")
done
CK=()
for E in "${EXPS[@]}"; do for F in 0 1 2 3 4; do CK+=("${FIL_ROOT:-/workspace}/runs/$E/fold$F.pt"); done; done
if [ ! -f ${FIL_ROOT:-/workspace}/runs/test/$TAG.npy ]; then
  python -u scripts/predict_test.py --ckpts "${CK[@]}" --out $TAG --tta d4 2>&1 | grep -v -i warn | tail -1
fi
echo GPU_PART_DONE $TAG
python -u scripts/sweep_pp.py --exp "${D4[0]}" 2>&1 | grep -A3 "best configs"
python -u scripts/cand_build.py --exps "${D4[@]}" --out $TAG 2>&1 | tail -1
echo "--- scorer: all annotated filaments"
python -u scripts/cand_train.py --cand $TAG --save $TAG 2>&1 | grep -E "t=0.45|t=0.50|rmse|lambda|oracle|saved"
echo "--- scorer: Left/Right filaments only (PQ against Left/Right ground truth)"
python -u scripts/cand_train.py --cand $TAG --target lr --save ${TAG}_lr 2>&1 | grep -E "rmse|lambda|oracle|saved"
for L in $LAMS; do
  python -u scripts/make_sub.py --prob $TAG --out ${TAG}_std_lam${L} --scorer $TAG --lam $L 2>&1 | tail -1
done
python -u scripts/make_sub.py --prob $TAG --out ${TAG}_lr_lam0.50 --scorer ${TAG}_lr --lam 0.50 2>&1 | tail -1
echo PIPELINE_DONE $TAG
