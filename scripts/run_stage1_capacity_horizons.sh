#!/bin/bash
# Is "capacity is not the bottleneck" true beyond ETTh1/96?
#
# The original capacity result was a single setting, and a negative claim from
# n=1 does not travel. Long horizons differ where it matters most: the fraction
# of candidates with positive utility falls from 0.53 at pred 96 to 0.09 at 336
# and 0.03 at 720, so the target's ambiguity structure is not the same problem.
#
# d_model=32 is a positive control, not a data point. If 128 -> 512 is flat, the
# flatness has two readings -- capacity is not binding, or the metric cannot see
# capacity at all. A 32-wide encoder must be visibly worse, or the measurement
# proves nothing.
#
#   ARMS  32:64  128:256  256:512  512:1024
#   PRED  96  192  336  720        (seq_len = pred_len, as everywhere else)
#
# Stage-1 only; no Stage-2 is trained. Protocol is otherwise identical to
# scripts/run_stage1_capacity_scaling.sh so the new rows are comparable to the old.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-ETTh1}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})
export CONFIGS="${CONFIGS:-32:64 128:256 256:512 512:1024}"
export OUT_CSV="${OUT_CSV:-./metrics/stage1_capacity_horizons.csv}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export MAX_QUERIES="${MAX_QUERIES:-512}"
export SEED="${SEED:-0}"

for PRED in "${PRED_LENS[@]}"; do
  echo "=============================================================="
  echo "[capacity-horizons] ${DATASET} / pred ${PRED} / configs: ${CONFIGS}"
  echo "=============================================================="
  DATASET="${DATASET}" PRED_LEN="${PRED}" \
    LOG_DIR="${LOG_DIR_BASE:-./logs/stage1_capacity_horizons}/${DATASET}_pred${PRED}" \
    bash scripts/run_stage1_capacity_scaling.sh \
    || echo "[FAILED] ${DATASET}/pred${PRED}"
done
echo "csv: ${OUT_CSV}"
