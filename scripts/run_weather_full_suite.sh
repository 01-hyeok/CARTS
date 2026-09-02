#!/bin/bash
set -euo pipefail

# Weather: the seven retrieval-ablation conditions followed by the two
# source-selection controls, run sequentially in one pass.
#
#   1 no_retrieval   base head only
#   2 pearson        encoder-free RAFT-style Pearson retrieval
#   3 identity       encoder-free cosine retrieval
#   4 random         frozen randomly initialised encoder
#   5 ema            current learned encoder (EMA embedding teacher)
#   6 mse_teacher    learned encoder with a future-MSE teacher
#   7 full_oracle    ground-truth candidate selection (upper bound)
#   A self-only      each target retrieves with its own channel only
#   B random source  target paired with arbitrary channels instead of correlated ones
#
# Conditions 5 and 6 share tau_student/tau_teacher so the comparison isolates
# the teacher signal. Weather has 21 channels, so the relation graph and the
# memory bank are considerably larger than on ETT: expect this to run long.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/Weather/full_suite_sequence"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PRED_LENS="${PRED_LENS:-96 192 336 720}"
export RELATION_TOP_N="${RELATION_TOP_N:-3}"
export SEED="${SEED:-0}"
export STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
export TEACHER_TEMP="${TEACHER_TEMP:-0.07}"

CONDITIONS=(${CONDITIONS:-no_retrieval pearson identity random ema mse_teacher full_oracle})
RUN_SOURCE_CONTROLS="${RUN_SOURCE_CONTROLS:-1}"
export VARIANTS="${VARIANTS:-self random}"
export RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"

total=${#CONDITIONS[@]}
i=0
for CONDITION in "${CONDITIONS[@]}"; do
  i=$((i + 1))
  echo "[Weather ${i}/${total}] condition: ${CONDITION}"
  "${SCRIPT_DIR}/run_condition_seqeqpred.sh" Weather "${CONDITION}" \
    2>&1 | tee "${LOG_DIR}/${CONDITION}.log"
done

if [ "${RUN_SOURCE_CONTROLS}" = '1' ]; then
  # Depends on metrics/relation_graphs/weather/pearson_self_top{N}.json, which
  # the auto source_mode conditions above will have produced by now.
  echo '[Weather] source-selection controls (self-only / random source)'
  "${SCRIPT_DIR}/run_source_graph_ablation_seqeqpred.sh" Weather \
    2>&1 | tee "${LOG_DIR}/source_graph_ablation.log"
fi

echo "Weather full suite completed: ${LOG_DIR}"
