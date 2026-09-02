#!/bin/bash
set -euo pipefail

# Controls A (self-only) and B (random source) on ETTh1 then ETTm1.
# Both run on top of the current learned EMA encoder so the source-selection
# rule is the only thing that varies against the retrieval ablation suite.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/source_graph_ablation_sequence"
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
export STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
export TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
export VARIANTS="${VARIANTS:-self random}"
# The random draw is the treatment, so average it over several graphs.
export RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})

for DATASET in "${DATASETS[@]}"; do
  echo "[${DATASET}] Source-selection controls (self-only / random source)"
  "${SCRIPT_DIR}/run_source_graph_ablation_seqeqpred.sh" "${DATASET}" \
    2>&1 | tee "${LOG_DIR}/${DATASET,,}_source_graph_ablation.log"
done

echo "Source-graph ablation completed for ${DATASETS[*]}: ${LOG_DIR}"
