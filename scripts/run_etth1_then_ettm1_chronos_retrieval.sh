#!/bin/bash
set -euo pipefail

# Chronos retrieval: frozen (RAF-style) vs fine-tuned encoder, plus the
# random-init control, on ETTh1 then ETTm1.
#
# Frozen runs first because it is cheap and validates the Chronos install and
# the embedding dimensions before the expensive fine-tuning runs start.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/chronos_retrieval_sequence"
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
export CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
export CHRONOS_LR="${CHRONOS_LR:-}"
export MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE:-256}"
export TAU_TOPK="${TAU_TOPK:-0.10}"
export CHRONOS_PROJECTION_MODE="${CHRONOS_PROJECTION_MODE:-cross_only}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
MODES=(${MODES:-frozen finetune random})

for DATASET in "${DATASETS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "[${DATASET}] Chronos retrieval mode: ${MODE}"
    "${SCRIPT_DIR}/run_chronos_retrieval_seqeqpred.sh" "${DATASET}" "${MODE}" \
      2>&1 | tee "${LOG_DIR}/${DATASET,,}_chronos_${MODE}.log"
  done
done

echo "Chronos retrieval runs completed for ${DATASETS[*]}: ${LOG_DIR}"
