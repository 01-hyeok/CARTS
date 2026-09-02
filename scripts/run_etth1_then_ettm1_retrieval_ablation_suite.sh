#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/retrieval_ablation_suite_sequence"
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
export SEEDS="${SEEDS:-0}"

# Both distillation conditions share these temperatures so that comparing the
# EMA teacher against the future-MSE teacher isolates the teacher signal.
export STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
export TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
export TEACHER_TEMPS="${TEACHER_TEMPS:-0.07}"

# Every condition is rerun from scratch on both datasets: earlier partial runs
# used tau_teacher=0.10 and a 3-seed random-encoder protocol, so they are not
# comparable with the current settings.
DATASETS=(${DATASETS:-ETTh1 ETTm1})

for DATASET in "${DATASETS[@]}"; do
  echo "[${DATASET}] Full six-condition retrieval ablation suite"
  "${SCRIPT_DIR}/run_dataset_identity_random_encoder.sh" "${DATASET}" \
    2>&1 | tee "${LOG_DIR}/${DATASET,,}_retrieval_ablation_suite.log"
done

echo "Retrieval ablation suite completed for ${DATASETS[*]}: ${LOG_DIR}"
