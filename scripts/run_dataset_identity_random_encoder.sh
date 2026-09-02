#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1}" >&2
  exit 2
fi

DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/retrieval_ablation_suite"

case "${DATASET}" in
  ETTh1)
    ENCODER_SCRIPT="${SCRIPT_DIR}/ETTh1/run_ema_mlp_repeat_seqeqpred_all.sh"
    ;;
  ETTm1)
    ENCODER_SCRIPT="${SCRIPT_DIR}/ETTm1/run_ema_mlp_repeat_seqeqpred_all.sh"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

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

echo "[${DATASET} 1/6] Current learned EMA encoder experiments"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}" \
TEACHER_TEMPS="${TEACHER_TEMPS:-0.07}" \
  "${ENCODER_SCRIPT}" \
  2>&1 | tee "${LOG_DIR}/encoder.log"

echo "[${DATASET} 2/6] Encoder-free identity retrieval experiments"
"${SCRIPT_DIR}/run_identity_distribution_seqeqpred.sh" "${DATASET}" \
  2>&1 | tee "${LOG_DIR}/identity.log"

echo "[${DATASET} 3/6] Frozen random MLP encoder experiments"
"${SCRIPT_DIR}/run_random_retrieval_backbone_seed_average.sh" "${DATASET}" \
  2>&1 | tee "${LOG_DIR}/random.log"

echo "[${DATASET} 4/6] No-retrieval baseline experiments"
"${SCRIPT_DIR}/run_no_retrieval_seqeqpred.sh" "${DATASET}" \
  2>&1 | tee "${LOG_DIR}/no_retrieval.log"

echo "[${DATASET} 5/6] Future-MSE-teacher learned encoder experiments"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}" \
TEACHER_TEMP="${TEACHER_TEMP:-0.07}" \
  "${SCRIPT_DIR}/run_mse_teacher_encoder_seqeqpred.sh" "${DATASET}" \
  2>&1 | tee "${LOG_DIR}/mse_teacher_encoder.log"

echo "[${DATASET} 6/6] Full-oracle retrieval experiments"
"${SCRIPT_DIR}/run_full_oracle_seqeqpred.sh" "${DATASET}" \
  2>&1 | tee "${LOG_DIR}/full_oracle.log"

echo "${DATASET} six-condition retrieval ablation suite completed: ${LOG_DIR}"
