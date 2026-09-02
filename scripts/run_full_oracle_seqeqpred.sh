#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1}" >&2
  exit 2
fi

DATASET="$1"
case "${DATASET}" in
  ETTh1) DATA_PATH="ETTh1.csv" ;;
  ETTm1) DATA_PATH="ETTm1.csv" ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/full_oracle"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}.log"

  echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][encoder-free full oracle] Stage-2"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model_id "CARTS_stage2_full_oracle_${DATASET}_${PRED_LEN}" \
    --model RelationStage2 \
    --data "${DATASET}" \
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path "${DATA_PATH}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --batch_size 32 \
    --num_workers 0 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --patch_len 16 \
    --stride 16 \
    --learning_rate 1e-2 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --candidate_mask raft \
    --relation_input_space delta_last \
    --relation_teacher_space delta_last \
    --relation_value_space delta_last \
    --source_mode auto \
    --relation_top_n "${RELATION_TOP_N}" \
    --target_mode all \
    --base_head_mode shared_target_linear \
    --stage1_encoder_init none \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --relation_mixer_input retrieved \
    --fusion_mode raft_concat \
    --stage2_oracle_train_mode full \
    --des "stage2_full_oracle_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
    2>&1 | tee "${LOG_PATH}"
done

echo "Full-oracle logs: ${LOG_DIR}"
