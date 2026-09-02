#!/bin/bash
set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

DATASET="${1:-ETTh1}"
case "${DATASET}" in
  ETTh1)
    DATA_PATH="ETTh1.csv"
    ;;
  ETTm1)
    DATA_PATH="ETTm1.csv"
    ;;
  *)
    echo "Usage: $0 {ETTh1|ETTm1}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/identity_retrieval"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  EXPERIMENT="identity_raw_relation_top${RELATION_TOP_N}"
  LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}.log"

  echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][encoder-free identity] Stage-2"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
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
    --relation_value_space delta_last \
    --source_mode auto \
    --relation_top_n "${RELATION_TOP_N}" \
    --target_mode all \
    --base_head_mode shared_target_linear \
    --stage1_encoder_init none \
    --stage2_retrieval_backbone identity \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --fusion_mode raft_concat \
    --oracle_candidate_eval 1 \
    --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
    2>&1 | tee "${LOG_PATH}"
done

echo "Encoder-free identity retrieval logs: ${LOG_DIR}"
