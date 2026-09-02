#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
CHRONOS_DTYPE="${CHRONOS_DTYPE:-bfloat16}"
CHRONOS_BATCH_SIZE="${CHRONOS_BATCH_SIZE:-16}"
MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE:-16}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  if [ "${PRED_LEN}" -gt 512 ]; then
    SEQ_LEN=512
  else
    SEQ_LEN="${PRED_LEN}"
  fi

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][Chronos T5 base concat] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_s2_chronos_t5_base_concat_ETTh1_${PRED_LEN}" \
    --model RelationStage2 \
    --data ETTh1 \
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --batch_size "${CHRONOS_BATCH_SIZE}" \
    --num_workers 0 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --patch_len 16 \
    --stride 16 \
    --candidate_mask raft \
    --relation_input_space absolute \
    --relation_value_space delta_last \
    --source_mode auto \
    --relation_top_n "${RELATION_TOP_N}" \
    --target_mode all \
    --base_head_mode shared_target_linear \
    --stage1_encoder_init none \
    --stage2_retrieval_backbone chronos \
    --chronos_model_id "${CHRONOS_MODEL_ID}" \
    --chronos_embedding_dim 768 \
    --chronos_context_length 512 \
    --chronos_dtype "${CHRONOS_DTYPE}" \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size "${MEMORY_CHUNK_SIZE}" \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --fusion_mode raft_concat \
    --train_epochs 10 \
    --patience 5 \
    --des "s2_chronos_t5_base_top${RELATION_TOP_N}_concat_p${PRED_LEN}"
done
