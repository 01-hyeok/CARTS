#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PRED_LENS=(${PRED_LENS:-96 192 336 720})
ENCODER_NAME="mlp_linear"
ENCODER_ARGS=(
  --relation_encoder_type mlp
  --relation_self_fill linear
)

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  STAGE1_CKPT_PATH="./checkpoints/stage1/ETTm1/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_CARTS_stage1_mse_${ENCODER_NAME}_ETTm1_${PRED_LEN}_RelationStage1_ETTm1_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_stage1_mse_${ENCODER_NAME}_ETTm1_seq${SEQ_LEN}_pred${PRED_LEN}_0/checkpoint.pth"

  COMMON_ARGS=(
    --data ETTm1
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
    --data_path ETTm1.csv
    --features M
    --seq_len "${SEQ_LEN}"
    --label_len 0
    --pred_len "${PRED_LEN}"
    --enc_in 7
    --batch_size 32
    --num_workers 0
    --d_model 128
    --n_heads 4
    --e_layers 2
    --d_ff 256
    --patch_len 16
    --stride 16
    --candidate_mask raft
    --relation_input_space delta_last
    --relation_teacher_space delta_last
    --relation_value_space delta_last
    --source_mode all
    --target_mode all
    "${ENCODER_ARGS[@]}"
  )

  echo "[ETTm1][seq${SEQ_LEN}_pred${PRED_LEN}][mse_${ENCODER_NAME}] Stage 1"
  python -u run.py \
    --task_name stage1_relation \
    --learning_rate 1e-3 \
    --is_training 1 \
    --model_id "CARTS_stage1_mse_${ENCODER_NAME}_ETTm1_${PRED_LEN}" \
    --model RelationStage1 \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --tau_student 0.10 \
    --tau_teacher 0.1 \
    --teacher_mse_space normalized \
    --stage1_teacher_mode mse \
    --des "stage1_mse_${ENCODER_NAME}_ETTm1_seq${SEQ_LEN}_pred${PRED_LEN}"

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[ETTm1][seq${SEQ_LEN}_pred${PRED_LEN}][mse_${ENCODER_NAME}] Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
    exit 1
  fi

  echo "[ETTm1][seq${SEQ_LEN}_pred${PRED_LEN}][mse_${ENCODER_NAME}] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_mse_${ENCODER_NAME}_ETTm1_${PRED_LEN}" \
    --model RelationStage2 \
    --base_head_mode shared_target_linear \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 1 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --relation_mixer_input retrieved \
    --fusion_mode residual \
    --gate_mode scalar \
    --des "stage2_mse_${ENCODER_NAME}_ETTm1_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
