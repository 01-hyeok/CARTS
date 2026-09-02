#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

SEQ_LEN=720
PRED_LENS=(${PRED_LENS:-96 192 336 720})

for PRED_LEN in "${PRED_LENS[@]}"; do
  STAGE1_CKPT_PATH="./checkpoints/stage1/ETTh1/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_CARTS_stage1_mse_mlp_linear_ETTh1_${PRED_LEN}_RelationStage1_ETTh1_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_stage1_mse_mlp_linear_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}_0/checkpoint.pth"

  COMMON_ARGS=(
    --data ETTh1
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
    --data_path ETTh1.csv
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
    --relation_encoder_type mlp
    --relation_self_fill linear
    --candidate_mask raft
    --relation_input_space delta_last
    --relation_teacher_space delta_last
    --relation_value_space delta_last
    --source_mode all
    --target_mode all
  )

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][mse_mlp_linear_lambda0] Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
    exit 1
  fi

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][mse_mlp_linear_lambda0] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_mse_mlp_linear_lambda0_ETTh1_${PRED_LEN}" \
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
    --fixed_lambda 0.0 \
    --disable_retrieval 1 \
    --des "stage2_mse_mlp_linear_lambda0_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
