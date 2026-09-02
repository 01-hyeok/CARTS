#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

SEQ_LEN=720
PRED_LENS=(${PRED_LENS:-96 192 336 720})

for PRED_LEN in "${PRED_LENS[@]}"; do
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

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][random_mlp_concat] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_random_mlp_concat_ETTh1_${PRED_LEN}" \
    --model RelationStage2 \
    --base_head_mode shared_target_linear \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --stage1_encoder_init random \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 1 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --relation_mixer_input retrieved \
    --fusion_mode raft_concat \
    --des "stage2_random_mlp_concat_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
