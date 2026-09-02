#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEQ_LEN="${SEQ_LEN:-720}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
RELATION_TARGET_CHUNK_SIZE="${RELATION_TARGET_CHUNK_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RELATION_GRAPH_PATH="${RELATION_GRAPH_PATH:-./metrics/relation_graphs/electricity/pearson_self_top${RELATION_TOP_N}.json}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  STAGE1_CKPT_PATH="./checkpoints/stage1/custom/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_CARTS_stage1_mse_mlp_linear_Electricity_${PRED_LEN}_RelationStage1_custom_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_stage1_mse_mlp_linear_Electricity_seq${SEQ_LEN}_pred${PRED_LEN}_topn${RELATION_TOP_N}_0/checkpoint.pth"

  COMMON_ARGS=(
    --data custom
    --root_path ../Dataset/Time-Series-Library_dataset/electricity/
    --data_path electricity.csv
    --features M
    --target OT
    --seq_len "${SEQ_LEN}"
    --label_len 0
    --pred_len "${PRED_LEN}"
    --enc_in 321
    --batch_size "${BATCH_SIZE}"
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
    --source_mode auto
    --relation_top_n "${RELATION_TOP_N}"
    --relation_graph_threshold 21
    --relation_graph_path "${RELATION_GRAPH_PATH}"
    --relation_target_chunk_size "${RELATION_TARGET_CHUNK_SIZE}"
    --target_mode all
    --relation_encoder_type mlp
    --relation_self_fill linear
    --focus_channel OT
  )

  echo "[Electricity][seq${SEQ_LEN}_pred${PRED_LEN}][mse_mlp_linear][topn${RELATION_TOP_N}] Stage 1"
  python -u run.py \
    --task_name stage1_relation \
    --learning_rate 1e-3 \
    --is_training 1 \
    --model_id "CARTS_stage1_mse_mlp_linear_Electricity_${PRED_LEN}" \
    --model RelationStage1 \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --stage1_key_chunk_size 256 \
    --stage1_probe_vis 0 \
    --tau_student 0.10 \
    --tau_teacher 0.1 \
    --teacher_mse_space normalized \
    --stage1_teacher_mode mse \
    --des "stage1_mse_mlp_linear_Electricity_seq${SEQ_LEN}_pred${PRED_LEN}_topn${RELATION_TOP_N}"

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[Electricity][seq${SEQ_LEN}_pred${PRED_LEN}] Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
    exit 1
  fi

  echo "[Electricity][seq${SEQ_LEN}_pred${PRED_LEN}][mse_mlp_linear][topn${RELATION_TOP_N}] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_mse_mlp_linear_Electricity_${PRED_LEN}" \
    --model RelationStage2 \
    --base_head_mode shared_target_linear \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 1 \
    --memory_chunk_size 256 \
    --top_k 10 \
    --tau_topk 0.10 \
    --relation_mixer_input retrieved \
    --fusion_mode residual \
    --gate_mode scalar \
    --des "stage2_mse_mlp_linear_Electricity_seq${SEQ_LEN}_pred${PRED_LEN}_topk10_topn${RELATION_TOP_N}"
done
