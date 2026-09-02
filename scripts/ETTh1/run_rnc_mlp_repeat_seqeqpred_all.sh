#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  STAGE1_MODEL_ID="CARTS_stage1_rnc_tau02_mlp_linear_ETTh1_${PRED_LEN}"
  STAGE1_DES="stage1_rnc_tau02_mlp_linear_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}"
  STAGE1_SETTING="stage1_${STAGE1_MODEL_ID}_RelationStage1_ETTh1_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${STAGE1_DES}_0"
  STAGE1_CKPT_PATH="./checkpoints/stage1/ETTh1/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING}/checkpoint.pth"

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
    --candidate_mask raft
    --relation_input_space delta_last
    --relation_teacher_space delta_last
    --relation_value_space delta_last
    --source_mode all
    --target_mode all
    --relation_encoder_type mlp
    --relation_self_fill linear
  )

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][rnc_tau02_mlp_linear] Stage 1"
  python -u run.py \
    --task_name stage1_relation \
    --learning_rate 1e-3 \
    --is_training 1 \
    --model_id "${STAGE1_MODEL_ID}" \
    --model RelationStage1 \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --tau_student 0.10 \
    --stage1_loss_mode rnc \
    --rnc_quality_source future_mse \
    --rnc_temperature 0.2 \
    --des "${STAGE1_DES}"

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][rnc_tau02_mlp_linear] Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
    exit 1
  fi

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][rnc_tau02_mlp_linear] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_from_rnc_tau02_mlp_linear_ETTh1_${PRED_LEN}" \
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
    --des "stage2_from_rnc_tau02_mlp_linear_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
