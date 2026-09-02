#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
INFONCE_TOP_K="${INFONCE_TOP_K:-10}"
INFONCE_WEIGHT="0.5"
INFONCE_POSITIVE_SOURCE="${INFONCE_POSITIVE_SOURCE:-target_mse}"
case "${INFONCE_POSITIVE_SOURCE}" in
  target_mse)
    POSITIVE_NAME="target_mse_topk"
    ;;
  ema_cosine)
    POSITIVE_NAME="branch_ema_cosine_topk"
    ;;
  *)
    echo "Unsupported INFONCE_POSITIVE_SOURCE: ${INFONCE_POSITIVE_SOURCE}" >&2
    exit 2
    ;;
esac
EXPERIMENT="ema_branchwise_kl_infonce05_${POSITIVE_NAME}_mlp_linear_top${RELATION_TOP_N}_tau_s0p10_t0p10_k0p10"
SETTING_COMPONENT_MAX_BYTES=200

shorten_path_component() {
  local value="$1"
  local byte_length
  byte_length="$(printf '%s' "${value}" | wc -c)"

  if (( byte_length <= SETTING_COMPONENT_MAX_BYTES )); then
    printf '%s' "${value}"
    return
  fi

  local digest
  local prefix_max_bytes
  local prefix
  digest="$(printf '%s' "${value}" | sha256sum)"
  digest="${digest%% *}"
  digest="${digest:0:12}"
  prefix_max_bytes=$((SETTING_COMPONENT_MAX_BYTES - ${#digest} - 1))
  prefix="${value:0:${prefix_max_bytes}}"
  while [[ "${prefix}" == *'.' || "${prefix}" == *' ' || "${prefix}" == *'_' || "${prefix}" == *'-' ]]; do
    prefix="${prefix%?}"
  done
  printf '%s_%s' "${prefix}" "${digest}"
}

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  STAGE1_MODEL_ID="CARTS_stage1_${EXPERIMENT}_ETTh1_${PRED_LEN}"
  STAGE1_DES="stage1_${EXPERIMENT}_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}"
  STAGE1_SETTING_FULL="stage1_${STAGE1_MODEL_ID}_RelationStage1_ETTh1_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${STAGE1_DES}_0"
  STAGE1_SETTING="$(shorten_path_component "${STAGE1_SETTING_FULL}")"
  STAGE1_CKPT_PATH="./checkpoints/stage1/ETTh1/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING}/checkpoint.pth"
  LEGACY_STAGE1_CKPT_PATH="./checkpoints/stage1/ETTh1/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING_FULL}/checkpoint.pth"

  # Checkpoints created before setting-name shortening remain reusable.
  if [ ! -f "${STAGE1_CKPT_PATH}" ] && [ -f "${LEGACY_STAGE1_CKPT_PATH}" ]; then
    STAGE1_CKPT_PATH="${LEGACY_STAGE1_CKPT_PATH}"
  fi

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
    --source_mode auto
    --relation_top_n "${RELATION_TOP_N}"
    --target_mode all
    --relation_encoder_type mlp
    --relation_self_fill linear
  )

  if [ -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}] Reusing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
  else
    echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][EMA MLP-linear 0.5 KL + 0.5 InfoNCE] Stage 1"
    python -u run.py \
      --task_name stage1_relation \
      --learning_rate 1e-3 \
      --is_training 1 \
      --model_id "${STAGE1_MODEL_ID}" \
      --model RelationStage1 \
      "${COMMON_ARGS[@]}" \
      --train_epochs 10 \
      --patience 5 \
      --top_k 10 \
      --tau_student 0.10 \
      --tau_teacher 0.1 \
      --teacher_mse_space normalized \
      --stage1_teacher_mode ema_target \
      --stage1_loss_mode kl_infonce \
      --stage1_infonce_weight "${INFONCE_WEIGHT}" \
      --stage1_infonce_top_k "${INFONCE_TOP_K}" \
      --stage1_infonce_positive_source "${INFONCE_POSITIVE_SOURCE}" \
      --stage1_ema_momentum_base 0.99 \
      --stage1_ema_momentum_final 0.9995 \
      --des "${STAGE1_DES}"
  fi

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}] Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"
    exit 1
  fi

  echo "[ETTh1][seq${SEQ_LEN}_pred${PRED_LEN}][Student MLP-linear concat] Stage 2"
  python -u run.py \
    --task_name stage2_relation \
    --learning_rate 1e-2 \
    --is_training 1 \
    --model_id "CARTS_stage2_student_${EXPERIMENT}_concat_ETTh1_${PRED_LEN}" \
    --model RelationStage2 \
    --base_head_mode shared_target_linear \
    "${COMMON_ARGS[@]}" \
    --train_epochs 10 \
    --patience 5 \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
    --stage2_retrieval_encoder online \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --fusion_mode raft_concat \
    --des "stage2_student_${EXPERIMENT}_concat_ETTh1_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
