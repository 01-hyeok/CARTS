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
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/mse_teacher_encoder"
SETTING_COMPONENT_MAX_BYTES=200

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

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

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
# Must match the EMA-teacher condition so the comparison isolates the teacher
# signal instead of the teacher softmax temperature.
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"

STUDENT_TEMP_TAG="${STUDENT_TEMP/./p}"
TEACHER_TEMP_TAG="${TEACHER_TEMP/./p}"
TEMP_TAG="tau_s${STUDENT_TEMP_TAG}_t${TEACHER_TEMP_TAG}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  EXPERIMENT="mse_teacher_mlp_linear_top${RELATION_TOP_N}_${TEMP_TAG}"
  STAGE1_MODEL_ID="CARTS_stage1_${EXPERIMENT}_${DATASET}_${PRED_LEN}"
  STAGE1_DES="stage1_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}"
  STAGE1_SETTING_FULL="stage1_${STAGE1_MODEL_ID}_RelationStage1_${DATASET}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${STAGE1_DES}_0"
  STAGE1_SETTING="$(shorten_path_component "${STAGE1_SETTING_FULL}")"
  STAGE1_CKPT_PATH="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING}/checkpoint.pth"
  LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}_${TEMP_TAG}.log"

  COMMON_ARGS=(
    --data "${DATASET}"
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
    --data_path "${DATA_PATH}"
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

  echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][future-MSE teacher] Stage-1"
  python -u run.py \
    --task_name stage1_relation \
    --is_training 1 \
    --model_id "${STAGE1_MODEL_ID}" \
    --model RelationStage1 \
    --learning_rate 1e-3 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --tau_student "${STUDENT_TEMP}" \
    --tau_teacher "${TEACHER_TEMP}" \
    --teacher_mse_space normalized \
    --stage1_teacher_mode mse \
    --stage1_loss_mode kl \
    --stage1_probe_vis 0 \
    --des "${STAGE1_DES}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"

  if [ ! -f "${STAGE1_CKPT_PATH}" ]; then
    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}] Missing Stage-1 checkpoint: ${STAGE1_CKPT_PATH}" >&2
    exit 1
  fi

  echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][future-MSE teacher, online student] Stage-2"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
    --model RelationStage2 \
    --learning_rate 1e-2 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --base_head_mode shared_target_linear \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
    --stage2_retrieval_encoder online \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --relation_mixer_input retrieved \
    --fusion_mode raft_concat \
    --oracle_candidate_eval 1 \
    --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${LOG_PATH}"
done

echo "Future-MSE-teacher encoder logs: ${LOG_DIR}"
