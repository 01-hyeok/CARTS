#!/bin/bash
set -euo pipefail

# Single entry point for one retrieval-ablation condition on one dataset.
#
#   Usage: run_condition_seqeqpred.sh {ETTh1|ETTm1|Weather} <condition>
#   conditions: no_retrieval pearson identity random ema mse_teacher full_oracle
#
# Argument values mirror the per-condition scripts used for the ETTh1/ETTm1
# suite so results produced here stay comparable with those runs.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {ETTh1|ETTm1|Weather} {no_retrieval|pearson|identity|random|ema|mse_teacher|full_oracle}" >&2
  exit 2
fi

DATASET="$1"
CONDITION="$2"

case "${DATASET}" in
  ETTh1)
    DATA_NAME="ETTh1"; DATA_PATH="ETTh1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  ETTm1)
    DATA_NAME="ETTm1"; DATA_PATH="ETTm1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  Weather)
    DATA_NAME="custom"; DATA_PATH="weather.csv"; ENC_IN=21
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/weather/"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/${CONDITION}"
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
  local digest prefix_max_bytes prefix
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
SEED="${SEED:-0}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"

STUDENT_TEMP_TAG="${STUDENT_TEMP/./p}"
TEACHER_TEMP_TAG="${TEACHER_TEMP/./p}"
TEMP_TAG="tau_s${STUDENT_TEMP_TAG}_t${TEACHER_TEMP_TAG}"

# Shared across every retrieval-enabled condition.
shared_data_args() {
  local seq_len="$1" pred_len="$2"
  printf '%s\n' \
    --data "${DATA_NAME}" \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --features M \
    --seq_len "${seq_len}" \
    --label_len 0 \
    --pred_len "${pred_len}" \
    --enc_in "${ENC_IN}" \
    --batch_size 32 \
    --num_workers 0 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --patch_len 16 \
    --stride 16
}

shared_stage2_args() {
  printf '%s\n' \
    --base_head_mode shared_target_linear \
    --memory_cache_mode precompute \
    --refresh_memory_every_epoch 0 \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --stage2_relation_fusion gate \
    --fusion_mode raft_concat
}

shared_relation_args() {
  printf '%s\n' \
    --candidate_mask raft \
    --relation_input_space delta_last \
    --relation_value_space delta_last \
    --source_mode auto \
    --relation_top_n "${RELATION_TOP_N}" \
    --target_mode all
}

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}.log"
  mapfile -t DATA_ARGS < <(shared_data_args "${SEQ_LEN}" "${PRED_LEN}")
  mapfile -t S2_ARGS < <(shared_stage2_args)
  mapfile -t REL_ARGS < <(shared_relation_args)

  case "${CONDITION}" in

    no_retrieval)
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][no retrieval] Stage-2"
      python -u run.py \
        --task_name stage2_relation --is_training 1 --model RelationStage2 \
        --model_id "CARTS_stage2_no_retrieval_base_only_${DATASET}_${PRED_LEN}" \
        "${DATA_ARGS[@]}" \
        --learning_rate 1e-2 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --relation_input_space delta_last --relation_value_space delta_last \
        --source_mode all --target_mode all \
        --base_head_mode shared_target_linear \
        --stage1_encoder_init none --freeze_stage1_encoder 1 \
        --disable_retrieval 1 \
        --use_aux_base_loss 0 --use_aux_ret_loss 0 --beta_entropy_reg 0 \
        --memory_cache_mode precompute --fusion_mode raft_concat \
        --des "stage2_no_retrieval_base_only_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}" \
        2>&1 | tee "${LOG_PATH}"
      ;;

    pearson|identity)
      if [ "${CONDITION}" = 'pearson' ]; then
        BACKBONE=pearson; EXPERIMENT="pearson_raw_relation_top${RELATION_TOP_N}"
        LABEL='encoder-free raw Pearson'
      else
        BACKBONE=identity; EXPERIMENT="identity_raw_relation_top${RELATION_TOP_N}"
        LABEL='encoder-free identity'
      fi
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${LABEL}] Stage-2"
      python -u run.py \
        --task_name stage2_relation --is_training 1 --model RelationStage2 \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
        "${DATA_ARGS[@]}" "${REL_ARGS[@]}" "${S2_ARGS[@]}" \
        --learning_rate 1e-2 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --stage1_encoder_init none \
        --stage2_retrieval_backbone "${BACKBONE}" \
        --freeze_stage1_encoder 1 \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee "${LOG_PATH}"
      ;;

    random)
      EXPERIMENT="random_retrieval_backbone_mlp_linear_top${RELATION_TOP_N}_gate_raft_tauk0p10_seed${SEED}"
      LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}_seed${SEED}.log"
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][seed${SEED}] Random retrieval backbone"
      python -u run.py \
        --task_name stage2_relation --is_training 1 --model RelationStage2 \
        --seed "${SEED}" \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
        "${DATA_ARGS[@]}" "${REL_ARGS[@]}" "${S2_ARGS[@]}" \
        --learning_rate 1e-2 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --relation_encoder_type mlp --relation_self_fill linear \
        --stage1_encoder_init random \
        --stage2_retrieval_encoder online \
        --freeze_stage1_encoder 1 \
        --relation_mixer_input retrieved \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee "${LOG_PATH}"
      ;;

    full_oracle)
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][encoder-free full oracle] Stage-2"
      python -u run.py \
        --task_name stage2_relation --is_training 1 --model RelationStage2 \
        --model_id "CARTS_stage2_full_oracle_${DATASET}_${PRED_LEN}" \
        "${DATA_ARGS[@]}" "${REL_ARGS[@]}" "${S2_ARGS[@]}" \
        --learning_rate 1e-2 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --relation_teacher_space delta_last \
        --stage1_encoder_init none --freeze_stage1_encoder 1 \
        --relation_mixer_input retrieved \
        --stage2_oracle_train_mode full \
        --des "stage2_full_oracle_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee "${LOG_PATH}"
      ;;

    ema|mse_teacher)
      if [ "${CONDITION}" = 'ema' ]; then
        TEACHER_MODE=ema_target
        EXPERIMENT="ema_branchwise_mlp_linear_top${RELATION_TOP_N}_${TEMP_TAG}"
        EXTRA_STAGE1=(--stage1_ema_momentum_base 0.99 --stage1_ema_momentum_final 0.9995)
        LABEL='learned EMA teacher'
      else
        TEACHER_MODE=mse
        EXPERIMENT="mse_teacher_mlp_linear_top${RELATION_TOP_N}_${TEMP_TAG}"
        EXTRA_STAGE1=()
        LABEL='future-MSE teacher'
      fi
      LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}_${TEMP_TAG}.log"
      STAGE1_MODEL_ID="CARTS_stage1_${EXPERIMENT}_${DATASET}_${PRED_LEN}"
      STAGE1_DES="stage1_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}"
      STAGE1_SETTING_FULL="stage1_${STAGE1_MODEL_ID}_RelationStage1_${DATA_NAME}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${STAGE1_DES}_0"
      STAGE1_SETTING="$(shorten_path_component "${STAGE1_SETTING_FULL}")"
      STAGE1_CKPT="./checkpoints/stage1/${DATA_NAME}/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING}/checkpoint.pth"

      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${LABEL}] Stage-1"
      python -u run.py \
        --task_name stage1_relation --is_training 1 --model RelationStage1 \
        --model_id "${STAGE1_MODEL_ID}" \
        "${DATA_ARGS[@]}" "${REL_ARGS[@]}" \
        --relation_teacher_space delta_last \
        --relation_encoder_type mlp --relation_self_fill linear \
        --learning_rate 1e-3 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --tau_student "${STUDENT_TEMP}" --tau_teacher "${TEACHER_TEMP}" \
        --teacher_mse_space normalized \
        --stage1_teacher_mode "${TEACHER_MODE}" \
        --stage1_loss_mode kl \
        --stage1_probe_vis 0 \
        "${EXTRA_STAGE1[@]+"${EXTRA_STAGE1[@]}"}" \
        --des "${STAGE1_DES}" \
        2>&1 | tee "${LOG_PATH}"

      if [ ! -f "${STAGE1_CKPT}" ]; then
        echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}] Missing Stage-1 checkpoint: ${STAGE1_CKPT}" >&2
        exit 1
      fi

      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${LABEL}] Stage-2"
      python -u run.py \
        --task_name stage2_relation --is_training 1 --model RelationStage2 \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
        "${DATA_ARGS[@]}" "${REL_ARGS[@]}" "${S2_ARGS[@]}" \
        --relation_teacher_space delta_last \
        --relation_encoder_type mlp --relation_self_fill linear \
        --learning_rate 1e-2 --train_epochs "${TRAIN_EPOCHS}" --patience "${PATIENCE}" \
        --stage1_ckpt_path "${STAGE1_CKPT}" \
        --stage2_retrieval_encoder online \
        --freeze_stage1_encoder 1 \
        --relation_mixer_input retrieved \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee -a "${LOG_PATH}"
      ;;

    *)
      echo "Unsupported condition: ${CONDITION}" >&2
      exit 2
      ;;
  esac
done

echo "[${DATASET}][${CONDITION}] logs: ${LOG_DIR}"
