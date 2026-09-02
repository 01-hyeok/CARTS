#!/bin/bash
set -euo pipefail

# End-to-end CARTS: one training run, random init, no staged hand-off.
#
#   loss = forecasting MSE + lambda * future-MSE KL
#
#   Pearson channel graph -> shared encoder -> cosine scores -> softmax over
#   Top-K -> relation-wise future weighted sum -> relation mixer -> concat with
#   base forecast -> linear
#
# The encoder starts from random weights and is trained in the same loop as the
# base head, gate and mixer. Pre-training the retriever first would just be the
# 2-stage pipeline again, which is the thing end-to-end is meant to replace.
#
# lambda=0 is the control that matters: the forecasting loss reaches the encoder
# only through the Top-K softmax weights, and torch.topk is not differentiable,
# so it can reweight the candidates already selected but never promote one it
# ranked out. The KL is the only term that touches every candidate.
#
# The 2-stage baseline arm is the one thing that still needs a Stage-1 run: it
# is the pipeline being compared against, so Stage-1 is trained only for it.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1}" >&2
  exit 2
fi

DATASET="$1"

case "${DATASET}" in
  ETTh1)
    DATA_NAME="ETTh1"; DATA_PATH="ETTh1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  ETTm1)
    DATA_NAME="ETTm1"; DATA_PATH="ETTm1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/end_to_end_lambda"
SETTING_COMPONENT_MAX_BYTES="${SETTING_COMPONENT_MAX_BYTES:-200}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

# run.py shortens setting names past 200 bytes; reproduce it so the checkpoint
# path the warm-up wrote is the one the end-to-end phase reads.
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 720 is in: the Chronos 512-token truncation that excluded it from the
# retrieval-backbone sweep does not apply to the Stage-1 relation encoder.
PRED_LENS=(${PRED_LENS:-96 192 336 720})
# Measured on real runs: KL lands at ~2.0 (trained) to ~3.2 (random init)
# while the forecasting MSE is ~0.24-0.43, so the two terms carry equal
# weight near lambda 0.2. 0 is the control, 0.5 over-weights retrieval -
# which is the side worth probing when the encoder starts from scratch.
LAMBDAS=(${LAMBDAS:-0 0.2 0.5})
# Teacher the retrieval KL distils from. ema is an EMA of the student itself, so
# a constant embedding is a trivial minimum of the KL; future_mse scores every
# candidate against the ground-truth future, which has no such degenerate
# solution. The two are the same objective apart from that, so they are the
# comparison that isolates the collapse.
RETRIEVAL_KL_TEACHER="${RETRIEVAL_KL_TEACHER:-ema}"
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
TAU_TOPK="${TAU_TOPK:-0.10}"
# Also run the frozen-encoder 2-stage arm, the baseline end-to-end has to beat.
RUN_TWO_STAGE_BASELINE="${RUN_TWO_STAGE_BASELINE:-1}"
# random keeps the run single-phase; checkpoint reproduces the earlier
# "2-stage then fine-tune the retriever" arms instead of true end-to-end.
E2E_ENCODER_INIT="${E2E_ENCODER_INIT:-random}"

TEMP_TAG="tau_s${STUDENT_TEMP/./p}_t${TEACHER_TEMP/./p}"

common_args_for() {
  local seq_len="$1" pred_len="$2"
  COMMON_ARGS=(
    --data "${DATA_NAME}"
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_PATH}"
    --features M
    --seq_len "${seq_len}"
    --label_len 0
    --pred_len "${pred_len}"
    --enc_in "${ENC_IN}"
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
}

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  common_args_for "${SEQ_LEN}" "${PRED_LEN}"

  # ---------------------------------------------------------------- warm-up
  WARMUP_EXP="e2e_warmup_mlp_linear_top${RELATION_TOP_N}_${TEMP_TAG}"
  STAGE1_MODEL_ID="CARTS_stage1_${WARMUP_EXP}_${DATASET}_${PRED_LEN}"
  STAGE1_DES="stage1_${WARMUP_EXP}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}"
  STAGE1_SETTING_FULL="stage1_${STAGE1_MODEL_ID}_RelationStage1_${DATA_NAME}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${STAGE1_DES}_0"
  STAGE1_SETTING="$(shorten_path_component "${STAGE1_SETTING_FULL}")"
  STAGE1_CKPT="./checkpoints/stage1/${DATA_NAME}/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE1_SETTING}/checkpoint.pth"

  # Stage-1 is trained only because the 2-stage baseline arm needs it. The
  # end-to-end arms never read it when E2E_ENCODER_INIT=random.
  NEED_STAGE1=0
  [ "${RUN_TWO_STAGE_BASELINE}" = "1" ] && NEED_STAGE1=1
  [ "${E2E_ENCODER_INIT}" = "checkpoint" ] && NEED_STAGE1=1

  if [ "${NEED_STAGE1}" = "0" ]; then
    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}] single-phase end-to-end; no Stage-1 run"
  elif [ -f "${STAGE1_CKPT}" ]; then
    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}] Stage-1 checkpoint exists, reusing (baseline arm only)"
  else
    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}] Stage-1 retriever training (baseline arm only)"
    python -u run.py \
      --task_name stage1_relation \
      --is_training 1 \
      --model RelationStage1 \
      --model_id "${STAGE1_MODEL_ID}" \
      --learning_rate 1e-3 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --tau_student "${STUDENT_TEMP}" \
      --tau_teacher "${TEACHER_TEMP}" \
      --teacher_mse_space normalized \
      --stage1_teacher_mode ema_target \
      --stage1_loss_mode kl \
      --stage1_probe_vis 0 \
      --stage1_ema_momentum_base 0.99 \
      --stage1_ema_momentum_final 0.9995 \
      --des "${STAGE1_DES}" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "${LOG_DIR}/warmup_seq${SEQ_LEN}_pred${PRED_LEN}.log"
  fi

  if [ "${NEED_STAGE1}" = "1" ] && [ ! -f "${STAGE1_CKPT}" ]; then
    echo "Missing Stage-1 checkpoint after training: ${STAGE1_CKPT}" >&2
    exit 1
  fi

  # ------------------------------------------------------- stage-2 arms
  # Each arm reads the same warm-up checkpoint. freeze=1 is the 2-stage
  # baseline; freeze=0 with a lambda is end-to-end.
  run_stage2() {
    local tag="$1" freeze="$2" kl_weight="$3" encoder_init="${4:-checkpoint}"
    # Only the arms that actually carry a KL term get the teacher suffix, so the
    # lambda=0 control and the 2-stage baseline keep their existing log paths.
    if [ "${RETRIEVAL_KL_TEACHER}" != "ema" ] && [ "${kl_weight}" != "0" ]; then
      tag="${tag}_${RETRIEVAL_KL_TEACHER}teacher"
    fi
    local init_args=(--stage1_encoder_init "${encoder_init}")
    if [ "${encoder_init}" = "checkpoint" ]; then
      init_args+=(--stage1_ckpt_path "${STAGE1_CKPT}")
    fi
    local experiment="e2e_${tag}_top${RELATION_TOP_N}_${TEMP_TAG}"
    local arm_log="${LOG_DIR}/${tag}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
    if grep -q 'Stage2 Test Final' "${arm_log}" 2>/dev/null; then
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${tag}] already finished, skipping"
      return 0
    fi
    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${tag}] Stage-2 (freeze=${freeze} lambda=${kl_weight})"
    python -u run.py \
      --task_name stage2_relation \
      --is_training 1 \
      --model RelationStage2 \
      --model_id "CARTS_stage2_${experiment}_${DATASET}_${PRED_LEN}" \
      --learning_rate 1e-2 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --tau_student "${STUDENT_TEMP}" \
      --tau_teacher "${TEACHER_TEMP}" \
      --base_head_mode shared_target_linear \
      --stage2_retrieval_backbone stage1 \
      "${init_args[@]}" \
      --freeze_stage1_encoder "${freeze}" \
      --retrieval_kl_weight "${kl_weight}" \
      --retrieval_kl_teacher "${RETRIEVAL_KL_TEACHER}" \
      --memory_cache_mode precompute \
      --refresh_memory_every_epoch "$([ "${freeze}" = "1" ] && echo 0 || echo 1)" \
      --memory_chunk_size 1024 \
      --top_k 10 \
      --tau_topk "${TAU_TOPK}" \
      --stage2_relation_fusion gate \
      --relation_mixer_input retrieved \
      --fusion_mode raft_concat \
      --oracle_candidate_eval 1 \
      --des "stage2_${experiment}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "${LOG_DIR}/${tag}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
  }

  if [ "${RUN_TWO_STAGE_BASELINE}" = "1" ]; then
    run_stage2 "two_stage_frozen" 1 0 checkpoint
  fi

  # scratch = true end-to-end; warmstart = the earlier 2-stage-then-finetune arms.
  ARM_PREFIX="e2e_scratch"
  [ "${E2E_ENCODER_INIT}" = "checkpoint" ] && ARM_PREFIX="e2e"
  for LAMBDA in "${LAMBDAS[@]}"; do
    run_stage2 "${ARM_PREFIX}_lambda${LAMBDA/./p}" 0 "${LAMBDA}" "${E2E_ENCODER_INIT}"
  done
done

echo "[${DATASET}] end-to-end lambda sweep complete: ${LOG_DIR}"
