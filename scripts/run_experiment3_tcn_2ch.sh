#!/usr/bin/env bash
set -uo pipefail

# Experiment 3: does a TCN retrieval encoder fed delta_last AND the first-order
# difference as two input channels retrieve more forecasting-useful Top-K
# candidates than the current MLP encoder fed delta_last alone?
#
# Only the encoder architecture and the encoder input representation change
# across arms. Retrieval pipeline, candidate memory, masking, teacher, KL loss,
# Top-K and the forecasting module are held at the Experiment 2 settings, so the
# 2x2 below isolates the two factors:
#
#   arm             encoder   input
#   mlp_delta_last  mlp       delta_last          <- baseline (current system)
#   mlp_2ch         mlp       delta_last + diff1  <- input only
#   tcn_delta_last  tcn       delta_last          <- encoder only
#   tcn_2ch         tcn       delta_last + diff1  <- both (the hypothesis)
#
# Conventions follow the repository's self-only sweeps: arm-first execution,
# resumable Stage-2 logs, reusable Stage-1 checkpoints, per-arm failure
# isolation, and seq_len == pred_len.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/logs/experiment3_tcn_2ch_self_only_seed0}"
mkdir -p "${RESULT_ROOT}"
DRIVER_LOG="${DRIVER_LOG:-${RESULT_ROOT}/driver.log}"
EVENT_LOG="${EVENT_LOG:-${RESULT_ROOT}/events.log}"
STATE_FILE="${STATE_FILE:-${RESULT_ROOT}/current_state}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

exec > >(tee -a "${DRIVER_LOG}") 2>&1

CURRENT_RUN="driver_init"
HEARTBEAT_PID=""

set_current() {
  CURRENT_RUN="$1"
  printf '%s\n' "${CURRENT_RUN}" > "${STATE_FILE}"
}

log_event() {
  printf '%s | pid=%s ppid=%s | %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "${PPID}" "$*" \
    | tee -a "${EVENT_LOG}"
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

on_signal() {
  local signal="$1"
  log_event "SIGNAL=${signal} current=${CURRENT_RUN}"
  exit 128
}

on_exit() {
  local status=$?
  trap - EXIT
  stop_heartbeat
  log_event "EXIT status=${status} current=${CURRENT_RUN}"
  exit "${status}"
}

trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

(
  while true; do
    sleep "${HEARTBEAT_SECONDS}" || exit 0
    heartbeat_current="unknown"
    [[ -r "${STATE_FILE}" ]] && read -r heartbeat_current < "${STATE_FILE}"
    log_event "HEARTBEAT current=${heartbeat_current}"
  done
) &
HEARTBEAT_PID=$!
set_current "driver_init"
log_event "START host=$(hostname) cwd=${PROJECT_ROOT} gpu=${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
ARMS=(${ARMS:-mlp_delta_last mlp_2ch tcn_delta_last tcn_2ch})
SEED=0
TOP_K=10
RELATION_TOP_N=1
STUDENT_TEMP=0.10
TEACHER_TEMP=0.07
TAU_TOPK=0.10
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
EMA_BASE=0.99
EMA_FINAL=0.9995
# Teacher is a controlled variable here, not an arm. future_mse and ema both
# work with multi-channel input; ema_target does not and is rejected upstream.
TEACHER_TYPE="${TEACHER_TYPE:-future_mse}"
TCN_KERNEL_SIZE="${TCN_KERNEL_SIZE:-3}"
TCN_CHANNELS="${TCN_CHANNELS:-0}"
FORCE="${FORCE:-0}"

if [[ "${RELATION_TOP_N}" -ne 1 ]]; then
  echo "Experiment 3 must remain self-only: RELATION_TOP_N=${RELATION_TOP_N}" >&2
  exit 2
fi

# Smallest dilation depth whose receptive field covers the whole relation
# window, so the TCN sees as much history as the MLP's flattened input does.
# With 2 convs per block the field is 1 + 2*(k-1)*(2^n - 1).
tcn_layers_for() {
  local relation_len="$1"
  local layers=1
  local field
  while (( layers < 12 )); do
    field=$(( 1 + 2 * (TCN_KERNEL_SIZE - 1) * ((1 << layers) - 1) ))
    (( field >= relation_len )) && break
    layers=$(( layers + 1 ))
  done
  printf '%s\n' "${layers}"
}

find_stage1_ckpt() {
  local directory="$1"
  local model_id="$2"
  ls -1 "${directory}"/stage1_"${model_id}"_*/checkpoint.pth 2>/dev/null \
    | head -1 || true
}

append_failure() {
  local message="$1"
  log_event "ARM_FAILURE ${message}"
  echo "[FAILED] ${message}" | tee -a "${LOG_DIR}/_failures.txt" >&2
}

archive_partial_log() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    local archived="${path%.log}.partial_$(date -u '+%Y%m%dT%H%M%SZ').log"
    mv "${path}" "${archived}"
    log_event "ARCHIVE partial_log=${path} archived=${archived}"
  fi
}

run_stage1_trained() {
  local is_training="$1"
  python -u run.py \
    --task_name stage1_relation \
    --is_training "${is_training}" \
    --model_id "${MODEL_ID}" \
    --model RelationStage1 \
    --learning_rate 1e-3 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --relation_teacher_type "${TEACHER_TYPE}" \
    --tau_student "${STUDENT_TEMP}" \
    --tau_teacher "${TEACHER_TEMP}" \
    --teacher_mse_space normalized \
    --stage1_loss_mode kl \
    --stage1_ema_momentum_base "${EMA_BASE}" \
    --stage1_ema_momentum_final "${EMA_FINAL}" \
    --stage1_probe_vis 0 \
    --stage1_collapse_metrics 1 \
    --des "${DESCRIPTION}" \
    "${COMMON_ARGS[@]}"
}

run_stage2() {
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model_id "exp3_tcn2ch_${ARM}_stage2_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_seed0" \
    --model RelationStage2 \
    --learning_rate 1e-2 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --stage2_retrieval_backbone stage1 \
    --stage1_encoder_init checkpoint \
    --stage1_ckpt_path "${STAGE1_CKPT}" \
    --stage2_retrieval_encoder online \
    --freeze_stage1_encoder 1 \
    --retrieval_soft_all 0 \
    --retrieval_kl_weight 0 \
    --tau_student "${STUDENT_TEMP}" \
    --tau_teacher "${TEACHER_TEMP}" \
    --refresh_memory_every_epoch 0 \
    --memory_cache_mode precompute \
    --memory_chunk_size 1024 \
    --base_head_mode shared_target_linear \
    --stage2_relation_fusion gate \
    --relation_mixer_input retrieved \
    --fusion_mode raft_concat \
    --tau_topk "${TAU_TOPK}" \
    --oracle_candidate_eval 1 \
    --des "exp3_tcn2ch_${ARM}_stage2_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_seed0" \
    "${COMMON_ARGS[@]}"
}

echo "=== Experiment 3: TCN encoder x 2-channel relation input, self-only, 2-stage Top-K ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  arms       : ${ARMS[*]}"
echo "  seq/pred   : ${PRED_LENS[*]} (seq_len == pred_len)"
echo "  teacher    : ${TEACHER_TYPE} (held fixed across arms)"
echo "  seed       : ${SEED}"
echo "  tau S/T/K  : ${STUDENT_TEMP} / ${TEACHER_TEMP} / ${TAU_TOPK}"
echo "  force      : ${FORCE}"
echo

for DATASET in "${DATASETS[@]}"; do
  case "${DATASET}" in
    ETTh1) DATA_PATH=ETTh1.csv ;;
    ETTm1) DATA_PATH=ETTm1.csv ;;
    *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
  esac
  SELF_GRAPH_PATH="./metrics/relation_graphs/${DATA_PATH%.csv}/pearson_self_top1.json"
  LOG_DIR="${RESULT_ROOT}/${DATASET}"
  mkdir -p "${LOG_DIR}"

  for ARM in "${ARMS[@]}"; do
    case "${ARM}" in
      mlp_delta_last) ENCODER_TYPE=mlp; INPUT_SPACE=delta_last ;;
      mlp_2ch)        ENCODER_TYPE=mlp; INPUT_SPACE=delta_last_diff1 ;;
      tcn_delta_last) ENCODER_TYPE=tcn; INPUT_SPACE=delta_last ;;
      tcn_2ch)        ENCODER_TYPE=tcn; INPUT_SPACE=delta_last_diff1 ;;
      *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
    esac
    if [[ "${ENCODER_TYPE}" == tcn ]]; then
      POOLING=last
    else
      POOLING=cls
    fi

    for PRED_LEN in "${PRED_LENS[@]}"; do
      SEQ_LEN="${PRED_LEN}"
      LOG_PATH="${LOG_DIR}/${ARM}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
      MODEL_ID="exp3_tcn2ch_${ARM}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_seed0"
      DESCRIPTION="${MODEL_ID}"
      STAGE1_DIR="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}"

      # delta_last_diff1 crops both features to the trailing L-1 steps.
      if [[ "${INPUT_SPACE}" == delta_last ]]; then
        RELATION_LEN="${SEQ_LEN}"
      else
        RELATION_LEN=$(( SEQ_LEN - 1 ))
      fi
      TCN_LAYERS="$(tcn_layers_for "${RELATION_LEN}")"

      COMMON_ARGS=(
        --data "${DATASET}"
        --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
        --data_path "${DATA_PATH}"
        --features M
        --seq_len "${SEQ_LEN}"
        --label_len 0
        --pred_len "${PRED_LEN}"
        --enc_in 7
        --dec_in 7
        --c_out 7
        --batch_size 32
        --num_workers 0
        --seed "${SEED}"
        --d_model 128
        --n_heads 4
        --e_layers 2
        --d_layers 1
        --d_ff 256
        --patch_len 16
        --stride 16
        --candidate_mask raft
        --relation_input_space "${INPUT_SPACE}"
        --relation_teacher_space delta_last
        --relation_value_space delta_last
        --source_mode auto
        --relation_top_n "${RELATION_TOP_N}"
        --relation_graph_path "${SELF_GRAPH_PATH}"
        --target_mode all
        --relation_encoder_type "${ENCODER_TYPE}"
        --relation_pooling "${POOLING}"
        --relation_self_fill linear
        --relation_tcn_layers "${TCN_LAYERS}"
        --relation_tcn_kernel_size "${TCN_KERNEL_SIZE}"
        --relation_tcn_channels "${TCN_CHANNELS}"
        --retrieval_similarity cosine
        --top_k "${TOP_K}"
      )

      if [[ "${FORCE}" != 1 ]] \
        && grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
        echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] already finished, skipping"
        log_event "SKIP completed=${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}"
        continue
      fi
      archive_partial_log "${LOG_PATH}"
      [[ "${FORCE}" == 1 ]] && : > "${LOG_PATH}"
      set_current "${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}"
      log_event "ARM_START current=${CURRENT_RUN} encoder=${ENCODER_TYPE} input=${INPUT_SPACE} tcn_layers=${TCN_LAYERS}"

      STAGE1_CKPT="$(find_stage1_ckpt "${STAGE1_DIR}" "${MODEL_ID}")"
      if [[ "${FORCE}" == 1 || -z "${STAGE1_CKPT}" ]]; then
        set_current "${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_train"
        log_event "PHASE_START current=${CURRENT_RUN} teacher=${TEACHER_TYPE}"
        echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] Stage-1 encoder=${ENCODER_TYPE} input=${INPUT_SPACE}"
        if ! run_stage1_trained 1 2>&1 | tee "${LOG_PATH}"; then
          append_failure "${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM} Stage-1"
          continue
        fi
        log_event "PHASE_DONE current=${CURRENT_RUN}"
        STAGE1_CKPT="$(find_stage1_ckpt "${STAGE1_DIR}" "${MODEL_ID}")"
        if [[ -z "${STAGE1_CKPT}" ]]; then
          append_failure "${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM}: no Stage-1 checkpoint"
          continue
        fi
      else
        echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] reusing ${STAGE1_CKPT}"
        if ! grep -q '^Stage1 Test |' "${LOG_PATH}" 2>/dev/null; then
          set_current "${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_test"
          log_event "PHASE_START current=${CURRENT_RUN} checkpoint=${STAGE1_CKPT}"
          if ! run_stage1_trained 0 2>&1 | tee -a "${LOG_PATH}"; then
            append_failure "${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM} Stage-1 evaluation"
            continue
          fi
          log_event "PHASE_DONE current=${CURRENT_RUN}"
        fi
      fi

      set_current "${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}/stage2_train_test"
      log_event "PHASE_START current=${CURRENT_RUN}"
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] Stage-2 frozen Top-K"
      if ! run_stage2 2>&1 | tee -a "${LOG_PATH}"; then
        append_failure "${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM} Stage-2"
        continue
      fi
      log_event "PHASE_DONE current=${CURRENT_RUN}"
      log_event "ARM_DONE current=${DATASET}/${ARM}/seq${SEQ_LEN}_pred${PRED_LEN}"
    done
  done

  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    RESULT_DIR="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}"
    mkdir -p "${RESULT_DIR}"
    if ! python scripts/summarize_experiment3_tcn_2ch.py \
      --log-dir "${LOG_DIR}" \
      --log-suffix "_seq${SEQ_LEN}_pred${PRED_LEN}" \
      --output-prefix "${RESULT_DIR}/experiment3_tcn_2ch_results"; then
      echo "[summary] ${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} is incomplete" >&2
    fi
  done
done

echo
echo "logs/results: ${RESULT_ROOT}/<dataset>/"
set_current "driver_complete"
