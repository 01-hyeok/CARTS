#!/bin/bash
# Candidate-side Gradient Recovery: Stage-1 retrieval -> Stage-2 forecasting.
#
# Three Stage-1 arms, identical in every other respect, each followed by a
# Stage-2 run on its own best checkpoint with the encoder frozen:
#
#   full_bank_kl            full memory bank, no candidate gradient   (baseline)
#   selected100_detached_kl Bank Top-M + Oracle injection, bank embeddings
#   selected100_reencode_kl same candidates, re-encoded with grad ON
#
# A vs B isolates the candidate-subset effect; B vs C isolates the
# candidate-gradient effect, which is the question this experiment exists for.
#
# Usage
#   SMOKE=1 bash scripts/run_candidate_reencode_kl_full.sh   # ETTh1 pred96, 1 epoch
#   bash scripts/run_candidate_reencode_kl_full.sh           # full 8-setting sweep
#
# Retrieval is self-only by default: relation_top_n=1 keeps just the self source,
# because the Pearson relation graph ranks self first and top_n counts the whole
# source list including self. That matches the protocol these arms are being
# compared under -- each channel retrieves from its own history, no
# cross-channel relation. Set RELATION_TOP_N=3 for the cross-channel variant.
#
# Env knobs: DATASETS, PRED_LENS, ARMS, MINE_TOP_M, STAGE1_EPOCHS, STAGE2_EPOCHS,
#            RELATION_TOP_N, FORCE=1 (ignore completion markers and rerun).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
if [ "${SMOKE}" = "1" ]; then
  DATASETS=(${DATASETS:-ETTh1})
  PRED_LENS=(${PRED_LENS:-96})
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
  STAGE1_PATIENCE=1
  STAGE2_PATIENCE=1
  RUN_TAG="smoke"
else
  DATASETS=(${DATASETS:-ETTh1 ETTm1})
  PRED_LENS=(${PRED_LENS:-96 192 336 720})
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
  STAGE1_PATIENCE="${STAGE1_PATIENCE:-5}"
  STAGE2_PATIENCE="${STAGE2_PATIENCE:-5}"
  RUN_TAG="full"
fi

ARMS=(${ARMS:-full_bank_kl selected100_detached_kl selected100_reencode_kl})
MINE_TOP_M="${MINE_TOP_M:-100}"
ORACLE_INJECT_K="${ORACLE_INJECT_K:-10}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-recall10}"
SEED="${SEED:-0}"
# Stage-1 re-encodes selected candidates per (target, source) branch. Splitting
# the targets across batches keeps that cost bounded; 0 trains all targets at
# once, matching the existing protocol.
TARGET_CHUNK="${TARGET_CHUNK:-0}"
# Self-only: 1 source per target, so 7 relation branches instead of 49.
RELATION_TOP_N="${RELATION_TOP_N:-1}"

LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/candidate_reencode_kl_${RUN_TAG}}"
SUMMARY_DIR="${SUMMARY_DIR:-${PROJECT_ROOT}/metrics/candidate_reencode_kl_${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${SUMMARY_DIR}"

arm_subset_mode() {
  case "$1" in
    full_bank_kl|full_bank_cov)            echo none ;;
    selected100_detached_kl|selected100_detached_cov) echo selected_detached ;;
    selected100_reencode_kl|selected100_reencode_cov) echo selected_reencode ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# _cov arms replace the KL distillation with the explicit Oracle Top-K coverage
# loss the tiny-overfit diagnostic used, so the full-scale run tests the same
# objective that showed the candidate-gradient effect.
arm_loss_mode() {
  case "$1" in
    *_cov) echo topk_coverage ;;
    *)     echo kl ;;
  esac
}

dataset_root() {
  case "$1" in
    ETTh1|ETTm1) echo "../Dataset/Time-Series-Library_dataset/ETT-small/" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

FAILED_RUNS=()
COMPLETED_RUNS=()
SKIPPED_RUNS=()

for DATASET in "${DATASETS[@]}"; do
  ROOT_PATH="$(dataset_root "${DATASET}")" || exit 1
  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED_LEN}"
    mkdir -p "${LOG_DIR}"

    for ARM in "${ARMS[@]}"; do
      SUBSET_MODE="$(arm_subset_mode "${ARM}")" || exit 1
      LOSS_MODE="$(arm_loss_mode "${ARM}")" || exit 1
      RUN_ID="${DATASET}/pred${PRED_LEN}/${ARM}"
      LOG_PATH="${LOG_DIR}/${ARM}.log"
      DONE_MARKER="${LOG_DIR}/${ARM}.done"

      if [ "${FORCE}" != "1" ] && [ -f "${DONE_MARKER}" ]; then
        echo "[skip] ${RUN_ID} already completed (${DONE_MARKER})"
        SKIPPED_RUNS+=("${RUN_ID}")
        continue
      fi

      # Short ids keep the generated setting under the path-length limit, so the
      # checkpoint path below stays predictable instead of being hashed.
      S1_ID="carts_s1_${ARM}_${DATASET}_${PRED_LEN}"
      S1_DES="s1_${ARM}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      S2_ID="carts_s2_${ARM}_${DATASET}_${PRED_LEN}"
      S2_DES="s2_${ARM}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      S1_SETTING="stage1_${S1_ID}_RelationStage1_${DATASET}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${S1_DES}_0"
      S1_CKPT="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}/${S1_SETTING}/checkpoint.pth"

      COMMON_ARGS=(
        --data "${DATASET}"
        --root_path "${ROOT_PATH}"
        --data_path "${DATASET}.csv"
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
        --seed "${SEED}"
        --candidate_mask raft
        --relation_input_space "${INPUT_SPACE:-delta_last}"
        --relation_teacher_space delta_last
        --relation_value_space delta_last
        --source_mode auto
        --relation_top_n "${RELATION_TOP_N}"
        --target_mode all
        --relation_encoder_type mlp
        --relation_self_fill linear
      )

      echo "=============================================================="
      echo "[${RUN_ID}] subset_mode=${SUBSET_MODE} loss=${LOSS_MODE} top_m=${MINE_TOP_M} gpu=${CUDA_VISIBLE_DEVICES}"
      echo "=============================================================="

      {
        echo "### RUN ${RUN_ID} arm=${ARM} subset_mode=${SUBSET_MODE} $(date -Is)"

        echo "### STAGE1"
        python -u run.py \
          --task_name stage1_relation \
          --is_training 1 \
          --model_id "${S1_ID}" \
          --model RelationStage1 \
          "${COMMON_ARGS[@]}" \
          --learning_rate 1e-3 \
          --train_epochs "${STAGE1_EPOCHS}" \
          --patience "${STAGE1_PATIENCE}" \
          --top_k 10 \
          --tau_student 0.10 \
          --tau_teacher 0.1 \
          --teacher_mse_space normalized \
          --stage1_teacher_mode mse \
          --stage1_loss_mode "${LOSS_MODE}" \
          --stage1_coverage_top_k "${ORACLE_INJECT_K}" \
          --stage1_candidate_subset_mode "${SUBSET_MODE}" \
          --stage1_candidate_mine_top_m "${MINE_TOP_M}" \
          --stage1_candidate_oracle_inject_k "${ORACLE_INJECT_K}" \
          --stage1_checkpoint_metric "${CHECKPOINT_METRIC}" \
          --relation_target_chunk_size "${TARGET_CHUNK}" \
          --stage1_probe_vis 0 \
          --des "${S1_DES}" || exit 21

        if [ ! -f "${S1_CKPT}" ]; then
          echo "### ERROR missing Stage-1 checkpoint: ${S1_CKPT}"
          exit 22
        fi

        echo "### STAGE2 stage1_ckpt=${S1_CKPT}"
        python -u run.py \
          --task_name stage2_relation \
          --is_training 1 \
          --model_id "${S2_ID}" \
          --model RelationStage2 \
          --base_head_mode shared_target_linear \
          "${COMMON_ARGS[@]}" \
          --learning_rate 1e-2 \
          --train_epochs "${STAGE2_EPOCHS}" \
          --patience "${STAGE2_PATIENCE}" \
          --stage1_ckpt_path "${S1_CKPT}" \
          --freeze_stage1_encoder 1 \
          --memory_cache_mode precompute \
          --refresh_memory_every_epoch 1 \
          --memory_chunk_size 1024 \
          --top_k 10 \
          --tau_topk 0.10 \
          --relation_mixer_input retrieved \
          --fusion_mode residual \
          --gate_mode scalar \
          --des "${S2_DES}" || exit 23

        echo "### RUN COMPLETE ${RUN_ID} $(date -Is)"
      } 2>&1 | tee "${LOG_PATH}"

      STATUS="${PIPESTATUS[0]}"
      if [ "${STATUS}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG_PATH}"; then
        touch "${DONE_MARKER}"
        COMPLETED_RUNS+=("${RUN_ID}")
        echo "[ok] ${RUN_ID}"
      else
        FAILED_RUNS+=("${RUN_ID} (exit ${STATUS})")
        echo "[FAILED] ${RUN_ID} exit=${STATUS} -- continuing with the next arm"
      fi
    done
  done
done

echo
echo "=============================================================="
echo "completed: ${#COMPLETED_RUNS[@]}  skipped: ${#SKIPPED_RUNS[@]}  failed: ${#FAILED_RUNS[@]}"
for RUN in "${FAILED_RUNS[@]:-}"; do
  [ -n "${RUN}" ] && echo "  FAILED ${RUN}"
done
echo "logs:    ${LOG_ROOT}"
echo "=============================================================="

python -u scripts/summarize_candidate_reencode_kl.py \
  --log_root "${LOG_ROOT}" \
  --out_dir "${SUMMARY_DIR}" || echo "[warn] summary generation failed"

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  exit 1
fi
