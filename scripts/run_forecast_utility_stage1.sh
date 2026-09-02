#!/bin/bash
# Retrieval supervision by measured forecast utility, end to end.
#
# The alignment study showed Future-MSE similarity ranks candidates almost
# independently of what actually helps Stage-2 (mean Spearman 0.178). This sweep
# replaces the Stage-1 teacher and asks the only question that settles it:
# does the canonical Stage-2 test MSE go down?
#
# Arms -- identical encoder, budget, seed and Stage-2 configuration throughout,
# and from A2 onward identical candidate ids as well:
#
#   future_kl_full        full memory bank, Future-MSE teacher      (incumbent)
#   future_kl_pool        same teacher on the fixed Top-100 pool    (subset cost)
#   residual_kl           residual-similarity teacher
#   utility_kl            measured Stage-2 utility teacher
#   expected_utility      maximise the gain the student would collect
#   utility_kl_null       utility teacher + learned abstention
#   expected_utility_null expected utility + learned abstention
#
#   future_kl_full vs future_kl_pool  = cost of restricting the pool
#   future_kl_pool vs the rest        = effect of the teacher itself
#
# Usage
#   STEPS="1 2" bash scripts/run_forecast_utility_stage1.sh   # diagnostics only
#   SMOKE=1 bash scripts/run_forecast_utility_stage1.sh       # 1 epoch, ETTh1/96
#   bash scripts/run_forecast_utility_stage1.sh               # first wave
#
# Env knobs: STEPS, DATASETS, PRED_LENS, ARMS, GPU, SMOKE, FORCE, TEACHER_TAU,
#            STAGE1_EPOCHS, STAGE2_EPOCHS, POOL_M.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
STEPS="${STEPS:-1 2 3 4 5}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
POOL_M="${POOL_M:-100}"
TEACHER_TAU="${TEACHER_TAU:-0.05}"
SEED="${SEED:-0}"

if [ "${SMOKE}" = "1" ]; then
  DATASETS=(${DATASETS:-ETTh1}); PRED_LENS=(${PRED_LENS:-96})
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"; STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
  STAGE1_PATIENCE=1; STAGE2_PATIENCE=1; RUN_TAG="smoke"
else
  DATASETS=(${DATASETS:-ETTh1 ETTm1}); PRED_LENS=(${PRED_LENS:-96})
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"; STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
  STAGE1_PATIENCE="${STAGE1_PATIENCE:-5}"; STAGE2_PATIENCE="${STAGE2_PATIENCE:-5}"
  RUN_TAG="full"
fi
ARMS=(${ARMS:-future_kl_pool residual_kl utility_kl expected_utility utility_kl_null expected_utility_null})

METRIC_DIR="${PROJECT_ROOT}/metrics/forecast_utility_stage1"
LOG_ROOT="${PROJECT_ROOT}/logs/forecast_utility_stage1_${RUN_TAG}"
CACHE_ROOT="${PROJECT_ROOT}/cache/utility_teacher"
mkdir -p "${METRIC_DIR}" "${LOG_ROOT}"

has_step() { [[ " ${STEPS} " == *" $1 "* ]]; }
dataset_root() { echo "../Dataset/Time-Series-Library_dataset/ETT-small/"; }
reference_ckpt() {
  ls -d checkpoints/stage2/$1/seq$2_pred$2/*s2_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1
}
cache_dir() { echo "${CACHE_ROOT}/$1_pred$2_m${POOL_M}"; }

# arm -> teacher target, objective, null mode
arm_target()    { case "$1" in future_kl_pool) echo future ;; residual_kl) echo residual ;; *) echo utility ;; esac; }
arm_objective() { case "$1" in expected_utility|expected_utility_null) echo expected_utility ;; *) echo kl ;; esac; }
arm_null()      { case "$1" in *_null) echo query ;; *) echo off ;; esac; }

# ---------- STEP 0: shared candidate pool and its measured teachers ----------
for DATASET in "${DATASETS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    CACHE="$(cache_dir "${DATASET}" "${PRED_LEN}")"
    if [ ! -f "${CACHE}/test.pt" ] || [ "${FORCE}" = "1" ]; then
      REF="$(reference_ckpt "${DATASET}" "${PRED_LEN}")"
      [ -z "${REF}" ] && { echo "[miss] no reference Stage-2 for ${DATASET}/${PRED_LEN}"; continue; }
      echo "[step0] precomputing teachers for ${DATASET}/${PRED_LEN}"
      python -u scripts/precompute_utility_teacher.py --checkpoint "${REF}" \
        --pool_m "${POOL_M}" --splits train,val,test \
        > "${LOG_ROOT}/precompute_${DATASET}_${PRED_LEN}.log" 2>&1 \
        || echo "[FAILED] step0 ${DATASET}/${PRED_LEN}"
    fi
  done
done

# ---------- STEP 1: what is the current encoder aligned with? ----------
if has_step 1; then
  CSV="${METRIC_DIR}/current_encoder_alignment.csv"
  [ "${FORCE}" = "1" ] && rm -f "${CSV}"
  CACHES=()
  for DATASET in "${DATASETS[@]}"; do for PRED_LEN in "${PRED_LENS[@]}"; do
    for SPLIT in train val test; do
      F="$(cache_dir "${DATASET}" "${PRED_LEN}")/${SPLIT}.pt"; [ -f "${F}" ] && CACHES+=("${F}")
    done
  done; done
  if [ ! -f "${CSV}" ] && [ ${#CACHES[@]} -gt 0 ]; then
    echo "[step1] encoder alignment over ${#CACHES[@]} caches"
    python -u scripts/analyze_current_encoder_alignment.py --cache "${CACHES[@]}" --csv "${CSV}" \
      > "${LOG_ROOT}/step1_alignment.log" 2>&1 || echo "[FAILED] step1"
  else
    echo "[skip] step1"
  fi
fi

# ---------- STEP 2: what do the teachers themselves look like? ----------
if has_step 2; then
  CSV="${METRIC_DIR}/teacher_distribution.csv"
  [ "${FORCE}" = "1" ] && rm -f "${CSV}"
  CACHES=()
  for DATASET in "${DATASETS[@]}"; do for PRED_LEN in "${PRED_LENS[@]}"; do
    F="$(cache_dir "${DATASET}" "${PRED_LEN}")/test.pt"; [ -f "${F}" ] && CACHES+=("${F}")
  done; done
  if [ ! -f "${CSV}" ] && [ ${#CACHES[@]} -gt 0 ]; then
    echo "[step2] teacher distributions over ${#CACHES[@]} caches"
    python -u scripts/analyze_teacher_distribution.py --cache "${CACHES[@]}" --csv "${CSV}" \
      > "${LOG_ROOT}/step2_teachers.log" 2>&1 || echo "[FAILED] step2"
  else
    echo "[skip] step2"
  fi
fi

# ---------- STEP 3/4: teacher ablation training, then real Stage-2 ----------
FAILED_RUNS=(); COMPLETED_RUNS=(); SKIPPED_RUNS=()
if has_step 3; then
for DATASET in "${DATASETS[@]}"; do
  ROOT_PATH="$(dataset_root "${DATASET}")"
  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    CACHE="$(cache_dir "${DATASET}" "${PRED_LEN}")"
    [ -f "${CACHE}/test.pt" ] || { echo "[miss] teacher cache ${CACHE}"; continue; }
    LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED_LEN}"; mkdir -p "${LOG_DIR}"

    for ARM in "${ARMS[@]}"; do
      RUN_ID="${DATASET}/pred${PRED_LEN}/${ARM}"
      LOG_PATH="${LOG_DIR}/${ARM}.log"; DONE_MARKER="${LOG_DIR}/${ARM}.done"
      if [ "${FORCE}" != "1" ] && [ -f "${DONE_MARKER}" ]; then
        echo "[skip] ${RUN_ID} already completed"; SKIPPED_RUNS+=("${RUN_ID}"); continue
      fi

      TARGET="$(arm_target "${ARM}")"; OBJECTIVE="$(arm_objective "${ARM}")"; NULL_MODE="$(arm_null "${ARM}")"
      S1_ID="carts_fu1_${ARM}_${DATASET}_${PRED_LEN}"
      S1_DES="fu1_${ARM}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      S2_ID="carts_fu2_${ARM}_${DATASET}_${PRED_LEN}"
      S2_DES="fu2_${ARM}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      S1_SETTING="stage1_${S1_ID}_RelationStage1_${DATASET}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${S1_DES}_0"
      S1_CKPT="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}/${S1_SETTING}/checkpoint.pth"

      COMMON_ARGS=(
        --data "${DATASET}" --root_path "${ROOT_PATH}" --data_path "${DATASET}.csv"
        --features M --seq_len "${SEQ_LEN}" --label_len 0 --pred_len "${PRED_LEN}"
        --enc_in 7 --batch_size 32 --num_workers 0
        --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256
        --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft
        --relation_input_space delta_last --relation_teacher_space delta_last
        --relation_value_space delta_last
        --source_mode auto --relation_top_n 1 --target_mode all
        --relation_encoder_type mlp --relation_self_fill linear
      )

      echo "=============================================================="
      echo "[${RUN_ID}] teacher=${TARGET} objective=${OBJECTIVE} null=${NULL_MODE} gpu=${CUDA_VISIBLE_DEVICES}"
      echo "=============================================================="
      {
        echo "### RUN ${RUN_ID} teacher=${TARGET} objective=${OBJECTIVE} null=${NULL_MODE} $(date -Is)"
        echo "### STAGE1"
        python -u run.py \
          --task_name stage1_relation --is_training 1 \
          --model_id "${S1_ID}" --model RelationStage1 "${COMMON_ARGS[@]}" \
          --learning_rate 1e-3 --train_epochs "${STAGE1_EPOCHS}" --patience "${STAGE1_PATIENCE}" \
          --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
          --teacher_mse_space normalized --stage1_teacher_mode mse --stage1_loss_mode kl \
          --stage1_teacher_cache "${CACHE}" \
          --stage1_teacher_target "${TARGET}" \
          --stage1_teacher_loss "${OBJECTIVE}" \
          --stage1_teacher_tau "${TEACHER_TAU}" \
          --stage1_null_mode "${NULL_MODE}" \
          --stage1_checkpoint_metric utility_gap_recovery \
          --stage1_probe_vis 0 --des "${S1_DES}" || exit 21

        [ -f "${S1_CKPT}" ] || { echo "### ERROR missing Stage-1 checkpoint: ${S1_CKPT}"; exit 22; }

        echo "### STAGE2 stage1_ckpt=${S1_CKPT}"
        python -u run.py \
          --task_name stage2_relation --is_training 1 \
          --model_id "${S2_ID}" --model RelationStage2 \
          --base_head_mode shared_target_linear "${COMMON_ARGS[@]}" \
          --learning_rate 1e-2 --train_epochs "${STAGE2_EPOCHS}" --patience "${STAGE2_PATIENCE}" \
          --stage1_ckpt_path "${S1_CKPT}" --freeze_stage1_encoder 1 \
          --memory_cache_mode precompute --refresh_memory_every_epoch 1 \
          --memory_chunk_size 1024 --top_k 10 --tau_topk 0.10 \
          --relation_mixer_input retrieved --fusion_mode residual --gate_mode scalar \
          --des "${S2_DES}" || exit 23
        echo "### RUN COMPLETE ${RUN_ID} $(date -Is)"
      } 2>&1 | tee "${LOG_PATH}"

      STATUS="${PIPESTATUS[0]}"
      if [ "${STATUS}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG_PATH}"; then
        touch "${DONE_MARKER}"; COMPLETED_RUNS+=("${RUN_ID}"); echo "[ok] ${RUN_ID}"
      else
        FAILED_RUNS+=("${RUN_ID} (exit ${STATUS})"); echo "[FAILED] ${RUN_ID} exit=${STATUS}"
      fi
    done
  done
done
fi

# ---------- STEP 5: collect Stage-1 and Stage-2 numbers into one table ----------
if has_step 5; then
  echo "[step5] collecting results"
  python -u scripts/collect_forecast_utility_results.py \
    --log_root "${LOG_ROOT}" --baseline_log_root "${PROJECT_ROOT}/logs/candidate_reencode_kl_full" \
    --metric_dir "${METRIC_DIR}" || echo "[FAILED] step5"
  python -u scripts/build_forecast_utility_report.py --root "${METRIC_DIR}" || echo "[FAILED] report"
fi

echo "=============================================================="
echo "completed=${#COMPLETED_RUNS[@]} skipped=${#SKIPPED_RUNS[@]} failed=${#FAILED_RUNS[@]}"
for run in "${FAILED_RUNS[@]:-}"; do [ -n "${run}" ] && echo "  FAILED ${run}"; done
