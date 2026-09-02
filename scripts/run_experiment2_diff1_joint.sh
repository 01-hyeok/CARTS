#!/usr/bin/env bash
set -uo pipefail

# Experiment 2, joint (end-to-end) arm.
#
# The three arms in run_experiment2_diff1.sh are all 2-stage: Stage-2 loads a
# retrieval encoder (or none at all) and freezes it, so the memory bank is a
# preprocessing step and the forecasting loss never reaches the retriever.
# This script adds the missing arm, where the encoder trains inside Stage-2.
#
# The forecasting loss alone cannot do it -- Top-K is not differentiable, so
# gradients reach the values that were retrieved but never the ranking that
# chose them (see --retrieval_kl_weight in run.py). The KL term is what makes
# the ranking trainable, so the objective is
#
#     loss = forecasting MSE + LAMBDA * KL(future-MSE teacher || cosine student)
#
# Everything else is copied verbatim from run_experiment2_diff1.sh so that
# joint vs. 2-stage is the only thing that differs: diff1 relation input,
# self-only Top-1 relation graph, seq_len == pred_len, seed 0, Top-K = 10.
#
# Deliberately a separate file: run_experiment2_diff1.sh is long-running and
# bash re-reads a script from a byte offset while it executes, so editing that
# file in place while it runs can misdirect the remaining configs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/logs/experiment2_diff1_self_only_seed0}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"

DATASETS=(${DATASETS:-ETTh1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
SEED=0
TOP_K=10
RELATION_TOP_N=1
STUDENT_TEMP=0.10
TEACHER_TEMP=0.07
TAU_TOPK=0.10
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
LAMBDA="${LAMBDA:-1.0}"
# future_mse scores candidates against the ground-truth future, so a constant
# embedding is not a minimum of the KL. An ema teacher is an EMA of the student
# itself, which does admit that collapse. RESULTS_SUMMARY.md measures the gap:
# e2e_scratch lambda=1.0 is 0.3736 on ETTh1/96 with the ema teacher and 0.3610
# with future_mse.
KL_TEACHER="${KL_TEACHER:-future_mse}"
# 'random' is what the existing e2e sweep calls scratch: the Stage-2 encoder is
# randomly initialised and no Stage-1 checkpoint is read, so no Stage-1 runs.
ENCODER_INIT="${ENCODER_INIT:-random}"
ARM_TAG="${ARM_TAG:-joint}"
FORCE="${FORCE:-0}"

if [[ "${RELATION_TOP_N}" -ne 1 ]]; then
  echo "Experiment 2 must remain self-only: RELATION_TOP_N=${RELATION_TOP_N}" >&2
  exit 2
fi

append_failure() {
  local message="$1"
  echo "[FAILED] ${message}" | tee -a "${LOG_DIR}/_failures_joint.txt" >&2
}

echo "=== Experiment 2 joint arm: self-only Diff1, end-to-end + retrieval KL ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  seq/pred   : ${PRED_LENS[*]} (seq_len == pred_len)"
echo "  seed       : ${SEED}"
echo "  lambda     : ${LAMBDA}   (teacher=${KL_TEACHER})"
echo "  encoder    : ${ENCODER_INIT} init, trained inside Stage-2"
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

  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    LOG_PATH="${LOG_DIR}/${ARM_TAG}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
    DESCRIPTION="exp2_diff1_self_top1_${ARM_TAG}_stage2_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_seed0"

    # Identical to run_experiment2_diff1.sh COMMON_ARGS.
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
      --relation_input_space diff1
      --relation_teacher_space delta_last
      --relation_value_space delta_last
      --source_mode auto
      --relation_top_n "${RELATION_TOP_N}"
      --relation_graph_path "${SELF_GRAPH_PATH}"
      --target_mode all
      --relation_encoder_type mlp
      --relation_pooling cls
      --relation_self_fill linear
      --retrieval_similarity cosine
      --top_k "${TOP_K}"
    )

    if [[ "${FORCE}" != 1 ]] \
      && grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM_TAG}] already finished, skipping"
      continue
    fi

    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM_TAG}] Stage-2 end-to-end (lambda=${LAMBDA})"
    # Differences from the 2-stage arms, and only these:
    #   freeze_stage1_encoder 0   -> the encoder joins the optimizer
    #   retrieval_kl_weight       -> makes the ranking trainable
    #   refresh_memory_every_epoch 1 -> keys are re-encoded as the encoder moves
    #   stage1_encoder_init random   -> no Stage-1 checkpoint is read
    if ! python -u run.py \
      --task_name stage2_relation \
      --is_training 1 \
      --model_id "exp2_diff1_self_top1_${ARM_TAG}_stage2_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_seed0" \
      --model RelationStage2 \
      --learning_rate 1e-2 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --stage2_retrieval_backbone stage1 \
      --stage1_encoder_init "${ENCODER_INIT}" \
      --stage2_retrieval_encoder online \
      --freeze_stage1_encoder 0 \
      --retrieval_kl_weight "${LAMBDA}" \
      --retrieval_kl_teacher "${KL_TEACHER}" \
      --refresh_memory_every_epoch 1 \
      --retrieval_soft_all 0 \
      --tau_student "${STUDENT_TEMP}" \
      --tau_teacher "${TEACHER_TEMP}" \
      --memory_cache_mode precompute \
      --memory_chunk_size 1024 \
      --base_head_mode shared_target_linear \
      --stage2_relation_fusion gate \
      --relation_mixer_input retrieved \
      --fusion_mode raft_concat \
      --tau_topk "${TAU_TOPK}" \
      --oracle_candidate_eval 1 \
      --des "${DESCRIPTION}" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "${LOG_PATH}"; then
      append_failure "${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM_TAG} Stage-2"
      continue
    fi
  done
done

echo
echo "logs: ${RESULT_ROOT}/<dataset>/${ARM_TAG}_seq*_pred*.log"
