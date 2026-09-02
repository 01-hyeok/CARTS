#!/bin/bash
# Stage-1 Oracle Top-K memorization sanity check (single condition).
#
# Question: with only the past window as encoder input, can the current
# encoder + cosine geometry memorize the Future-MSE Oracle Top-K on a tiny
# fixed training set? Generalization is explicitly not measured here.
#
# The encoder never sees a future. Futures are used only to build the Oracle
# Top-K labels. The objective is topk_coverage alone -- no KL, no EMA teacher,
# no ranking/InfoNCE/expected-MSE term, no VICReg, no Stage-2, no cross-channel
# relation (self-only, single target channel).
#
# Condition knobs (env vars):
#   INPUT_SPACE   absolute | delta_last          (encoder input space)
#   CANDIDATE_MODE key_bank | differentiable     (Experiment 1 vs Experiment 2)
#
# Example:
#   INPUT_SPACE=absolute CANDIDATE_MODE=differentiable \
#     bash scripts/ETTh1/run_stage1_topk_memorization_sanity.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INPUT_SPACE="${INPUT_SPACE:-absolute}"
CANDIDATE_MODE="${CANDIDATE_MODE:-key_bank}"

case "${INPUT_SPACE}" in
  absolute|delta_last) ;;
  *) echo "INPUT_SPACE must be absolute or delta_last, got ${INPUT_SPACE}" >&2; exit 2 ;;
esac

case "${CANDIDATE_MODE}" in
  key_bank)
    DIFFERENTIABLE_KEYS=0
    KEY_REFRESH=step
    ;;
  differentiable)
    DIFFERENTIABLE_KEYS=1
    # Ignored while differentiable keys are on; kept for a readable setting name.
    KEY_REFRESH=step
    ;;
  *) echo "CANDIDATE_MODE must be key_bank or differentiable, got ${CANDIDATE_MODE}" >&2; exit 2 ;;
esac

# The Oracle is d(q,k) = MSE(y_q_future, y_k_future). Set TEACHER_SPACE=delta_last
# to score futures relative to each window's last past value instead, which is
# what CARTS' production Stage-1 uses. Hold it fixed across the four conditions
# so absolute and delta_last are compared against the same Oracle.
TEACHER_SPACE="${TEACHER_SPACE:-absolute}"

SEED="${SEED:-2024}"
QUERIES="${QUERIES:-16}"
CANDIDATES="${CANDIDATES:-256}"
COVERAGE_TOP_K="${COVERAGE_TOP_K:-10}"
STEPS="${STEPS:-300}"
EPOCHS="${EPOCHS:-10}"
LOG_EVERY="${LOG_EVERY:-25}"
TAU_STUDENT="${TAU_STUDENT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

TAG="topk_memorization_${INPUT_SPACE}_${CANDIDATE_MODE}_seed${SEED}"
SUMMARY_DIR="${SUMMARY_DIR:-./metrics/stage1_topk_memorization}"
SUMMARY_PATH="${SUMMARY_PATH:-${SUMMARY_DIR}/${TAG}.json}"
mkdir -p "${SUMMARY_DIR}"

echo "[sanity] input_space=${INPUT_SPACE} candidate_mode=${CANDIDATE_MODE} teacher_space=${TEACHER_SPACE} seed=${SEED}"

python -u run.py \
  --task_name stage1_relation \
  --is_training 1 \
  --model_id "CARTS_stage1_${TAG}_ETTh1_96" \
  --model RelationStage1 \
  --data ETTh1 \
  --root_path "${ROOT_PATH:-../Dataset/Time-Series-Library_dataset/ETT-small/}" \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.0 \
  --patch_len 16 \
  --stride 16 \
  --seed "${SEED}" \
  --learning_rate "${LEARNING_RATE}" \
  --lradj cosine \
  --train_epochs "${EPOCHS}" \
  --patience "${EPOCHS}" \
  --top_k "${COVERAGE_TOP_K}" \
  --candidate_mask raft \
  --relation_input_space "${INPUT_SPACE}" \
  --relation_teacher_space "${TEACHER_SPACE}" \
  --teacher_mse_space normalized \
  --retrieval_similarity cosine \
  --source_mode all \
  --target_mode single \
  --target_channel 6 \
  --relation_encoder_type mlp \
  --relation_self_fill linear \
  --tau_student "${TAU_STUDENT}" \
  --tau_teacher 0.1 \
  --stage1_teacher_mode mse \
  --stage1_loss_mode topk_coverage \
  --stage1_coverage_top_k "${COVERAGE_TOP_K}" \
  --stage1_variance_weight 0.0 \
  --stage1_covariance_weight 0.0 \
  --stage1_use_rank_loss 0 \
  --stage1_overfit_queries "${QUERIES}" \
  --stage1_overfit_candidates "${CANDIDATES}" \
  --stage1_overfit_steps "${STEPS}" \
  --stage1_overfit_oracle_per_query "${ORACLE_PER_QUERY:-20}" \
  --stage1_overfit_key_refresh "${KEY_REFRESH}" \
  --stage1_overfit_self_only 1 \
  --stage1_overfit_differentiable_keys "${DIFFERENTIABLE_KEYS}" \
  --stage1_overfit_log_every "${LOG_EVERY}" \
  --stage1_overfit_summary_path "${SUMMARY_PATH}" \
  --stage1_probe_vis 0 \
  --des "stage1_${TAG}_ETTh1_seq96_pred96"
