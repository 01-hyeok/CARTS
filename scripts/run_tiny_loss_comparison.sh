#!/bin/bash
# Can the encoder memorise Oracle Top-K, and does that depend on the loss?
#
# The original memorization check reached Recall@10 = 1.0 with regret 0, but it
# ran one loss (topk_coverage) with candidate gradient on, and the KL run that
# exists alongside it differs in three things at once -- loss, candidate gradient,
# and problem size -- so it cannot be read as "KL fails to memorise".
#
# Here the loss is the only thing that changes. Everything else is held at the
# original condition: 16 queries, 256 candidates, one target channel, self-only,
# differentiable candidate keys, no VICReg / rank / InfoNCE term.
#
# And validation now uses held-out queries against the same 256 candidates. The
# original reported val = train, so its Recall@10 of 1.0 said only that sixteen
# queries were memorised; whether any of it transfers was never measured.
#
#   train R@10 high, val R@10 high  ->  the mapping is learnable, scale is the issue
#   train R@10 high, val R@10 low   ->  memorisation only; the target may not be a
#                                       predictable function of the past
#   train R@10 low                  ->  this loss cannot even fit 16 queries
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SEED="${SEED:-2024}"
QUERIES="${QUERIES:-16}"
CANDIDATES="${CANDIDATES:-256}"
STEPS="${STEPS:-300}"
EPOCHS="${EPOCHS:-10}"
INPUT_SPACE="${INPUT_SPACE:-absolute}"
LOSSES=(${LOSSES:-topk_coverage kl weighted_topk_ce kl_expected_mse})
OUT="${OUT:-./metrics/tiny_loss_comparison}"
LOG_DIR="${LOG_DIR:-./logs/tiny_loss_comparison}"
mkdir -p "${OUT}" "${LOG_DIR}"

for LOSS in "${LOSSES[@]}"; do
  TAG="tiny_${INPUT_SPACE}_${LOSS}_seed${SEED}"
  LOG="${LOG_DIR}/${TAG}.log"; MARKER="${LOG_DIR}/${TAG}.done"
  if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then echo "[skip] ${TAG}"; continue; fi
  echo "=============================================================="
  echo "[tiny] loss=${LOSS} input=${INPUT_SPACE} queries=${QUERIES} candidates=${CANDIDATES}"
  echo "=============================================================="
  {
    echo "### RUN ${TAG} $(date -Is)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "CARTS_${TAG}_ETTh1_96" --model RelationStage1 \
      --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len 96 --label_len 0 --pred_len 96 \
      --enc_in 7 --batch_size 16 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft \
      --relation_input_space "${INPUT_SPACE}" --relation_teacher_space absolute \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --retrieval_similarity cosine \
      --source_mode all --target_mode single --target_channel 6 \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience 99 \
      --top_k 10 --tau_student 0.1 --tau_teacher 0.1 \
      --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k 10 \
      --stage1_variance_weight 0.0 --stage1_covariance_weight 0.0 \
      --stage1_use_rank_loss 0 \
      --stage1_overfit_queries "${QUERIES}" \
      --stage1_overfit_candidates "${CANDIDATES}" \
      --stage1_overfit_steps "${STEPS}" \
      --stage1_overfit_oracle_per_query 20 \
      --stage1_overfit_key_refresh step \
      --stage1_overfit_self_only 1 \
      --stage1_overfit_differentiable_keys 1 \
      --stage1_overfit_holdout_val 1 \
      --stage1_overfit_log_every 25 \
      --stage1_overfit_summary_path "${OUT}/${TAG}.json" \
      --stage1_checkpoint_metric recall10 --stage1_probe_vis 0 \
      --des "${TAG}_ETTh1_seq96_pred96" || exit 21
    echo "### RUN COMPLETE ${TAG} $(date -Is)"
  } 2>&1 | tee "${LOG}"
  if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
    touch "${MARKER}"; echo "[ok] ${TAG}"
  else
    echo "[FAILED] ${TAG}"
  fi
done
echo "tiny loss comparison finished $(date -Is)"
