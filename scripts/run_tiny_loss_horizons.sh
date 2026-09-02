#!/bin/bash
# The tiny memorisation check across all four horizons, with the loss as the only
# variable and validation on held-out queries.
#
# The original check ran ETTh1/96 alone, and every result since has split by
# horizon -- capacity rises monotonically at 96 and 192 but not at 336 and 720,
# and 96 is the one horizon where the asymmetric score loses. A 96-only answer
# cannot be placed beside those.
#
# Everything except the loss and the horizon is held at the original condition:
# 16 queries, 256 candidates, one target channel, self-only, differentiable
# candidate keys, no VICReg / rank / InfoNCE term. seq_len tracks pred_len, as in
# every other sweep here.
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
PRED_LENS=(${PRED_LENS:-96 192 336 720})
LOSSES=(${LOSSES:-topk_coverage kl weighted_topk_ce kl_expected_mse})
OUT="${OUT:-./metrics/tiny_loss_comparison}"
LOG_DIR="${LOG_DIR:-./logs/tiny_loss_comparison}"
mkdir -p "${OUT}" "${LOG_DIR}"

for PRED in "${PRED_LENS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    TAG="tiny_${INPUT_SPACE}_${LOSS}_pred${PRED}_seed${SEED}"
    LOG="${LOG_DIR}/${TAG}.log"; MARKER="${LOG_DIR}/${TAG}.done"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then echo "[skip] ${TAG}"; continue; fi
    echo "=============================================================="
    echo "[tiny] pred=${PRED} loss=${LOSS} queries=${QUERIES} candidates=${CANDIDATES}"
    echo "=============================================================="
    {
      echo "### RUN ${TAG} $(date -Is)"
      python -u run.py --task_name stage1_relation --is_training 1 \
        --model_id "CARTS_${TAG}_ETTh1_${PRED}" --model RelationStage1 \
        --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
        --data_path ETTh1.csv --features M \
        --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
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
        --des "${TAG}_ETTh1_seq${PRED}_pl${PRED}" || exit 21
      echo "### RUN COMPLETE ${TAG} $(date -Is)"
    } 2>&1 | tee "${LOG}"
    if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
      touch "${MARKER}"; echo "[ok] ${TAG}"
    else
      echo "[FAILED] ${TAG}"
    fi
  done
done
echo "tiny loss horizons finished $(date -Is)"
