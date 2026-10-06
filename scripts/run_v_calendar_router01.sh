#!/usr/bin/env bash
# TRACK-V-CALENDAR-ROUTER01, one cell at a time:
#   bash scripts/run_v_calendar_router01.sh <cell> <ref_ckpt> <root_path> <data_path> <pred_len> <seq_len> <base_ckpt>
# Reuses the Shared-Top-100 V5 best-retMSE checkpoint produced by
# TRACK-V-MULTIQUERY-GENERALIZATION01's own pool_top100/ pipeline
# (never retrains V5 redundantly) -- if that pipeline hasn't reached
# this cell's V5 yet, this script WAITS (polls) rather than racing it
# with a second, independent V5 training run on the same files.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; ROOT_PATH="$3"; DATA_PATH="$4"; PRED_LEN="$5"; SEQ_LEN="$6"
POOL_SIZE="${POOL_SIZE:-100}"
DATASET="${CELL%_*}"
HORIZON="${CELL##*_}"

V5_RESULTS="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${DATASET}/H${HORIZON}/seed0/pool_top100"
V5_CKPT_ROOT="checkpoints/track_v_multiquery_generalization01/${DATASET}/H${HORIZON}/seed0/pool_top100"
POOL_CACHE="${V5_RESULTS}/shared_candidate_pool"
V5_CKPT="${V5_CKPT_ROOT}/${CELL}/V5/checkpoint_best_retmse.pth"
V5_DONE_SIGNAL="${V5_RESULTS}/${CELL}/V5/final_test_metrics_best_retmse.json"
V5_RR_METRICS="${V5_RESULTS}/${CELL}/V5/final_test_metrics_best_retmse.json"

OUT_ROOT="results/TRACK-V-CALENDAR-ROUTER01/${DATASET}/H${HORIZON}/seed0"
CKPT_ROOT="checkpoints/track_v_calendar_router01/${DATASET}/H${HORIZON}/seed0"

echo "=== [${CELL}] waiting for Shared-Top-100 V5 (reused, not retrained) ==="
WAITED=0
while [[ ! -f "${V5_DONE_SIGNAL}" ]]; do
  if [[ $WAITED -eq 0 ]]; then
    echo "---- [${CELL}] V5 not ready yet at ${V5_DONE_SIGNAL} -- polling every 120s ----"
  fi
  sleep 120
  WAITED=1
done
echo "---- [${CELL}] V5 ready: ${V5_CKPT} ----"

mkdir -p "${OUT_ROOT}/head_oracle" "${OUT_ROOT}/calendar_router" "${OUT_ROOT}/calendar_router_shuffled"

echo "=== [${CELL}] STEP C: Head Oracle cache ==="
if [[ ! -f "${OUT_ROOT}/head_oracle/head_oracle_distribution.json" ]]; then
  python -u scripts/build_head_oracle_cache01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
    --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
    --v5_checkpoint "${V5_CKPT}" --out_dir "${OUT_ROOT}/head_oracle"
else
  echo "---- [${CELL}] Head Oracle already done, skipping ----"
fi

echo "=== [${CELL}] STEP E/F: Calendar Router training (real calendar) ==="
if [[ ! -f "${CKPT_ROOT}/calendar_router/router_checkpoint_best_retmse.pth" ]]; then
  python -u scripts/train_calendar_router01.py \
    --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
    --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
    --v5_checkpoint "${V5_CKPT}" --head_oracle_dir "${OUT_ROOT}/head_oracle" \
    --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" \
    --out_dir "${OUT_ROOT}/calendar_router" --checkpoints "${CKPT_ROOT}/calendar_router"
else
  echo "---- [${CELL}] Calendar Router already trained, skipping ----"
fi

echo "=== [${CELL}] STEP 11: CRH-Shuffled control ==="
if [[ ! -f "${CKPT_ROOT}/calendar_router_shuffled/router_checkpoint_best_retmse.pth" ]]; then
  python -u scripts/train_calendar_router01.py \
    --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
    --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
    --v5_checkpoint "${V5_CKPT}" --head_oracle_dir "${OUT_ROOT}/head_oracle" \
    --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" --shuffle_calendar --shuffle_seed 0 \
    --out_dir "${OUT_ROOT}/calendar_router_shuffled" --checkpoints "${CKPT_ROOT}/calendar_router_shuffled"
else
  echo "---- [${CELL}] CRH-Shuffled already trained, skipping ----"
fi

echo "=== [${CELL}] STEP 12: CRH retrieval cache (best-retMSE V5 + best-retMSE Router) + Stage-2 ==="
if [[ ! -f "${OUT_ROOT}/stage2/metrics.json" ]]; then
  python -u scripts/build_crh_retrieval_cache01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
    --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
    --v5_checkpoint "${V5_CKPT}" --router_checkpoint "${CKPT_ROOT}/calendar_router/router_checkpoint_best_retmse.pth" \
    --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" --out_dir "${OUT_ROOT}/cache"
  python -u scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
    --base_checkpoint "$7" --cache_dir "${OUT_ROOT}/cache" --out_dir "${OUT_ROOT}/stage2"
else
  echo "---- [${CELL}] CRH Stage-2 already done, skipping ----"
fi

echo "=== [${CELL}] STEP: router head selection dump (for regret/accuracy diagnostics) ==="
python -u scripts/compute_router_head_selection01.py \
  --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --router_checkpoint "${CKPT_ROOT}/calendar_router/router_checkpoint_best_retmse.pth" \
  --head_oracle_test "${OUT_ROOT}/head_oracle/head_oracle_test.pt" \
  --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" \
  --out_path "${OUT_ROOT}/head_oracle/router_head_selection_test.pt"

echo "=== [${CELL}] STEP: diagnostics (specialization / regret / baselines) ==="
python -u scripts/compute_crh_diagnostics01.py \
  --head_oracle_dir "${OUT_ROOT}/head_oracle" \
  --v5_roundrobin_metrics "${V5_RR_METRICS}" \
  --crh_router_test_metrics "${OUT_ROOT}/calendar_router/final_test_metrics_best_retmse.json" \
  --shuffled_router_test_metrics "${OUT_ROOT}/calendar_router_shuffled/final_test_metrics_best_retmse.json" \
  --root_path "${ROOT_PATH}" --data_path "${DATA_PATH}" --seq_len "${SEQ_LEN}" \
  --out_dir "${OUT_ROOT}/diagnostics"

echo "=== [${CELL}] TRACK-V-CALENDAR-ROUTER01 pipeline complete ==="
