#!/usr/bin/env bash
# TRACK-V Shared-Top-100 re-run. One setting at a time:
#   bash scripts/run_v_sharedtop100_01.sh <cell> <ref_ckpt> <base_ckpt> <pred_len> <seq_len>
# Never touches the existing Full-memory TRACK-V results -- everything
# goes under a new pool_top100/ subtree.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; BASE="$3"; PRED_LEN="$4"; SEQ_LEN="$5"
POOL_SIZE="${POOL_SIZE:-100}"

# cell "ETTh1_96" -> dataset=ETTh1 horizon=96 ; "Weather_720" -> dataset=Weather horizon=720
DATASET="${CELL%_*}"
HORIZON="${CELL##*_}"

RESULTS_ROOT="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${DATASET}/H${HORIZON}/seed0/pool_top100"
CKPT_ROOT="checkpoints/track_v_multiquery_generalization01/${DATASET}/H${HORIZON}/seed0/pool_top100"
POOL_CACHE="${RESULTS_ROOT}/shared_candidate_pool"

echo "=== [${CELL}] STEP 1: precompute Shared Top-${POOL_SIZE} pool (once) ==="
if [[ -f "${POOL_CACHE}/val.pt" ]]; then
  echo "---- pool cache already exists at ${POOL_CACHE}, skipping precompute ----"
else
  python -u scripts/precompute_candidate_pool01.py \
    --reference_ckpt "${REF}" --dataset "${DATASET}" --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" \
    --candidate_pool_size "${POOL_SIZE}" --query_chunk_size 256 --out_dir "${POOL_CACHE}"
fi

echo "=== [${CELL}] STEP 2: R0 Raw Cosine cache + Stage-2 ==="
if [[ ! -f "${RESULTS_ROOT}/raw_cosine/stage2/metrics.json" ]]; then
  python -u scripts/build_r0_rawcosine_pool_cache01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
    --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
    --out_dir "${RESULTS_ROOT}/raw_cosine/cache"
  python -u scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
    --base_checkpoint "${BASE}" --cache_dir "${RESULTS_ROOT}/raw_cosine/cache" \
    --out_dir "${RESULTS_ROOT}/raw_cosine/stage2"
else
  echo "---- R0 already done, skipping ----"
fi

for S in 0 1 2 5; do
  ARM="V${S}"
  # train_v_sharedtop100_01.py itself appends <cell>/<arm> to --out_dir/--checkpoints
  ARM_CKPT="${CKPT_ROOT}/${CELL}/${ARM}/checkpoint_best_retmse.pth"
  ARM_RESULT_DIR="${RESULTS_ROOT}/${CELL}/${ARM}"
  echo "=== [${CELL}] STEP 3.${S}: ${ARM} Stage-1 (dual checkpoint, P100 support) ==="
  if [[ ! -f "${ARM_CKPT}" ]]; then
    python -u scripts/train_v_sharedtop100_01.py \
      --reference_ckpt "${REF}" --num_query_views "${S}" --cell "${CELL}" \
      --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --candidate_pool_size "${POOL_SIZE}" \
      --candidate_pool_cache "${POOL_CACHE}" \
      --out_dir "${RESULTS_ROOT}" --checkpoints "${CKPT_ROOT}"
  else
    echo "---- [${CELL}] ${ARM} Stage-1 already done, skipping ----"
  fi

  echo "=== [${CELL}] STEP 3.${S}b: ${ARM} cache (best-retMSE checkpoint ONLY) + Stage-2 ==="
  if [[ ! -f "${ARM_RESULT_DIR}/stage2/metrics.json" ]]; then
    python -u scripts/build_retrieval_cache_pool01.py \
      --reference_ckpt "${REF}" --num_query_views "${S}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
      --candidate_pool_mode coarse_topk --candidate_pool_size "${POOL_SIZE}" --candidate_pool_cache "${POOL_CACHE}" \
      --retriever_checkpoint "${ARM_CKPT}" \
      --out_dir "${ARM_RESULT_DIR}/cache"
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
      --base_checkpoint "${BASE}" --cache_dir "${ARM_RESULT_DIR}/cache" \
      --out_dir "${ARM_RESULT_DIR}/stage2"
  else
    echo "---- [${CELL}] ${ARM} Stage-2 already done, skipping ----"
  fi
done

echo "=== [${CELL}] Shared-Top-${POOL_SIZE} pipeline complete ==="
