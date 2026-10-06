#!/usr/bin/env bash
# Candidate-pool-abstraction retriever runner. Switch full <-> coarse_topk
# purely via environment variables -- same trainer/cache-builder either way.
#
#   CANDIDATE_POOL_MODE=full \
#     bash scripts/run_retriever_pool01.sh <reference_ckpt> <cell> <pred_len> <seq_len> <num_query_views>
#
#   CANDIDATE_POOL_MODE=coarse_topk CANDIDATE_POOL_SIZE=100 \
#     bash scripts/run_retriever_pool01.sh <reference_ckpt> <cell> <pred_len> <seq_len> <num_query_views>
#
# coarse_topk mode expects scripts/precompute_candidate_pool01.py to have
# already been run for this cell into
# "${CANDIDATE_POOL_CACHE_ROOT}/<cell>/" -- this runner does NOT trigger
# that precompute itself (spec: never an implicit expensive precompute
# inside a training run).
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

REFERENCE_CKPT="$1"
CELL="$2"
PRED_LEN="$3"
SEQ_LEN="$4"
NUM_QUERY_VIEWS="$5"

CANDIDATE_POOL_MODE="${CANDIDATE_POOL_MODE:-full}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-100}"
CANDIDATE_POOL_METRIC="${CANDIDATE_POOL_METRIC:-delta_last_cosine}"
CANDIDATE_POOL_CACHE_ROOT="${CANDIDATE_POOL_CACHE_ROOT:-results/CANDIDATE-POOL01/pool_cache}"
OUT_DIR="${OUT_DIR:-results/CANDIDATE-POOL01}"
CHECKPOINTS="${CHECKPOINTS:-checkpoints/candidate_pool01}"

POOL_ARGS=(--candidate_pool_mode "${CANDIDATE_POOL_MODE}")
if [[ "${CANDIDATE_POOL_MODE}" == "coarse_topk" ]]; then
  POOL_ARGS+=(--candidate_pool_size "${CANDIDATE_POOL_SIZE}" --candidate_pool_metric "${CANDIDATE_POOL_METRIC}" \
             --candidate_pool_cache "${CANDIDATE_POOL_CACHE_ROOT}/${CELL}")
fi

echo "=== run_retriever_pool01: cell=${CELL} V${NUM_QUERY_VIEWS} pool_mode=${CANDIDATE_POOL_MODE} ==="
python -u scripts/train_retriever_pool01.py \
  --reference_ckpt "${REFERENCE_CKPT}" --num_query_views "${NUM_QUERY_VIEWS}" \
  --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --out_dir "${OUT_DIR}" --checkpoints "${CHECKPOINTS}" \
  "${POOL_ARGS[@]}"
