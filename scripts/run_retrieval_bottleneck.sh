#!/usr/bin/env bash
# Attribute the utility-aligned Stage-1 failure to pool, aggregation or a moving
# target -- in that order, so nothing is redesigned before the cause is known.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

OUT="${OUT:-./metrics/retrieval_bottleneck}"
LOGS="${OUT}/logs"; mkdir -p "${LOGS}"
MAX_QUERIES="${MAX_QUERIES:-512}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
ARMS=(${ARMS:-future_kl_pool residual_kl utility_kl expected_utility utility_kl_null expected_utility_null})
PRED=96

stage1_ckpt() {  # dataset arm
  ls -d checkpoints/stage1/$1/seq${PRED}_pred${PRED}/*fu1_$2_$1_${PRED}_*/checkpoint.pth 2>/dev/null | head -1
}
reference_ckpt() {
  ls -d checkpoints/stage2/$1/seq${PRED}_pred${PRED}/*s2_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1
}

for DATASET in "${DATASETS[@]}"; do
  REF="$(reference_ckpt "${DATASET}")"
  [ -z "${REF}" ] && { echo "[miss] reference Stage-2 for ${DATASET}"; continue; }
  METHODS=("future_kl_full=")
  for ARM in "${ARMS[@]}"; do
    CKPT="$(stage1_ckpt "${DATASET}" "${ARM}")"
    [ -z "${CKPT}" ] && { echo "[miss] Stage-1 ${DATASET}/${ARM}"; continue; }
    METHODS+=("${ARM}=${CKPT}")
  done
  MARKER="${LOGS}/exp1_${DATASET}.done"
  if [ -f "${MARKER}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] exp1 ${DATASET}"; continue
  fi
  echo "[exp1] ${DATASET}: ${#METHODS[@]} methods"
  python -u scripts/analyze_set_level_utility.py --reference "${REF}" \
    --method "${METHODS[@]}" --ks 1,3,5,10 --max_queries "${MAX_QUERIES}" \
    --out_dir "${OUT}" > "${LOGS}/exp1_${DATASET}.log" 2>&1 \
    && touch "${MARKER}" || echo "[FAILED] exp1 ${DATASET}"
  grep -E "^ETT" "${LOGS}/exp1_${DATASET}.log" | tail -30
done

# ---------- EXPERIMENT 3: does the utility target move when Stage-2 retrains? ----------
if [ "${EXP3:-1}" = "1" ]; then
for DATASET in "${DATASETS[@]}"; do
  BASE="$(reference_ckpt "${DATASET}")"
  [ -z "${BASE}" ] && continue
  MARKER="${LOGS}/exp3_${DATASET}.done"
  if [ -f "${MARKER}" ] && [ "${FORCE:-0}" != "1" ]; then echo "[skip] exp3 ${DATASET}"; continue; fi
  METHODS=()
  for ARM in "${ARMS[@]}"; do
    CKPT=$(ls -d checkpoints/stage2/${DATASET}/seq${PRED}_pred${PRED}/*fu2_${ARM}_${DATASET}_${PRED}_*/checkpoint.pth 2>/dev/null | head -1)
    [ -n "${CKPT}" ] && METHODS+=("${ARM}=${CKPT}")
  done
  # Pool-scaling Stage-2 checkpoints join in once they exist.
  for PS in $(ls -d checkpoints/stage2/${DATASET}/seq${PRED}_pred${PRED}/*ps2_*/checkpoint.pth 2>/dev/null); do
    NAME=$(echo "${PS}" | sed 's|.*_ps2_\([a-z0-9_]*\)_'"${DATASET}"'_.*|\1|')
    METHODS+=("pool_${NAME}=${PS}")
  done
  [ ${#METHODS[@]} -eq 0 ] && { echo "[miss] exp3 ${DATASET}: no Stage-2 checkpoints"; continue; }
  echo "[exp3] ${DATASET}: ${#METHODS[@]} checkpoints"
  python -u scripts/analyze_utility_policy_stability.py --baseline "${BASE}" \
    --method "${METHODS[@]}" --pool_size "${POOL_SIZE:-500}" --max_queries "${MAX_QUERIES}" \
    --out_dir "${OUT}" > "${LOGS}/exp3_${DATASET}.log" 2>&1 \
    && touch "${MARKER}" || echo "[FAILED] exp3 ${DATASET}"
  grep -E "^ETT" "${LOGS}/exp3_${DATASET}.log" | tail -20
done
fi
