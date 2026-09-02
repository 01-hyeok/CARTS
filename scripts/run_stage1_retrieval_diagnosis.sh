#!/bin/bash
# STEP 2 -- raw past-NN vs learned encoder, across every dataset/horizon.
#
# Every retriever is scored on the same candidate pool with the same protocol
# (full bank, no Oracle injection), so the comparison isolates the ranking
# function itself. Appends one CSV row per (dataset, horizon, retriever, split).
#
#   SPLITS="test"            which splits to score (default: test val)
#   RETRIEVERS="raw_l2 ..."  default: learned raw_l2 raw_cos random
#   MAX_QUERIES=512
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
RETRIEVERS=(${RETRIEVERS:-learned raw_l2 raw_cos random})
SPLITS=(${SPLITS:-test val})
MAX_QUERIES="${MAX_QUERIES:-512}"
# The learned arm these are compared against.
REF_ARM="${REF_ARM:-full_bank_kl}"
OUT_CSV="${OUT_CSV:-./metrics/next_stage1_diagnosis/retrieval_diagnosis.csv}"

rm -f "${OUT_CSV}"
FAILED=0
for DATASET in "${DATASETS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    CK=$(ls "./checkpoints/stage1/${DATASET}/seq${PRED_LEN}_pred${PRED_LEN}"/*_${REF_ARM}_${DATASET}_${PRED_LEN}_*/checkpoint.pth 2>/dev/null | head -1)
    if [ -z "${CK}" ]; then
      echo "[skip] no ${REF_ARM} checkpoint for ${DATASET}/pred${PRED_LEN}"
      continue
    fi
    for RETRIEVER in "${RETRIEVERS[@]}"; do
      for SPLIT in "${SPLITS[@]}"; do
        echo "[${DATASET}/pred${PRED_LEN}] ${RETRIEVER} (${SPLIT})"
        python -u scripts/diagnose_stage1_retrieval.py \
          --checkpoint "${CK}" \
          --retriever "${RETRIEVER}" \
          --split "${SPLIT}" \
          --max_queries "${MAX_QUERIES}" \
          --csv "${OUT_CSV}" > /dev/null 2>&1 \
          || { echo "  [FAILED] ${DATASET}/${PRED_LEN}/${RETRIEVER}/${SPLIT}"; FAILED=$((FAILED+1)); }
      done
    done
  done
done

echo
echo "failed: ${FAILED}"
echo "csv: ${OUT_CSV}"
