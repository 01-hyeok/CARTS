#!/bin/bash
# Run all four Stage-1 Oracle Top-K memorization conditions and print one
# side-by-side Train Recall@1/5/10 table.
#
#   1. absolute   + step-refresh key bank
#   2. delta_last + step-refresh key bank
#   3. absolute   + fully differentiable candidate encoding
#   4. delta_last + fully differentiable candidate encoding
#
# Everything else -- seed, query/candidate set, K, tau, encoder, Oracle space --
# is held fixed, so the four runs are directly comparable.
set -euo pipefail

cd "$(dirname "$0")/.."

export SEED="${SEED:-2024}"
export SUMMARY_DIR="${SUMMARY_DIR:-./metrics/stage1_topk_memorization}"
LOG_DIR="${LOG_DIR:-./logs/ETTh1/stage1_topk_memorization}"
mkdir -p "${SUMMARY_DIR}" "${LOG_DIR}"

for input_space in absolute delta_last; do
  for candidate_mode in key_bank differentiable; do
    tag="topk_memorization_${input_space}_${candidate_mode}_seed${SEED}"
    log_path="${LOG_DIR}/${tag}.log"
    echo ">>> ${tag}"
    INPUT_SPACE="${input_space}" CANDIDATE_MODE="${candidate_mode}" \
      bash scripts/ETTh1/run_stage1_topk_memorization_sanity.sh 2>&1 | tee "${log_path}"
  done
done

echo
python -u scripts/summarize_stage1_topk_memorization.py --summary_dir "${SUMMARY_DIR}"
