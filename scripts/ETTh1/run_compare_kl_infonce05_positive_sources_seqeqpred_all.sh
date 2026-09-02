#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/ETTh1"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo '[ETTh1] Experiment 1/2: KL 0.5 + target-MSE Top-K InfoNCE 0.5'
"${SCRIPT_DIR}/run_ema_mlp_repeat_kl_infonce05_target_mse_topk_concat_seqeqpred_all.sh" \
  2>&1 | tee "${LOG_DIR}/run_ema_mlp_linear_kl_infonce05_target_mse_topk_concat_seqeqpred_all.log"

echo '[ETTh1] Experiment 2/2: KL 0.5 + branch-wise EMA-cosine Top-K InfoNCE 0.5'
"${SCRIPT_DIR}/run_ema_mlp_repeat_kl_infonce05_branch_ema_cosine_topk_concat_seqeqpred_all.sh" \
  2>&1 | tee "${LOG_DIR}/run_ema_mlp_linear_kl_infonce05_branch_ema_cosine_topk_concat_seqeqpred_all.log"
