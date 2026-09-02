#!/bin/bash
set -uo pipefail

# Does sharpening the Top-K weighting help?
#
# alpha = softmax(top-k score / tau_topk). Measured across every finished arm,
# the effective candidate count exp(H(alpha)) sits at 8.9-10.0 out of 10, so the
# model is averaging the retrieved set rather than weighting it. tau_topk=0.10 is
# large relative to how tightly the top-k cosine scores cluster (0.83-0.98).
#
# Two arms, picked for how good their within-top-k ranking is:
#   identity     recall@1 ~0.004  - ranking barely informative
#   2stage_mse   recall@1 ~0.010  - best ranking in the study
# If sharpening only helps the second, the ranking is what limits this, not the
# temperature. If it helps neither, averaging is already the right choice and the
# Top-K weighting is not the bottleneck.
#
# Runs are named with a tau tag, so the tau=0.10 results already on disk stay put.
#
# Usage
#   bash scripts/run_tau_topk_sweep.sh
#   TAUS="0.05 0.01 0.002" bash scripts/run_tau_topk_sweep.sh
#   ARMS="identity" PRED_LENS=96 bash scripts/run_tau_topk_sweep.sh
#   nohup bash scripts/run_tau_topk_sweep.sh > logs/tau_sweep_driver.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
DATASETS=(${DATASETS:-ETTh1})
TAUS=(${TAUS:-0.05 0.01})
# no_retrieval is left out: it never builds a Top-K, so tau_topk does nothing.
ARMS="${ARMS:-identity identity_l2 random random_l2 chronos chronos_l2 chronos_eos chronos_tsrag 2stage_ema 2stage_ema_l2 2stage_mse 2stage_mse_l2 e2e_ema e2e_ema_l2 e2e_mse e2e_mse_l2}"

echo "=== tau_topk sweep ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  arms       : ${ARMS}"
echo "  taus       : ${TAUS[*]}   (0.10 already on disk)"
echo "  pred_lens  : ${PRED_LENS:-96 192 336}"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo

for TAU in "${TAUS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "########## tau_topk=${TAU}  ${DS} ##########"
    TAU_TOPK="${TAU}" bash "${SCRIPT_DIR}/run_self_topk.sh" "${DS}" ${ARMS} \
      || echo "[driver] tau=${TAU} ${DS} exited non-zero" >&2
    echo
  done
done

echo "  finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo
bash "${SCRIPT_DIR}/summarize_tau_topk.sh" || true
