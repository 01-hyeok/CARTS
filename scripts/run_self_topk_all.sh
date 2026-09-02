#!/bin/bash
set -uo pipefail

# Driver for the self-only Top-K study: ETTh1 first, then ETTm1.
#
# Both datasets run the same fifteen arms at pred_len 96/192/336, so this is
# 90 runs. 720 is left out on purpose: chronos-t5-base truncates its input to
# context_length=512, so at seq_len 720 it would retrieve from the last 512
# steps while every other arm read all 720 - the arms would not be comparable.
# Everything is resumable: a run whose log already contains
# "Stage2 Test Final" is skipped, and a two-stage arm reuses its Stage-1
# checkpoint if one is already on disk. A failing arm is recorded in
# logs/<dataset>/self_topk/_failures.txt and the sweep keeps going.
#
# Usage
#   bash scripts/run_self_topk_all.sh                  # GPU 1, all eight arms
#   FORCE=1 bash scripts/run_self_topk_all.sh          # re-run everything
#   GPU=2 bash scripts/run_self_topk_all.sh            # pick the GPU
#   ARMS="identity chronos" bash scripts/run_self_topk_all.sh
#   PRED_LENS="96 192" bash scripts/run_self_topk_all.sh
#   DATASETS="ETTm1" bash scripts/run_self_topk_all.sh
#
# To run detached and keep it alive after logout:
#   nohup bash scripts/run_self_topk_all.sh > logs/self_topk_driver.log 2>&1 &
#   tail -f logs/self_topk_driver.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})

echo "=== self-only Top-K sweep ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  arms       : ${ARMS:-no_retrieval identity identity_l2 random random_l2 chronos chronos_l2 2stage_ema 2stage_ema_l2 2stage_mse 2stage_mse_l2 e2e_ema e2e_ema_l2 e2e_mse e2e_mse_l2}"
echo "  pred_lens  : ${PRED_LENS:-96 192 336}"
echo "  force      : ${FORCE:-0}   (1 = re-run finished configs instead of skipping)"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo

STATUS=0
for DS in "${DATASETS[@]}"; do
  echo "########## ${DS} ##########"
  if ! bash "${SCRIPT_DIR}/run_self_topk.sh" "${DS}"; then
    echo "[driver] ${DS} sweep exited non-zero" >&2
    STATUS=1
  fi
  echo
done

echo "  finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo
bash "${SCRIPT_DIR}/summarize_self_topk.sh" || true
exit "${STATUS}"
