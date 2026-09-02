#!/bin/bash
set -uo pipefail

# Does Chronos do better under TS-RAG's own retrieval settings?
#
# The chronos arm in the main sweep uses this repo's conventions, which differ
# from TS-RAG on three axes. That leaves "you used the pretrained encoder wrong"
# as an open objection to the chronos result, and these two arms close it:
#
#   chronos_eos    EOS pooling, cosine, delta_last
#                  Only pooling moves. TS-RAG reads embeddings[:, -1, :], the EOS
#                  summary token; this repo drops EOS and averages value tokens.
#   chronos_tsrag  EOS pooling, l2, absolute input
#                  All three at once - TS-RAG as published (faiss IndexFlatL2 on
#                  un-normalised EOS embeddings of the raw window).
#
# Together with chronos (baseline) and chronos_l2 (distance only) from the main
# sweep, a loss can be attributed to pooling, to distance, or to neither.
#
# Runs on GPU 2 by default so it can go alongside the main sweep on GPU 1. Log
# and checkpoint names are keyed on the arm, so the two sweeps never collide.
#
# Usage
#   bash scripts/run_chronos_tsrag.sh                      # ETTh1 then ETTm1, 12 runs
#   DATASETS=ETTh1 bash scripts/run_chronos_tsrag.sh       # ETTh1 only, 6 runs
#   GPU=1 bash scripts/run_chronos_tsrag.sh
#   FORCE=1 bash scripts/run_chronos_tsrag.sh              # re-run finished configs
#
#   nohup bash scripts/run_chronos_tsrag.sh > logs/chronos_tsrag_driver.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-2}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
ARMS="${ARMS:-chronos_eos chronos_tsrag}"
export ARMS

echo "=== Chronos under TS-RAG retrieval settings ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  arms       : ${ARMS}"
echo "  pred_lens  : ${PRED_LENS:-96 192 336}"
echo "  force      : ${FORCE:-0}"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo

STATUS=0
for DS in "${DATASETS[@]}"; do
  echo "########## ${DS} ##########"
  if ! bash "${SCRIPT_DIR}/run_self_topk.sh" "${DS}" ${ARMS}; then
    echo "[driver] ${DS} sweep exited non-zero" >&2
    STATUS=1
  fi
  echo
done

echo "  finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo
bash "${SCRIPT_DIR}/summarize_self_topk.sh" || true
exit "${STATUS}"
