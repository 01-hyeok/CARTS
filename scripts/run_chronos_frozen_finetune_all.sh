#!/bin/bash
set -euo pipefail

# CARTS with the Chronos retrieval backbone: frozen vs fine-tuned encoder,
# ETTh1 then ETTm1, seq == pred over 96/192/336.
#
# 720 is excluded because Chronos truncates its input to 512 tokens, so a
# 720-long window would only be embedded from its last 512 points.
#
# tau_topk stays at 0.10 to match every other condition in the ablation table.
# Note that at this temperature the Top-K weights are close to uniform
# (alpha_entropy_norm >= 0.99 across all conditions), so frozen and fine-tuned
# may come out closer than the underlying retrieval quality difference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRED_LENS="${PRED_LENS:-96 192 336}"
export TAU_TOPK="${TAU_TOPK:-0.10}"
export DATASETS="${DATASETS:-ETTh1 ETTm1}"
export MODES="${MODES:-frozen finetune}"

exec "${SCRIPT_DIR}/run_etth1_then_ettm1_chronos_retrieval.sh"
