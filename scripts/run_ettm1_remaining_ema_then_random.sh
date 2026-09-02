#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

source /data/pjh_workspace/ts-env/bin/activate
cd "${PROJECT_ROOT}"

echo "[ETTm1 remaining] Starting EMA temperature experiments"
"${SCRIPT_DIR}/ETTm1/run_ema_mlp_repeat_seqeqpred_all.sh"

echo "[ETTm1 remaining] EMA experiments completed; starting random encoder experiments"
"${SCRIPT_DIR}/run_random_retrieval_backbone_seed_average.sh" ETTm1

echo "[ETTm1 remaining] EMA and random encoder experiments completed"
