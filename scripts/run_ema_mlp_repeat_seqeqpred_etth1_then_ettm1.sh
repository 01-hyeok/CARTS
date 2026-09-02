#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

source /data/pjh_workspace/ts-env/bin/activate
cd "${PROJECT_ROOT}"

echo "[sequential] Starting ETTh1 experiments"
"${SCRIPT_DIR}/ETTh1/run_ema_mlp_repeat_seqeqpred_all.sh"

echo "[sequential] ETTh1 completed; starting ETTm1 experiments"
"${SCRIPT_DIR}/ETTm1/run_ema_mlp_repeat_seqeqpred_all.sh"

echo "[sequential] ETTh1 and ETTm1 experiments completed"
