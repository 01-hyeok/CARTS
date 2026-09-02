#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

source /data/pjh_workspace/ts-env/bin/activate
cd "${PROJECT_ROOT}"

echo "[sequential random] Starting ETTh1 experiments"
"${SCRIPT_DIR}/run_random_retrieval_backbone_seed_average.sh" ETTh1

echo "[sequential random] ETTh1 completed; starting ETTm1 experiments"
"${SCRIPT_DIR}/run_random_retrieval_backbone_seed_average.sh" ETTm1

echo "[sequential random] ETTh1 and ETTm1 experiments completed"
