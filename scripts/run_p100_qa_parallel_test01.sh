#!/usr/bin/env bash
# TRACK-P100-QA-TEST-EVALUATION01 -- GPU1 parallel TEST evaluation.
# Thin wrapper; the real scheduling logic (GPU-memory-aware concurrency,
# per-job failure isolation, progress tracking) is in the Python driver
# `run_p100_qa_parallel_test01.py` -- bash `wait -n` job control is
# fragile across bash versions, Python's subprocess polling is not.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

ROOT="results/TRACK-P100-QA-TEST-EVALUATION01"
mkdir -p "$ROOT/test_metrics" "$ROOT/logs"

python -u scripts/run_p100_qa_parallel_test01.py \
  --locked_config "$ROOT/LOCKED_CONFIG.json" \
  --out_dir "$ROOT/test_metrics" \
  --max_parallel "${MAX_PARALLEL:-2}" \
  --min_free_mib_to_launch "${MIN_FREE_MIB:-8000}" \
  --log_dir "$ROOT/logs" \
  --status_json "$ROOT/evaluation_status.json" \
  --gpu_usage_json "$ROOT/gpu_usage_summary.json"
