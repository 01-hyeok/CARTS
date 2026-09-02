#!/usr/bin/env bash
# Forecast-utility alignment sweep: ETTh1/ETTm1 x {96,192,336,720}, sequential on GPU 1.
# Resumable -- a setting already present in the CSV is skipped, so an interrupted
# sweep continues instead of recomputing or duplicating rows.
set -u

ROOT=./metrics/forecast_utility_alignment
CSV=$ROOT/target_alignment.csv
LOGS=$ROOT/logs
POOL=${POOL:-500}
QUERIES=${QUERIES:-512}
CHUNK=${CHUNK:-25}
mkdir -p "$LOGS"

for dataset in ETTh1 ETTm1; do
  for pred in 96 192 336 720; do
    if [ -f "$CSV" ] && grep -q "^${dataset},${pred},future," "$CSV"; then
      echo "[skip] ${dataset}/${pred} already in $CSV"
      continue
    fi
    ckpt=$(ls -d checkpoints/stage2/${dataset}/seq${pred}_pred${pred}/*s2_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
      echo "[miss] no full_bank_kl checkpoint for ${dataset}/${pred}"
      continue
    fi
    echo "[run ] ${dataset}/${pred} pool=${POOL} queries=${QUERIES}"
    CUDA_VISIBLE_DEVICES=1 python scripts/analyze_forecast_utility_alignment.py \
      --checkpoint "$ckpt" --pool_size "$POOL" --max_queries "$QUERIES" \
      --candidate_chunk "$CHUNK" --csv "$CSV" \
      > "$LOGS/${dataset}_${pred}.log" 2>&1
    status=$?
    echo "[done] ${dataset}/${pred} exit=${status}"
  done
done

python scripts/build_utility_alignment_report.py --root "$ROOT"
