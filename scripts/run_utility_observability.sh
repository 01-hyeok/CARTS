#!/bin/bash
# Why does the learned reranker not recover the oracle headroom?
#
#   STEP 1  oracle feature ladder      A past -> B +cand R -> C +pred R_q -> D +true R_q -> E +future
#   STEP 2  similar-past oracle instability   (no training)
#   STEP 3  observable feature probe
#   STEP 4  permutation controls
#
# Separates information availability from model capacity: every rung shares the
# shortlist, the backbone and the loss, so a jump between rungs is a jump in
# what the selector is allowed to know. D and E are oracle diagnostics and are
# marked non-deployable wherever they appear.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

STEPS="${STEPS:-1 2 3 4}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
POOLS=(${POOLS:-100 500})
ARMS=(${ARMS:-A_past B_cand_residual C_pred_query_residual D_true_query_residual E_query_future})
TARGET="${TARGET:-regression}"
TOP_K="${TOP_K:-10}"
PRED="${PRED:-96}"
EPOCHS="${EPOCHS:-20}"
REF_ARM="${REF_ARM:-ps2_future_kl_m0}"
CACHE="${CACHE_DIR:-./cache/utility_reranker}"
FORCE="${FORCE:-0}"

M="${OUT_METRICS:-./metrics/utility_observability}"
L="${OUT_LOGS:-./logs/utility_observability}"
mkdir -p "$M" "$L"
LADDER="$M/feature_ladder.csv"; LADDER_F="$M/feature_ladder_forecast.csv"
STAB="$M/similar_past_oracle_stability.csv"
PROBE="$M/feature_probe.csv"; PERM="$M/permutation_control.csv"
[ "$FORCE" = "1" ] && rm -f "$M"/*.csv

has(){ [[ " ${STEPS} " == *" $1 "* ]]; }
ck(){ ls "./checkpoints/stage2/$1/seq${PRED}_pred${PRED}"/*_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1; }

if has 1; then
  for D in "${DATASETS[@]}"; do
    CK=$(ck "$D"); [ -z "$CK" ] && { echo "  [skip] no checkpoint $D"; continue; }
    for POOL in "${POOLS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        command grep -q "^${D},${PRED},${POOL},${TOP_K},${ARM}," "$LADDER" 2>/dev/null && {
          echo "  [skip] $D M=$POOL $ARM"; continue; }
        BASE=0
        command grep -q "^${D},${PRED},${POOL},${TOP_K},original," "$LADDER" 2>/dev/null || BASE=1
        echo "  [step1] $D M=$POOL $ARM"
        python -u scripts/train_feature_ladder.py --checkpoint "$CK" \
          --cache_dir "$CACHE" --dataset "$D" --pred_len "$PRED" --pool_m "$POOL" \
          --arm "$ARM" --target "$TARGET" --top_k "$TOP_K" --epochs "$EPOCHS" \
          --baselines "$BASE" --ladder_csv "$LADDER" --forecast_csv "$LADDER_F" \
          > "$L/step1_${D}_${POOL}_${ARM}.log" 2>&1 \
          || echo "    [FAILED] see $L/step1_${D}_${POOL}_${ARM}.log"
      done
    done
  done
  python -u scripts/build_observability_report.py --root "$M" 2>&1 | tee "$L/step1_report.log"
fi

if has 2; then
  for D in "${DATASETS[@]}"; do
    CK=$(ck "$D"); [ -z "$CK" ] && continue
    command grep -q "^${D},${PRED}," "$STAB" 2>/dev/null && { echo "  [skip] step2 $D"; continue; }
    echo "  [step2] $D similar-past oracle stability"
    python -u scripts/analyze_similar_past_stability.py --checkpoint "$CK" \
      --csv "$STAB" --summary_csv "$M/similar_past_summary.csv" \
      > "$L/step2_${D}.log" 2>&1 || echo "    [FAILED] see $L/step2_${D}.log"
  done
fi

if has 3 || has 4; then
  for D in "${DATASETS[@]}"; do
    for POOL in "${POOLS[@]}"; do
      command grep -q "^${D},${PRED},${POOL}," "$PROBE" 2>/dev/null && {
        echo "  [skip] probe $D M=$POOL"; continue; }
      echo "  [step3/4] $D M=$POOL feature probe + permutation controls"
      python -u scripts/analyze_feature_probe.py --cache_dir "$CACHE" \
        --dataset "$D" --pred_len "$PRED" --pool_m "$POOL" \
        --probe_csv "$PROBE" --permutation_csv "$PERM" \
        > "$L/step34_${D}_${POOL}.log" 2>&1 || echo "    [FAILED] see $L/step34_${D}_${POOL}.log"
    done
  done
fi

python -u scripts/build_observability_report.py --root "$M" 2>&1 | tee "$L/report.log"
echo; echo "metrics: $M"; echo "report:  $M/FINAL_REPORT.md"
