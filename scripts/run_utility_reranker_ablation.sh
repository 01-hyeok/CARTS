#!/bin/bash
# Broad retrieval + query-conditioned utility reranking.
#
#   PHASE 0  oracle headroom inside the frozen retriever's Top-M  <- the gate
#   PHASE 1+ only run if Phase 0 finds headroom to recover
#
# The retriever and the Stage-2 forecaster both come from the same frozen
# future_kl_full checkpoint, and candidate sets are evaluated by masking the
# memory down to the set and running the real forward -- verified to reproduce
# production output exactly when the set is the retriever's own Top-K.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

PHASE="${PHASE:-0}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
POOLS=(${POOLS:-50 100 200 500})
TOP_K="${TOP_K:-10}"
PRED="${PRED:-96}"
MAX_QUERIES="${MAX_QUERIES:-512}"
CANDIDATE_CHUNK="${CANDIDATE_CHUNK:-10}"
GREEDY_QUERIES="${GREEDY_QUERIES:-0}"
REF_ARM="${REF_ARM:-ps2_future_kl_m0}"
FORCE="${FORCE:-0}"

M="${OUT_METRICS:-./metrics/utility_reranker}"
L="${OUT_LOGS:-./logs/utility_reranker}"
mkdir -p "$M" "$L"
CSV="$M/oracle_headroom.csv"
[ "$FORCE" = "1" ] && rm -f "$CSV"

ck(){ ls "./checkpoints/stage2/$1/seq${PRED}_pred${PRED}"/*_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1; }

if [ "$PHASE" = "0" ]; then
  for D in "${DATASETS[@]}"; do
    CK=$(ck "$D")
    [ -z "$CK" ] && { echo "  [skip] no ${REF_ARM} checkpoint for $D"; continue; }
    for POOL in "${POOLS[@]}"; do
      command grep -q "^${D},${PRED},${POOL},${TOP_K}," "$CSV" 2>/dev/null && {
        echo "  [skip] $D M=$POOL"; continue; }
      echo "  [phase0] $D/pred${PRED} M=$POOL"
      python -u scripts/analyze_oracle_rerank_headroom.py \
        --checkpoint "$CK" --pool_m "$POOL" --top_k "$TOP_K" \
        --max_queries "$MAX_QUERIES" --candidate_chunk "$CANDIDATE_CHUNK" \
        --greedy_queries "$GREEDY_QUERIES" --csv "$CSV" \
        > "$L/phase0_${D}_${PRED}_m${POOL}.log" 2>&1 \
        || echo "    [FAILED] see $L/phase0_${D}_${PRED}_m${POOL}.log"
    done
  done
  python -u scripts/build_reranker_report.py --root "$M" --phase 0 2>&1 | tee "$L/phase0_report.log"
fi

CACHE="${CACHE_DIR:-./cache/utility_reranker}"
TRAIN_Q="${TRAIN_Q:-3072}"; VAL_Q="${VAL_Q:-768}"; TEST_Q="${TEST_Q:-1536}"
ARMS=(${ARMS:-past_pair residual_aware})
TARGETS=(${TARGETS:-regression listwise_kl})
EPOCHS="${EPOCHS:-20}"
RERANK_CSV="$M/reranker_metrics.csv"
FORECAST_CSV="$M/stage2_forecast.csv"

if [ "$PHASE" = "1" ] || [ "$PHASE" = "2" ]; then
  mkdir -p "$CACHE"
  for D in "${DATASETS[@]}"; do
    CK=$(ck "$D"); [ -z "$CK" ] && { echo "  [skip] no checkpoint $D"; continue; }
    for POOL in "${POOLS[@]}"; do
      for SPLIT in train val test; do
        case "$SPLIT" in
          train) Q="$TRAIN_Q" ;; val) Q="$VAL_Q" ;; test) Q="$TEST_Q" ;;
        esac
        OUT="$CACHE/${D}_${PRED}_top${POOL}_${SPLIT}.pt"
        [ "$FORCE" != "1" ] && [ -f "$OUT" ] && { echo "  [skip] cache $OUT"; continue; }
        echo "  [phase1] $D M=$POOL $SPLIT (<=$Q queries)"
        python -u scripts/build_reranker_pool.py --checkpoint "$CK" --split "$SPLIT" \
          --pool_m "$POOL" --max_queries "$Q" --candidate_chunk "$CANDIDATE_CHUNK" \
          --out "$OUT" > "$L/phase1_${D}_${POOL}_${SPLIT}.log" 2>&1 \
          || echo "    [FAILED] see $L/phase1_${D}_${POOL}_${SPLIT}.log"
      done
    done
  done
fi

if [ "$PHASE" = "2" ]; then
  [ "$FORCE" = "1" ] && rm -f "$RERANK_CSV" "$FORECAST_CSV"
  for D in "${DATASETS[@]}"; do
    CK=$(ck "$D"); [ -z "$CK" ] && continue
    for POOL in "${POOLS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        for TARGET in "${TARGETS[@]}"; do
          command grep -q "^${D},${PRED},${POOL},${TOP_K},${ARM},${TARGET}," "$RERANK_CSV" 2>/dev/null && {
            echo "  [skip] $D M=$POOL $ARM/$TARGET"; continue; }
          echo "  [phase2] $D M=$POOL $ARM/$TARGET"
          # Baselines are written once per (dataset, M) with the first arm.
          BASE=0
          command grep -q "^${D},${PRED},${POOL},${TOP_K},original," "$RERANK_CSV" 2>/dev/null || BASE=1
          python -u scripts/train_utility_reranker.py --checkpoint "$CK" \
            --cache_dir "$CACHE" --dataset "$D" --pred_len "$PRED" --pool_m "$POOL" \
            --arm "$ARM" --target "$TARGET" --top_k "$TOP_K" --epochs "$EPOCHS" \
            --baselines "$BASE" --rerank_csv "$RERANK_CSV" --forecast_csv "$FORECAST_CSV" \
            > "$L/phase2_${D}_${POOL}_${ARM}_${TARGET}.log" 2>&1 \
            || echo "    [FAILED] see $L/phase2_${D}_${POOL}_${ARM}_${TARGET}.log"
        done
      done
    done
  done
  python -u scripts/build_reranker_report.py --root "$M" --phase 2 2>&1 | tee "$L/phase2_report.log"
fi

echo; echo "metrics: $M"
