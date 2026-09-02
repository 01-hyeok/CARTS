#!/bin/bash
# Retrieval Ranking Problem -> Candidate Utility Filtering Problem.
#
#   STEP 1  Oracle utility filtering upper bound   (no training)
#   STEP 2  is "this candidate helps" learnable from past only?
#   STEP 3  learned filter -> actual Stage-2 forecast
#   STEP 4  report + decision
#
# STEP 2 runs only if STEP 1 shows filtering is worth it; STEP 3 only if the
# classifier beats prevalence. Each setting appends immediately, so a rerun
# resumes. STEP 1 trains nothing and reuses existing Stage-2 checkpoints.
#
#   STEPS="1" FORCE=1 GPU=1 bash scripts/run_candidate_utility_filtering.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
POOLS=(${POOLS:-10 50 100 200 500})
RETRIEVERS=(${RETRIEVERS:-learned raw_l2})
DELTAS=(${DELTAS:-0.0 0.01 0.05})
STEPS="${STEPS:-1 2 3 4}"
FORCE="${FORCE:-0}"
REF_ARM="${REF_ARM:-full_bank_kl}"
ALPHA="${ALPHA:-1.0}"
CLS_EPOCHS="${CLS_EPOCHS:-20}"
# Filtering is worth pursuing only if dropping harmful candidates actually moves
# the forecast; classification only if it beats simply predicting the prevalence.
MIN_FILTERING_GAIN="${MIN_FILTERING_GAIN:-0.02}"
MIN_PRAUC_MARGIN="${MIN_PRAUC_MARGIN:-0.05}"

M="${PROJECT_METRICS:-./metrics/candidate_utility_filtering}"
L="${PROJECT_LOGS:-./logs/candidate_utility_filtering}"
mkdir -p "$M" "$L" "$M/classifiers"

has(){ [[ " ${STEPS} " == *" $1 "* ]]; }
s2ck(){ ls "./checkpoints/stage2/$1/seq$2_pred$2"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1; }

# ------------------------------------------------------------------ STEP 1 --
if has 1; then
  echo "============ STEP 1  oracle utility filtering ============"
  CSV="$M/oracle_filtering.csv"; [ "$FORCE" = "1" ] && rm -f "$CSV"
  for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
    CK=$(s2ck "$D" "$P"); [ -z "$CK" ] && { echo "  [skip] no checkpoint $D/$P"; continue; }
    for R in "${RETRIEVERS[@]}"; do for POOL in "${POOLS[@]}"; do
      grep -q "^${D},${P},${R},${POOL}," "$CSV" 2>/dev/null && continue
      echo "  $D/pred$P  $R  M=$POOL"
      python -u scripts/analyze_oracle_utility_filtering.py --checkpoint "$CK" \
        --retriever "$R" --pool_m "$POOL" --alpha "$ALPHA" --csv "$CSV" \
        > "$L/step1_${D}_${P}_${R}_${POOL}.log" 2>&1 \
        || echo "    [FAILED] see $L/step1_${D}_${P}_${R}_${POOL}.log"
    done; done
  done; done
fi

# Gate: is filtering worth pursuing at all?
FILTER_OK=$(python - "$M/oracle_filtering.csv" "$MIN_FILTERING_GAIN" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception: print("0"); raise SystemExit
gains=[float(r['filtering_gain']) for r in rows if r['filtering_gain']]
print("1" if gains and max(gains) >= float(sys.argv[2]) else "0")
PY
)
BEST_POOL=$(python - "$M/oracle_filtering.csv" <<'PY'
import csv,sys,collections
try: rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['retriever']=='learned']
except Exception: print(100); raise SystemExit
agg=collections.defaultdict(list)
for r in rows: agg[r['candidate_pool_m']].append(float(r['filtering_gain']))
print(max(agg, key=lambda m: sum(agg[m])/len(agg[m])) if agg else 100)
PY
)
echo "  filtering worth pursuing: ${FILTER_OK}   best pool M: ${BEST_POOL}"

# ------------------------------------------------------------------ STEP 2 --
if has 2 && [ "$FILTER_OK" = "1" ]; then
  echo "============ STEP 2  utility classifier ============"
  CSV="$M/classifier_results.csv"; [ "$FORCE" = "1" ] && rm -f "$CSV"
  for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
    CK=$(s2ck "$D" "$P"); [ -z "$CK" ] && continue
    for DELTA in "${DELTAS[@]}"; do
      grep -q "^${D},${P},${DELTA},${BEST_POOL}," "$CSV" 2>/dev/null && continue
      echo "  $D/pred$P  delta=$DELTA  M=$BEST_POOL"
      python -u scripts/train_utility_classifier.py --checkpoint "$CK" \
        --delta "$DELTA" --pool_m "$BEST_POOL" --alpha "$ALPHA" \
        --epochs "$CLS_EPOCHS" --csv "$CSV" \
        --save "$M/classifiers/${D}_${P}_d${DELTA}.pt" \
        > "$L/step2_${D}_${P}_${DELTA}.log" 2>&1 \
        || echo "    [FAILED] see $L/step2_${D}_${P}_${DELTA}.log"
    done
  done; done
elif has 2; then
  echo "============ STEP 2 skipped -- STEP 1 shows no filtering headroom ============"
fi

CLS_OK=$(python - "$M/classifier_results.csv" "$MIN_PRAUC_MARGIN" <<'PY'
import csv,sys
try: rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['split']=='test']
except Exception: print("0"); raise SystemExit
ok=[r for r in rows if float(r['pr_auc'])-float(r['positive_prevalence'])>=float(sys.argv[2])]
print("1" if ok else "0")
PY
)
echo "  classifier beats prevalence: ${CLS_OK}"

# ------------------------------------------------------------------ STEP 3 --
if has 3 && [ "$CLS_OK" = "1" ]; then
  echo "============ STEP 3  learned filtering -> Stage-2 ============"
  CSV="$M/stage2_filtering.csv"; [ "$FORCE" = "1" ] && rm -f "$CSV"
  for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
    CK=$(s2ck "$D" "$P"); [ -z "$CK" ] && continue
    # Use the delta whose test PR-AUC margin was largest for this setting.
    DELTA=$(python - "$M/classifier_results.csv" "$D" "$P" <<'PY'
import csv,sys
best,bd=-9,'0.0'
try:
    for r in csv.DictReader(open(sys.argv[1])):
        if r['split']=='test' and r['dataset']==sys.argv[2] and r['pred_len']==sys.argv[3]:
            m=float(r['pr_auc'])-float(r['positive_prevalence'])
            if m>best: best,bd=m,r['delta']
except Exception: pass
print(bd)
PY
)
    CLS="$M/classifiers/${D}_${P}_d${DELTA}.pt"
    [ -f "$CLS" ] || { echo "  [skip] no classifier $D/$P"; continue; }
    grep -q "^${D},${P}," "$CSV" 2>/dev/null && continue
    echo "  $D/pred$P  delta=$DELTA"
    python -u scripts/evaluate_candidate_filtering_stage2.py --checkpoint "$CK" \
      --classifier "$CLS" --csv "$CSV" \
      > "$L/step3_${D}_${P}.log" 2>&1 || echo "    [FAILED] see $L/step3_${D}_${P}.log"
  done; done
elif has 3; then
  echo "============ STEP 3 skipped -- classifier at prevalence level ============"
fi

# ------------------------------------------------------------------ STEP 4 --
if has 4; then
  echo "============ STEP 4  report ============"
  python -u scripts/build_filtering_report.py --root "$M" 2>&1 | tee "$L/report.log"
fi
echo; echo "metrics: $M"; echo "report:  $M/FINAL_REPORT.md"
