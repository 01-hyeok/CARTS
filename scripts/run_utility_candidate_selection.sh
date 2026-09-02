#!/bin/bash
# Utility-aware Candidate Selection.
#
#   STEP 1  existing binary classifier, re-used as a ranker (no training)
#   STEP 2  utility ranker            (CE / soft-KL / regression arms)
#   STEP 3  + candidate residual features
#   STEP 4  predicted query residual selector, and the direct-correction control
#   STEP 5  one forecast table over all arms
#   STEP 6  report + decision
#
# Every arm is scored on the same candidate pool, so a better number means a
# better choice of candidate rather than an easier pool. Pilot runs on pred96
# and expands to all horizons only if a learned selector beats current CARTS.
#
#   STEPS="1 5 6" FORCE=1 GPU=1 bash scripts/run_utility_candidate_selection.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PILOT_PRED=(${PILOT_PRED:-96})
FULL_PRED=(${FULL_PRED:-96 192 336 720})
POOL_M="${POOL_M:-100}"
EVAL_POOLS=(${EVAL_POOLS:-100 500})
ALPHA="${ALPHA:-1.0}"
STEPS="${STEPS:-1 2 3 4 5 6}"
FORCE="${FORCE:-0}"
REF_ARM="${REF_ARM:-full_bank_kl}"
EPOCHS="${EPOCHS:-15}"
RES_EPOCHS="${RES_EPOCHS:-30}"
MIN_GAIN="${MIN_GAIN:-0.005}"

M="${OUT_METRICS:-./metrics/utility_candidate_selection}"
L="${OUT_LOGS:-./logs/utility_candidate_selection}"
CLS_DIR="${CLS_DIR:-./metrics/candidate_utility_filtering/classifiers}"
mkdir -p "$M" "$L" "$M/models"

has(){ [[ " ${STEPS} " == *" $1 "* ]]; }
ck(){ ls "./checkpoints/stage2/$1/seq$2_pred$2"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1; }
seen(){ grep -q "$2" "$1" 2>/dev/null; }

PRED_LENS=("${PILOT_PRED[@]}")
run_all(){
for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
  CK=$(ck "$D" "$P"); [ -z "$CK" ] && { echo "  [skip] no checkpoint $D/$P"; continue; }

  if has 2; then
    for LOSS in ce kl; do
      OUT="$M/utility_ranker.csv"
      seen "$OUT" "^${D},${P},utility_ranker_${LOSS}," && continue
      echo "  [STEP2] $D/$P loss=$LOSS"
      python -u scripts/train_utility_ranker.py --checkpoint "$CK" --loss "$LOSS" \
        --pool_m "$POOL_M" --alpha "$ALPHA" --epochs "$EPOCHS" --csv "$OUT" \
        --save "$M/models/${D}_${P}_ranker_${LOSS}.pt" \
        > "$L/step2_${D}_${P}_${LOSS}.log" 2>&1 || echo "    [FAILED]"
    done
  fi

  if has 3; then
    for LOSS in ce kl; do
      OUT="$M/utility_ranker.csv"
      seen "$OUT" "^${D},${P},residual_aware_ranker_${LOSS}," && continue
      echo "  [STEP3] $D/$P loss=$LOSS +candidate residual"
      python -u scripts/train_utility_ranker.py --checkpoint "$CK" --loss "$LOSS" \
        --use_candidate_residual 1 --pool_m "$POOL_M" --alpha "$ALPHA" \
        --epochs "$EPOCHS" --csv "$OUT" \
        --save "$M/models/${D}_${P}_resranker_${LOSS}.pt" \
        > "$L/step3_${D}_${P}_${LOSS}.log" 2>&1 || echo "    [FAILED]"
    done
  fi

  if has 4; then
    OUT="$M/predicted_residual_selector.csv"
    seen "$OUT" "^${D},${P}," || {
      echo "  [STEP4] $D/$P residual selector"
      python -u scripts/train_query_residual_selector.py --checkpoint "$CK" \
        --pool_m "$POOL_M" --alpha "$ALPHA" --epochs "$RES_EPOCHS" --csv "$OUT" \
        --save "$M/models/${D}_${P}_residual_selector.pt" \
        > "$L/step4_${D}_${P}.log" 2>&1 || echo "    [FAILED]"
    }
  fi

  if has 1 || has 5; then
    for POOL in "${EVAL_POOLS[@]}"; do
      OUT="$M/classifier_topr.csv"
      seen "$OUT" "^${D},${P},base,${POOL}," && continue
      CLS=$(ls "$CLS_DIR/${D}_${P}_d0.0.pt" 2>/dev/null | head -1)
      echo "  [STEP1/5] $D/$P pool=$POOL classifier=${CLS:-none}"
      python -u scripts/evaluate_classifier_topr.py --checkpoint "$CK" \
        ${CLS:+--classifier "$CLS"} --pool_m "$POOL" --alpha "$ALPHA" \
        --split test --csv "$OUT" \
        > "$L/step5_${D}_${P}_${POOL}.log" 2>&1 || echo "    [FAILED]"
    done
  fi
done; done
}

[ "$FORCE" = "1" ] && rm -f "$M"/*.csv
echo "============ pilot: pred ${PILOT_PRED[*]} ============"
run_all

# Expand only if a learned selector actually beat current CARTS somewhere.
EXPAND=$(python - "$M" "$MIN_GAIN" <<'PY'
import csv,sys,os
root,thr=sys.argv[1],float(sys.argv[2])
def rd(n):
    p=os.path.join(root,n)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
cur={}
for r in rd('classifier_topr.csv'):
    if r['split']=='test' and r['method']=='current_topk_avg':
        cur[(r['dataset'],r['pred_len'])]=float(r['forecast_mse'])
best={}
for n in ('utility_ranker.csv','predicted_residual_selector.csv'):
    for r in rd(n):
        if r.get('split')!='test': continue
        k=(r['dataset'],r['pred_len'])
        for key in ('forecast_mse','direct_correction_mse'):
            if r.get(key):
                best[k]=min(best.get(k,9e9),float(r[key]))
print(1 if any(k in cur and best[k] <= cur[k]-thr for k in best) else 0)
PY
)
echo "learned selector beats current CARTS: ${EXPAND}"
if [ "${EXPAND}" = "1" ] && [ "${#FULL_PRED[@]}" -gt "${#PILOT_PRED[@]}" ]; then
  echo "============ expanding to pred ${FULL_PRED[*]} ============"
  PRED_LENS=("${FULL_PRED[@]}")
  run_all
fi

has 6 && python -u scripts/build_selection_report.py --root "$M" 2>&1 | tee "$L/report.log"
echo; echo "metrics: $M"; echo "report:  $M/FINAL_REPORT.md"
