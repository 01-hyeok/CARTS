#!/bin/bash
# Can cross-channel context revive historical correction selection?
#
#   A  target-only ResDirect          no retrieval, target past only
#   B  cross-channel ResDirect        no retrieval, + source pasts
#   C  target-only ResSel             retrieval, target past only
#   D  query cross-channel ResSel     retrieval, + source pasts   <- core arm
#   E  query+candidate cross-channel ResSel                       (phase 2)
#   F  query cross-channel + candidate residual ResSel            (phase 2)
#
# The verdict is NOT "does cross-channel beat target-only". It is D vs B: given
# the same source information, does selecting a historical correction beat
# predicting one directly. Phase 1 runs A-D on pred96; E/F and the full horizon
# sweep only run if D actually improves on C.
#
#   PHASE=1 GPU=1 bash scripts/run_cross_channel_retrieval_ablation.sh
#   PHASE=2 FORCE=1 ... to redo the ablation arms
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PILOT_PRED=(${PILOT_PRED:-96})
FULL_PRED=(${FULL_PRED:-96 192 336 720})
PHASE="${PHASE:-1}"
TOPK="${TOPK:-5}"
POOL_M="${POOL_M:-100}"
ALPHA="${ALPHA:-1.0}"
SEL_LOSS="${SEL_LOSS:-top1_ce}"
SEL_EPOCHS="${SEL_EPOCHS:-15}"
DIR_EPOCHS="${DIR_EPOCHS:-30}"
SCALE_INIT="${SCALE_INIT:-1e-2}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
REF_ARM="${REF_ARM:-full_bank_kl}"
MIN_GAIN="${MIN_GAIN:-0.0}"

M="${OUT_METRICS:-./metrics/cross_channel_retrieval}"
L="${OUT_LOGS:-./logs/cross_channel_retrieval}"
mkdir -p "$M" "$L" "$M/models"
DIRECT_CSV="$M/resdirect.csv"
SEL_CSV="$M/ressel.csv"

[ "$FORCE" = "1" ] && rm -f "$M"/*.csv

ck(){ ls "./checkpoints/stage2/$1/seq$2_pred$2"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1; }
# Resume key: dataset, pred_len and arm name identify a finished run.
done_row(){ command grep -q "^$2,$3,$4," "$1" 2>/dev/null; }

run_direct(){  # dataset pred checkpoint cross_channel arm_name
  local D="$1" P="$2" CK="$3" CC="$4" ARM="$5"
  done_row "$DIRECT_CSV" "$D" "$P" "$ARM" && { echo "    [skip] $ARM $D/$P"; return; }
  echo "    [$ARM] $D/$P"
  # "0" is a non-empty string, so the attention flag needs a real test.
  local EXTRA=()
  [ "$CC" = "1" ] && EXTRA=(--attention_csv "$M/attention_summary.csv")
  python -u scripts/train_cross_channel_resdirect.py \
    --checkpoint "$CK" --use_cross_channel_context "$CC" \
    --cross_channel_topk "$TOPK" --cross_channel_scale_init "$SCALE_INIT" \
    --alpha "$ALPHA" --epochs "$DIR_EPOCHS" --seed "$SEED" \
    --csv "$DIRECT_CSV" --save "$M/models/${D}_${P}_${ARM}.pt" \
    "${EXTRA[@]}" --source_csv "$M/source_channels.csv" \
    > "$L/${D}_${P}_${ARM}.log" 2>&1 || echo "      [FAILED] see $L/${D}_${P}_${ARM}.log"
}

run_sel(){  # dataset pred checkpoint query_cc candidate_cc candidate_residual arm
  local D="$1" P="$2" CK="$3" QCC="$4" KCC="$5" KRES="$6" ARM="$7"
  done_row "$SEL_CSV" "$D" "$P" "$ARM" && { echo "    [skip] $ARM $D/$P"; return; }
  echo "    [$ARM] $D/$P"
  python -u scripts/train_cross_channel_ressel.py \
    --checkpoint "$CK" --use_cross_channel_context "$QCC" \
    --candidate_cross_channel_context "$KCC" \
    --use_candidate_residual_feature "$KRES" \
    --cross_channel_topk "$TOPK" --cross_channel_scale_init "$SCALE_INIT" \
    --utility_selection_loss "$SEL_LOSS" --pool_m "$POOL_M" --alpha "$ALPHA" \
    --epochs "$SEL_EPOCHS" --seed "$SEED" \
    --csv "$SEL_CSV" --save "$M/models/${D}_${P}_${ARM}.pt" \
    > "$L/${D}_${P}_${ARM}.log" 2>&1 || echo "      [FAILED] see $L/${D}_${P}_${ARM}.log"
}

run_settings(){  # phase, then the pred lens
  local phase="$1"; shift
  for D in "${DATASETS[@]}"; do for P in "$@"; do
    CK=$(ck "$D" "$P")
    [ -z "$CK" ] && { echo "  [skip] no Stage-2 checkpoint $D/$P"; continue; }
    echo "  == $D / pred$P =="
    if [ "$phase" = "1" ]; then
      run_direct "$D" "$P" "$CK" 0 target_only_resdirect
      run_direct "$D" "$P" "$CK" 1 cross_channel_resdirect
      run_sel    "$D" "$P" "$CK" 0 0 0 target_only_ressel
      run_sel    "$D" "$P" "$CK" 1 0 0 query_cross_channel_ressel
    else
      run_sel    "$D" "$P" "$CK" 1 1 0 query_candidate_cross_channel_ressel
      run_sel    "$D" "$P" "$CK" 1 0 1 query_cross_channel_residual_aware_ressel
    fi
  done; done
}

# Did query-side cross-channel selection improve on target-only selection, on
# test, in forecast MSE and in selection quality at the same time?
d_beats_c(){
  python - "$SEL_CSV" "$MIN_GAIN" <<'PY'
import csv, os, sys
path, thr = sys.argv[1], float(sys.argv[2])
rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
by = {}
for r in rows:
    if r['split'] != 'test':
        continue
    by[(r['dataset'], r['pred_len'], r['arm'])] = r
hit = 0
for (dataset, pred, arm), row in by.items():
    if arm != 'query_cross_channel_ressel':
        continue
    base = by.get((dataset, pred, 'target_only_ressel'))
    if not base:
        continue
    better_mse = float(row['forecast_mse']) <= float(base['forecast_mse']) - thr
    better_sel = (
        float(row['positive_at_1']) > float(base['positive_at_1'])
        and float(row['selected_utility_at_1']) > float(base['selected_utility_at_1'])
        and float(row['utility_regret_at_1']) < float(base['utility_regret_at_1'])
    )
    if better_mse and better_sel:
        hit = 1
print(hit)
PY
}

echo "============ phase 1: A-D on pred ${PILOT_PRED[*]} ============"
run_settings 1 "${PILOT_PRED[@]}"

EXPAND=$(d_beats_c)
echo "D improves on C (forecast and selection together): ${EXPAND}"

if [ "${EXPAND}" = "1" ]; then
  if [ "${#FULL_PRED[@]}" -gt "${#PILOT_PRED[@]}" ]; then
    echo "============ expanding A-D to pred ${FULL_PRED[*]} ============"
    run_settings 1 "${FULL_PRED[@]}"
  fi
  echo "============ phase 2: E/F ============"
  run_settings 2 "${PILOT_PRED[@]}"
elif [ "$PHASE" = "2" ]; then
  echo "============ phase 2 forced ============"
  run_settings 2 "${PILOT_PRED[@]}"
else
  echo "phase 2 skipped: query-side cross-channel selection did not improve on"
  echo "target-only selection, which is the precondition for the E/F ablation."
fi

python -u scripts/build_cross_channel_report.py --root "$M" 2>&1 | tee "$L/report.log"
echo; echo "metrics: $M"; echo "report:  $M/FINAL_REPORT.md"
