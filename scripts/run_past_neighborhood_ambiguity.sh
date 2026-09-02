#!/bin/bash
# Past-neighborhood future ambiguity.
#
# Asks whether the target past *contains* the information needed to identify a
# useful historical correction, with no encoder involved. Two distance views are
# run because in ETT the raw distance is dominated by level: 'raw' answers "same
# level and shape", 'znorm' answers "same shape at any level".
#
#   DATASETS="ETTh1" PRED_LENS="96" bash scripts/run_past_neighborhood_ambiguity.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96})
PAST_METRICS=(${PAST_METRICS:-raw znorm})
SPLIT="${SPLIT:-test}"
FRACTIONS="${FRACTIONS:-0,0.01,0.05,0.1,1.0}"
IDENTITY_POOL="${IDENTITY_POOL:-2000}"
CHUNK="${CHUNK:-64}"
REF_ARM="${REF_ARM:-full_bank_kl}"
FORCE="${FORCE:-0}"

M="${OUT_METRICS:-./metrics/past_neighborhood_ambiguity}"
L="${OUT_LOGS:-./logs/past_neighborhood_ambiguity}"
mkdir -p "$M" "$L"
CSV="$M/ambiguity.csv"
[ "$FORCE" = "1" ] && rm -f "$CSV"

for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
  CK=$(ls "./checkpoints/stage2/$D/seq${P}_pred${P}"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "  [skip] no Stage-2 checkpoint $D/$P"; continue; }
  for METRIC in "${PAST_METRICS[@]}"; do
    command grep -q "^${D},${P},.*,${METRIC},${SPLIT}," "$CSV" 2>/dev/null && {
      echo "  [skip] done $D/$P/$METRIC"; continue; }
    echo "  [$D/pred$P] past_metric=$METRIC"
    python -u scripts/analyze_past_neighborhood_ambiguity.py \
      --checkpoint "$CK" --split "$SPLIT" --past_metric "$METRIC" \
      --fractions "$FRACTIONS" --identity_pool "$IDENTITY_POOL" \
      --chunk "$CHUNK" --csv "$CSV" \
      > "$L/${D}_${P}_${METRIC}.log" 2>&1 || echo "    [FAILED] see $L/${D}_${P}_${METRIC}.log"
  done
done; done

python -u scripts/build_ambiguity_report.py --root "$M" 2>&1 | tee "$L/report.log"
echo; echo "metrics: $M"; echo "report:  $M/FINAL_REPORT.md"
