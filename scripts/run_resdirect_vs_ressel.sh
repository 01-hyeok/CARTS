#!/bin/bash
# Oracle Hybrid diagnostic: is retrieval complementary to direct correction?
#
# Reuses the residual selectors trained by run_utility_candidate_selection.sh --
# nothing is retrained, so ResDirect and ResSel come from the same checkpoint
# and the same split by construction.
#
# WAIT_FOR=1 blocks until the selection pipeline finishes, so this can be
# queued behind it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
REF_ARM="${REF_ARM:-full_bank_kl}"
SELECTOR_DIR="${SELECTOR_DIR:-./metrics/utility_candidate_selection/models}"
M="${OUT_METRICS:-./metrics/resdirect_vs_ressel}"
L="${OUT_LOGS:-./logs/resdirect_vs_ressel}"
FORCE="${FORCE:-0}"
mkdir -p "$M" "$L" "$M/query_level"

if [ "${WAIT_FOR:-0}" = "1" ]; then
  echo "waiting for run_utility_candidate_selection.sh to finish..."
  while pgrep -f "run_utility_candidate_selection.sh" > /dev/null; do sleep 30; done
  echo "selection pipeline finished; starting"
fi

CSV="$M/oracle_hybrid_summary.csv"
[ "$FORCE" = "1" ] && rm -f "$CSV"
for D in "${DATASETS[@]}"; do for P in "${PRED_LENS[@]}"; do
  CK=$(ls "./checkpoints/stage2/$D/seq${P}_pred${P}"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1)
  SEL="$SELECTOR_DIR/${D}_${P}_residual_selector.pt"
  [ -z "$CK" ] && { echo "  [skip] no Stage-2 checkpoint $D/$P"; continue; }
  [ -f "$SEL" ] || { echo "  [skip] no residual selector $D/$P"; continue; }
  grep -q "^${D},${P}," "$CSV" 2>/dev/null && { echo "  [skip] done $D/$P"; continue; }
  echo "  $D/pred$P"
  python -u scripts/analyze_resdirect_vs_ressel.py --checkpoint "$CK" --selector "$SEL" \
    --csv "$CSV" --query_csv "$M/query_level/${D}_${P}.csv" \
    > "$L/${D}_${P}.log" 2>&1 || echo "    [FAILED] see $L/${D}_${P}.log"
done; done

python -u scripts/build_hybrid_report.py --root "$M" 2>&1 | tee "$L/report.log"
echo; echo "metrics: $M"; echo "report:  $M/oracle_hybrid_report.md"
