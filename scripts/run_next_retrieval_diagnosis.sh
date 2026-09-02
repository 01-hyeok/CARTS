#!/bin/bash
# Master pipeline -- decide what Stage-1 should learn to retrieve.
#
#   STEP 0  checkpoint / dataset inventory
#   STEP 1  Stage-2 gate ablation inside one trained checkpoint
#   STEP 2  Stage-1 cosine vs unnormalized L2 geometry
#   STEP 3  residual Oracle upper bound
#   STEP 4  forecast-utility Oracle and predictability
#   STEP 5  merged report + recommendation
#
# Steps are independent: one failing does not stop the rest. Each finished step
# drops a marker and each setting appends its row immediately, so a rerun
# resumes instead of recomputing.
#
#   GPU=1 bash scripts/run_next_retrieval_diagnosis.sh
#   FORCE=1 ...            ignore markers and recompute
#   STEPS="1 3 4" ...      run a subset
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-1}}"
DATA_ROOT="${DATA_ROOT:-../Dataset/Time-Series-Library_dataset/ETT-small/}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
STEPS="${STEPS:-0 1 2 3 4 5}"
FORCE="${FORCE:-0}"
REF_ARM="${REF_ARM:-full_bank_kl}"
TOP_K="${TOP_K:-10}"

METRICS="${PROJECT_ROOT}/metrics/next_retrieval_diagnosis"
LOGS="${PROJECT_ROOT}/logs/next_retrieval_diagnosis"
mkdir -p "${METRICS}" "${LOGS}"

has_step() { [[ " ${STEPS} " == *" $1 "* ]]; }
marker() { echo "${METRICS}/.done_$1"; }
step_done() { [ "${FORCE}" != "1" ] && [ -f "$(marker "$1")" ]; }

stage2_ckpt() {
  ls "./checkpoints/stage2/$1/seq$2_pred$2"/*_s2_${REF_ARM}_*/checkpoint.pth 2>/dev/null | head -1
}
stage1_ckpt() {
  ls "./checkpoints/stage1/$1/seq$2_pred$2"/*_${REF_ARM}_$1_$2_*/checkpoint.pth 2>/dev/null | head -1
}

# --------------------------------------------------------------- STEP 0 ----
if has_step 0; then
  echo "=============== STEP 0  inventory ==============="
  MISSING=0
  for D in "${DATASETS[@]}"; do
    for P in "${PRED_LENS[@]}"; do
      S1=$(stage1_ckpt "$D" "$P"); S2=$(stage2_ckpt "$D" "$P")
      printf '  %-6s pred%-4s stage1:%s stage2:%s\n' "$D" "$P" \
        "$([ -n "$S1" ] && echo OK || echo MISSING)" \
        "$([ -n "$S2" ] && echo OK || echo MISSING)"
      [ -z "$S1" ] || [ -z "$S2" ] && MISSING=$((MISSING+1))
    done
  done
  echo "  settings missing a checkpoint: ${MISSING}"
fi

# --------------------------------------------------------------- STEP 1 ----
if has_step 1; then
  echo "=============== STEP 1  Stage-2 gate ablation ==============="
  if step_done stage2_gate; then
    echo "  [skip] already done"
  else
    CSV="${METRICS}/stage2_gate.csv"; [ "${FORCE}" = "1" ] && rm -f "${CSV}"
    for D in "${DATASETS[@]}"; do
      for P in "${PRED_LENS[@]}"; do
        CK=$(stage2_ckpt "$D" "$P")
        [ -z "${CK}" ] && { echo "  [skip] no Stage-2 checkpoint ${D}/${P}"; continue; }
        grep -q "^${D},${P}," "${CSV}" 2>/dev/null && { echo "  [skip] ${D}/${P}"; continue; }
        echo "  ${D}/pred${P}"
        python -u scripts/analyze_stage2_gate.py --checkpoint "${CK}" --csv "${CSV}" \
          > "${LOGS}/gate_${D}_${P}.log" 2>&1 || echo "    [FAILED] see ${LOGS}/gate_${D}_${P}.log"
      done
    done
    [ -s "${CSV}" ] && touch "$(marker stage2_gate)"
  fi
fi

# --------------------------------------------------------------- STEP 2 ----
if has_step 2; then
  echo "=============== STEP 2  cosine vs L2 geometry ==============="
  if step_done geometry; then
    echo "  [skip] already done"
  else
    CSV="${METRICS}/stage1_geometry.csv"; [ "${FORCE}" = "1" ] && rm -f "${CSV}"
    OUT_CSV="${CSV}" DATASETS="${DATASETS[*]}" PRED_LENS=96 \
      bash scripts/run_stage1_geometry_ablation.sh 2>&1 | tee "${LOGS}/geometry_pred96.log"
    # Expand to every horizon only if L2 clearly wins at pred96.
    EXPAND=$(python - "${CSV}" <<'PY'
import csv, sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception: print(0); raise SystemExit
by={}
for r in rows:
    if r['split']!='test': continue
    by[(r['dataset'], r.get('retrieval_similarity','cosine'))]=r
expand=0
for ds in {k[0] for k in by}:
    c,l = by.get((ds,'cosine')), by.get((ds,'l2'))
    if not c or not l: continue
    gap=float(l['oracle_gap_recovery_at_10'])-float(c['oracle_gap_recovery_at_10'])
    cr=float(c['oracle_recall_at_10']); lr=float(l['oracle_recall_at_10'])
    rel=(lr-cr)/cr if cr>0 else 0.0
    mse=float(c['retrieved_future_mse_at_10'])-float(l['retrieved_future_mse_at_10'])
    if gap>=0.03 or rel>=0.20 or mse>=0.01: expand=1
print(expand)
PY
)
    if [ "${EXPAND}" = "1" ]; then
      echo "  L2 wins at pred96 -> expanding to every horizon"
      OUT_CSV="${CSV}" DATASETS="${DATASETS[*]}" PRED_LENS="192 336 720" \
        bash scripts/run_stage1_geometry_ablation.sh 2>&1 | tee "${LOGS}/geometry_full.log"
    else
      echo "  L2 does not clearly win -> full sweep skipped by design"
    fi
    [ -s "${CSV}" ] && touch "$(marker geometry)"
  fi
fi

# --------------------------------------------------------------- STEP 3 ----
if has_step 3; then
  echo "=============== STEP 3  residual Oracle upper bound ==============="
  if step_done residual_oracle; then
    echo "  [skip] already done"
  else
    CSV="${METRICS}/residual_oracle.csv"; [ "${FORCE}" = "1" ] && rm -f "${CSV}"
    for D in "${DATASETS[@]}"; do
      for P in "${PRED_LENS[@]}"; do
        CK=$(stage2_ckpt "$D" "$P")
        [ -z "${CK}" ] && { echo "  [skip] no Stage-2 checkpoint ${D}/${P}"; continue; }
        grep -q "^${D},${P}," "${CSV}" 2>/dev/null && { echo "  [skip] ${D}/${P}"; continue; }
        echo "  ${D}/pred${P}"
        python -u scripts/analyze_residual_oracle.py --checkpoint "${CK}" \
          --top_k "${TOP_K}" --csv "${CSV}" \
          > "${LOGS}/residual_${D}_${P}.log" 2>&1 || echo "    [FAILED] see ${LOGS}/residual_${D}_${P}.log"
      done
    done
    [ -s "${CSV}" ] && touch "$(marker residual_oracle)"
  fi
fi

# --------------------------------------------------------------- STEP 4 ----
if has_step 4; then
  echo "=============== STEP 4  forecast utility ==============="
  if step_done forecast_utility; then
    echo "  [skip] already done"
  else
    CSV="${METRICS}/forecast_utility.csv"; [ "${FORCE}" = "1" ] && rm -f "${CSV}"
    for D in "${DATASETS[@]}"; do
      for P in "${PRED_LENS[@]}"; do
        CK=$(stage2_ckpt "$D" "$P")
        [ -z "${CK}" ] && { echo "  [skip] no Stage-2 checkpoint ${D}/${P}"; continue; }
        grep -q "^${D},${P}," "${CSV}" 2>/dev/null && { echo "  [skip] ${D}/${P}"; continue; }
        # Reuse the residual alpha chosen on validation in STEP 3.
        ALPHA=$(python - "${METRICS}/residual_oracle.csv" "$D" "$P" <<'PY'
import csv, sys
try:
    for r in csv.DictReader(open(sys.argv[1])):
        if r['dataset']==sys.argv[2] and r['pred_len']==sys.argv[3]:
            print(r['residual_oracle_best_alpha']); raise SystemExit
except Exception: pass
print('1.0')
PY
)
        echo "  ${D}/pred${P}  alpha=${ALPHA}"
        python -u scripts/analyze_forecast_utility.py --checkpoint "${CK}" \
          --top_k "${TOP_K}" --alpha "${ALPHA}" --csv "${CSV}" \
          > "${LOGS}/utility_${D}_${P}.log" 2>&1 || echo "    [FAILED] see ${LOGS}/utility_${D}_${P}.log"
      done
    done
    [ -s "${CSV}" ] && touch "$(marker forecast_utility)"
  fi
fi

# --------------------------------------------------------------- STEP 5 ----
if has_step 5; then
  echo "=============== STEP 5  final report ==============="
  python -u scripts/build_next_diagnosis_report.py --root "${METRICS}" \
    2>&1 | tee "${LOGS}/final_report.log"
fi

echo
echo "metrics: ${METRICS}"
echo "logs:    ${LOGS}"
echo "report:  ${METRICS}/FINAL_REPORT.md"
