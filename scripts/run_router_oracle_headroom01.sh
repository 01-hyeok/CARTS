#!/usr/bin/env bash
# ROUTER-ORACLE-HEADROOM01 Phase-1 orchestrator.
#   scripts/run_router_oracle_headroom01.sh <cell> [<cell> ...]
#   cell in: ETTh1_96 ETTh1_720 Weather_96 Weather_720 Solar_96 Solar_720
#
# Per cell: run scripts/diag_router_oracle_headroom01.py against the
# already-trained TRACK-A-TF-ORACLE-LEARNABILITY01 individual_tf_cosine
# checkpoint (R2's checkpoint provenance), covering train/val/test in one
# pass, then scripts/analyze_router_oracle_headroom01.py for the summary.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/ROUTER-ORACLE-HEADROOM01
LOGS=logs/router_oracle_headroom01
mkdir -p "$OUT" "$LOGS"

CKPT_ROOT=checkpoints/track_a_tf_oracle_learnability01

run_cell() {
  local cell="$1" pred_len ckpt
  case "$cell" in
    ETTh1_96)    pred_len=96 ;;
    ETTh1_720)   pred_len=720 ;;
    Weather_96)  pred_len=96 ;;
    Weather_720) pred_len=720 ;;
    Solar_96)    pred_len=96 ;;
    Solar_720)   pred_len=720 ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  ckpt="$CKPT_ROOT/$cell/individual_tf_cosine/checkpoint.pth"
  local marker="$OUT/$cell/DONE"
  if [[ -f "$marker" ]]; then
    echo "---- [$cell] already DONE, skipping ----"
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[ISSUE][SKIP] $cell: individual_tf_cosine checkpoint not found at $ckpt" >&2
    return 1
  fi
  echo "########## [$cell] Phase-1 diagnostic ##########"
  python -u scripts/diag_router_oracle_headroom01.py \
    --learned_ckpt "$ckpt" --pred_len "$pred_len" --cell "$cell" \
    --out_dir "$OUT" --top_k 10 --chunk_size 4096 --splits train,val,test \
    2>&1 | tee "$LOGS/${cell}.log"
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    echo "[FAIL] $cell diagnostic did not complete" >&2
    return 1
  fi
  python -u scripts/analyze_router_oracle_headroom01.py --out_dir "$OUT" --cells "$cell" \
    2>&1 | tee -a "$LOGS/${cell}.log"
  touch "$marker"
  echo "########## [$cell] complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "ROUTER-ORACLE-HEADROOM01 orchestrator finished $(date -Is)"
