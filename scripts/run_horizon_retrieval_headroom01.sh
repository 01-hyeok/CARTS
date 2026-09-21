#!/usr/bin/env bash
# TRACK-A-HORIZON-RETRIEVAL-HEADROOM01 orchestrator.
#   scripts/run_horizon_retrieval_headroom01.sh <cell> [<cell> ...]
#   cell in: ETTh1_720 Weather_720 Solar_720
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/TRACK-A-HORIZON-RETRIEVAL-HEADROOM01
LOGS=logs/horizon_retrieval_headroom01
mkdir -p "$OUT" "$LOGS"

S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S1_Solar_720="checkpoints/soft_set_mse/stage1/Solar/seq720_pred720/stage1_carts_softset_Solar_720_S0_wce_RelationStage1_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"

run_cell() {
  local cell="$1" s1
  case "$cell" in
    ETTh1_720)   s1="$S1_ETTh1_720" ;;
    Weather_720) s1="$S1_Weather_720" ;;
    Solar_720)   s1="$S1_Solar_720" ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  local marker="$OUT/$cell/DONE"
  if [[ -f "$marker" ]]; then
    echo "---- [$cell] already DONE, skipping ----"
    return 0
  fi
  if [[ ! -f "$s1" ]]; then
    echo "[ISSUE][SKIP] $cell: Stage-1 reference not found at $s1 (weights not loaded, config-only)" >&2
    return 1
  fi
  echo "########## [$cell] Horizon-Retrieval-Headroom diagnostic ##########"
  python -u scripts/diag_horizon_retrieval_headroom01.py \
    --reference_ckpt "$s1" --pred_len 720 --cell "$cell" \
    --out_dir "$OUT" --top_k 10 --chunk_size 4096 --splits train,val,test \
    2>&1 | tee "$LOGS/${cell}.log"
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    echo "[FAIL] $cell diagnostic did not complete" >&2; return 1
  fi
  python -u scripts/analyze_horizon_retrieval_headroom01.py --out_dir "$OUT" --cells "$cell" \
    2>&1 | tee -a "$LOGS/${cell}.log"
  touch "$marker"
  echo "########## [$cell] complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "TRACK-A-HORIZON-RETRIEVAL-HEADROOM01 orchestrator finished $(date -Is)"
