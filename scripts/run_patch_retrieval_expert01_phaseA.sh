#!/usr/bin/env bash
# TRACK-A-PATCH-RETRIEVAL-EXPERT01 Phase A orchestrator.
#   scripts/run_patch_retrieval_expert01_phaseA.sh <cell> [<cell> ...]
#   cell in: ETTh1_720 Weather_720 Solar_720
# Per cell: temperature calibration once (skipped if already present), then
# 4 patch arms in order (native_p16, p24, p48, p120) -- each independent
# (no shared init required: different architectures by construction, not a
# paired ablation like TRACK-A-HORIZON-RETRIEVAL-EXPERT01's global/block).
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/TRACK-A-PATCH-RETRIEVAL-EXPERT01
CKPT=checkpoints/track_a_patch_retrieval_expert01
LOGS=logs/patch_retrieval_expert01
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S1_Solar_720="checkpoints/soft_set_mse/stage1/Solar/seq720_pred720/stage1_carts_softset_Solar_720_S0_wce_RelationStage1_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"
S1_Solar_96_ARGS_FALLBACK="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"

# arm_name:patch_len
ARMS=(native_p16:16 p24:24 p48:48 p120:120)

run_cell() {
  local cell="$1" s1
  case "$cell" in
    ETTh1_720)   s1="$S1_ETTh1_720" ;;
    Weather_720) s1="$S1_Weather_720" ;;
    Solar_720)
      if [[ -f "$S1_Solar_720" ]]; then s1="$S1_Solar_720"
      else
        echo "[NOTE] Solar_720: no trained S0_wce checkpoint -- using Solar_96's args with pred_len/seq_len overridden to 720 (ARGS ONLY; consistent with TRACK-A-HORIZON-RETRIEVAL-EXPERT01's documented fallback)." >&2
        s1="$S1_Solar_96_ARGS_FALLBACK"
      fi
      ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  if [[ ! -f "$s1" ]]; then
    echo "[ISSUE][SKIP] $cell: reference checkpoint not found at $s1" >&2; return 1
  fi

  mkdir -p "$OUT/$cell"
  local calib="$OUT/$cell/temperature_calibration.json"
  if [[ ! -f "$calib" ]]; then
    echo "---- [$cell] temperature calibration ----"
    python -u scripts/calibrate_patch_retrieval_expert01_temperature.py \
      --reference_ckpt "$s1" --cell "$cell" --n_queries 512 --batch_size 16 \
      --out_dir "$OUT" 2>&1 | tee "$LOGS/${cell}_calibration.log"
  fi
  if [[ ! -f "$calib" ]]; then
    echo "[ISSUE][ABORT] $cell: calibration did not produce $calib" >&2; return 1
  fi
  local tau_t
  tau_t=$(python -c "import json; print(json.load(open('$calib'))['selected_tau_t'])")
  echo "[patch_retrieval_expert01] $cell: calibrated tau_t=$tau_t"

  for entry in "${ARMS[@]}"; do
    local arm="${entry%%:*}" patch_len="${entry##*:}"
    local marker="$OUT/$cell/DONE_${arm}.marker"
    if [[ -f "$marker" ]]; then
      echo "---- [$cell] $arm already DONE, skipping ----"
      continue
    fi
    echo "---- [$cell] $arm (patch_len=$patch_len) ----"
    python -u scripts/train_patch_retrieval_expert01.py \
      --reference_ckpt "$s1" --arm_name "$arm" --cell "$cell" \
      --patch_len "$patch_len" --tau_t "$tau_t" \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 \
      --weight_decay 0.0 --batch_size 32 --init_seed 0 --loader_seed 0 \
      --chunk_size 4096 \
      2>&1 | tee "$LOGS/${cell}_${arm}.log"
    if [[ ! -f "$marker" ]]; then
      echo "[ABORT] $cell/$arm did not produce its DONE marker -- stopping cell." >&2
      return 1
    fi
  done
  echo "########## [$cell] Phase A complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "TRACK-A-PATCH-RETRIEVAL-EXPERT01 Phase A orchestrator finished $(date -Is)"
