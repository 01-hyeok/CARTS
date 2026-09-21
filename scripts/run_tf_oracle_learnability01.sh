#!/usr/bin/env bash
# TRACK-A-TF-ORACLE-LEARNABILITY01 orchestrator.
#
#   scripts/run_tf_oracle_learnability01.sh <cell> [<cell> ...]
#   cell in: ETTh1_96 ETTh1_720 Weather_96 Weather_720 Solar_96 Solar_720
#
# Per cell: individual_tf_cosine (writes shared_init) then set_tf_cosine
# (reads shared_init) -- same scratch encoder/SetConditioner init and,
# because both use shuffle=True with the SAME --loader_seed, the same
# DataLoader batch order every epoch (asserted post-hoc via
# batch_order_hashes_*.json). Only --target differs between the two.
set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/TRACK-A-TF-ORACLE-LEARNABILITY01
CKPT=checkpoints/track_a_tf_oracle_learnability01
LOGS=logs/track_a_tf_oracle_learnability01
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1_ETTh1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S1_Solar_96="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"
S1_Solar_720="checkpoints/soft_set_mse/stage1/Solar/seq720_pred720/stage1_carts_softset_Solar_720_S0_wce_RelationStage1_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"

S2_ETTh1_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_ETTh1_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_Weather_96="checkpoints/stage2/custom/seq96_pred96/stage2_carts_softset_s2_Weather_96_S0_wce_RelationStage2_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S2_Weather_720="checkpoints/stage2/custom/seq720_pred720/stage2_carts_softset_s2_Weather_720_S0_wce_RelationStage2_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S2_Solar_96="checkpoints/stage2/Solar/seq96_pred96/stage2_carts_softset_s2_Solar_96_S0_wce_RelationStage2_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"
S2_Solar_720="checkpoints/stage2/Solar/seq720_pred720/stage2_carts_softset_s2_Solar_720_S0_wce_RelationStage2_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"

ARMS=(individual_tf_cosine set_tf_cosine)

run_cell() {
  local cell="$1"
  local pred_len s1 s2 extra_flags=()
  case "$cell" in
    ETTh1_96)    pred_len=96;  s1="$S1_ETTh1_96";    s2="$S2_ETTh1_96" ;;
    ETTh1_720)   pred_len=720; s1="$S1_ETTh1_720";   s2="$S2_ETTh1_720" ;;
    Weather_96)  pred_len=96;  s1="$S1_Weather_96";  s2="$S2_Weather_96" ;;
    Weather_720) pred_len=720; s1="$S1_Weather_720"; s2="$S2_Weather_720" ;;
    Solar_96)    pred_len=96;  s1="$S1_Solar_96";    s2="$S2_Solar_96"
                 extra_flags=(--channelwise_backward --memsafe) ;;
    Solar_720)   pred_len=720; s1="$S1_Solar_720";   s2="$S2_Solar_720"
                 extra_flags=(--channelwise_backward --memsafe) ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  if [[ ! -f "$s1" ]]; then
    echo "[ISSUE][SKIP] $cell: Stage-1 reference not found at $s1" >&2; return 1
  fi
  if [[ ! -f "$s2" ]]; then
    echo "[ISSUE][SKIP] $cell: Stage-2 host not found at $s2" >&2; return 1
  fi

  local init="$LOGS/shared_init_${cell}.pth"
  mkdir -p "$OUT/$cell"

  echo "########## cell $cell (pred_len=$pred_len) ##########"
  local first=1
  for arm in "${ARMS[@]}"; do
    local marker="$OUT/$cell/DONE_${arm}.marker"
    if [[ -f "$marker" ]]; then
      echo "---- [$cell] $arm already DONE, skipping ----"
      first=0
      continue
    fi
    local init_flag
    if [[ $first -eq 1 ]]; then init_flag="--shared_init_out $init"
    else init_flag="--shared_init_in $init"; fi

    echo "---- [$cell] $arm ----"
    python -u scripts/train_tf_oracle_learnability01.py \
      --reference_ckpt "$s1" --stage2_host "$s2" \
      --arm_name "$arm" --pred_len "$pred_len" --cell "$cell" \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 \
      --weight_decay 0.0 --batch_size 32 --init_seed 0 --loader_seed 0 \
      --chunk_size 4096 --test_epochs 1,5,10 --train_subset_size 512 \
      --train_subset_seed 0 --oracle_compute_impl safe "${extra_flags[@]}" \
      $init_flag \
      2>&1 | tee "$LOGS/${cell}_${arm}.log"
    if [[ ! -f "$marker" ]]; then
      echo "[ABORT] $cell/$arm did not produce its DONE marker -- stopping cell." >&2
      return 1
    fi
    first=0
  done

  # batch-order gate: both arms must share identical DataLoader order on
  # every overlapping epoch.
  python -u - "$cell" <<'PYEOF'
import json, sys
from pathlib import Path
cell = sys.argv[1]
out = Path("results/TRACK-A-TF-ORACLE-LEARNABILITY01") / cell
a = json.loads((out / "batch_order_hashes_individual_tf_cosine.json").read_text())
b = json.loads((out / "batch_order_hashes_set_tf_cosine.json").read_text())
shared = set(a) & set(b)
mismatched = [k for k in shared if a[k] != b[k]]
if mismatched:
    print(f"[ISSUE][ABORT] {cell}: batch-order mismatch on epochs {mismatched}")
    sys.exit(1)
print(f"[ok] {cell}: batch order identical on {len(shared)} shared epochs")
(out / ".batch_order_gate_passed").write_text("ok")
PYEOF

  echo "########## cell $cell complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "TRACK-A-TF-ORACLE-LEARNABILITY01 orchestrator finished $(date -Is)"
