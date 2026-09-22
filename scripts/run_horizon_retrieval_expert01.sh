#!/usr/bin/env bash
# TRACK-A-HORIZON-RETRIEVAL-EXPERT01 orchestrator.
#   scripts/run_horizon_retrieval_expert01.sh <cell> [<cell> ...]
#   cell in: ETTh1_720 Weather_720 Solar_720
# Per cell: global arm (writes shared_init) then block arm (reads it) --
# same scratch encoder init and, since both use shuffle=True with the SAME
# --loader_seed, the same DataLoader batch order every epoch (asserted
# post-hoc via batch_order_hashes_*.json).
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01
CKPT=checkpoints/track_a_horizon_retrieval_expert01
LOGS=logs/horizon_retrieval_expert01
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S1_Solar_720="checkpoints/soft_set_mse/stage1/Solar/seq720_pred720/stage1_carts_softset_Solar_720_S0_wce_RelationStage1_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"
# fallback: Solar_720 has no trained S0_wce checkpoint (documented OOM); its
# ARGS ONLY are needed here (weights never loaded by this trainer either),
# so Solar_96's checkpoint args are reused with pred_len/seq_len explicitly
# overridden to 720 -- NEVER described as "Solar_720's trained host".
S1_Solar_96_ARGS_FALLBACK="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"

ARMS=(global block)

run_cell() {
  local cell="$1" s1 extra_flags=()
  case "$cell" in
    ETTh1_720)   s1="$S1_ETTh1_720" ;;
    Weather_720) s1="$S1_Weather_720" ;;
    Solar_720)
      if [[ -f "$S1_Solar_720" ]]; then s1="$S1_Solar_720"
      else
        echo "[NOTE] Solar_720: no trained S0_wce checkpoint -- using Solar_96's args with pred_len/seq_len overridden to 720 (ARGS ONLY, no weights loaded by this trainer regardless)." >&2
        s1="$S1_Solar_96_ARGS_FALLBACK"
      fi
      extra_flags=(--channelwise_backward)
      ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  if [[ ! -f "$s1" ]]; then
    echo "[ISSUE][SKIP] $cell: reference checkpoint not found at $s1" >&2; return 1
  fi

  local init="$LOGS/shared_init_${cell}.pth"
  mkdir -p "$OUT/$cell"
  local first=1
  for arm in "${ARMS[@]}"; do
    local marker="$OUT/$cell/DONE_${arm}.marker"
    if [[ -f "$marker" ]]; then
      echo "---- [$cell] $arm already DONE, skipping ----"
      first=0; continue
    fi
    local init_flag
    if [[ $first -eq 1 ]]; then init_flag="--shared_init_out $init"
    else init_flag="--shared_init_in $init"; fi

    echo "---- [$cell] $arm ----"
    python -u scripts/train_horizon_retrieval_expert01.py \
      --reference_ckpt "$s1" --arm "$arm" --cell "$cell" --pred_len 720 \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 \
      --weight_decay 0.0 --batch_size 32 --init_seed 0 --loader_seed 0 \
      --chunk_size 4096 --tau_t 0.1 --tau_s 0.1 "${extra_flags[@]}" $init_flag \
      2>&1 | tee "$LOGS/${cell}_${arm}.log"
    if [[ ! -f "$marker" ]]; then
      echo "[ABORT] $cell/$arm did not produce its DONE marker -- stopping cell." >&2
      return 1
    fi
    first=0
  done

  python -u - "$cell" <<'PYEOF'
import json, sys
from pathlib import Path
cell = sys.argv[1]
out = Path("results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01") / cell
a = json.loads((out / "batch_order_hashes_global.json").read_text())
b = json.loads((out / "batch_order_hashes_block.json").read_text())
shared = set(a) & set(b)
mismatched = [k for k in shared if a[k] != b[k]]
if mismatched:
    print(f"[ISSUE][ABORT] {cell}: batch-order mismatch on epochs {mismatched}")
    sys.exit(1)
print(f"[ok] {cell}: batch order identical on {len(shared)} shared epochs")
(out / ".batch_order_gate_passed").write_text("ok")
PYEOF

  echo "########## [$cell] complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "TRACK-A-HORIZON-RETRIEVAL-EXPERT01 orchestrator finished $(date -Is)"
