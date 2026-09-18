#!/usr/bin/env bash
# TRACK-A-SET-LOSS-CONTROL02 -- ETTh1_96, 4 arms, one at a time, GPU1 only.
# A0 must already be done (writes the shared init) before this is run for
# A1-A3; if A0's DONE marker is missing this script builds it too (first
# in ARMS). DONE_<arm>.marker gates re-execution. After all 4 finish, runs
# the cross-arm batch-order hash gate (spec section 5) -- aborts loudly if
# it fails, but does NOT delete any already-written results.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
OUT="results/TRACK-A-SET-LOSS-CONTROL02"
INIT="logs/TRACK-A-SET-LOSS-CONTROL02/shared_init_ETTh1_H96_seed0.pth"

mkdir -p logs/TRACK-A-SET-LOSS-CONTROL02 "$OUT/ETTh1_96"

ARMS=(A0_hard_choice A1_adaptive_multipos A2_srm A3_setutility_softce)

for arm in "${ARMS[@]}"; do
  marker="$OUT/ETTh1_96/DONE_${arm}.marker"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_96] $arm already done, skipping ----"
    continue
  fi

  init_flag="--shared_init_in $INIT"
  [[ "$arm" == "A0_hard_choice" && ! -f "$INIT" ]] && init_flag="--shared_init_out $INIT"

  echo "---- [ETTh1_96] Stage-1 (CONTROL02): $arm ----"
  python -u scripts/train_set_loss_control02.py \
    --reference_ckpt "$S1_96" --stage2_host "$S2_96" --arm "$arm" \
    --pred_len 96 --cell ETTh1_96 --init_seed 0 --loader_seed 0 \
    $init_flag \
    2>&1 | tee "logs/TRACK-A-SET-LOSS-CONTROL02/stage1_ETTh1_96_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done

echo "---- batch-order cross-arm gate (spec section 5) ----"
python -u scripts/gate_set_loss_control02_batch_order.py "$OUT/ETTh1_96"
gate_status=$?
if [[ $gate_status -ne 0 ]]; then
  echo "[ISSUE][ABORT] batch-order gate FAILED for ETTh1_96 -- see output above." >&2
  exit 1
fi
touch "$OUT/ETTh1_96/.batch_order_gate_passed"

echo "########## TRACK-A-SET-LOSS-CONTROL02 ETTh1_96: all 4 arms complete, batch-order gate PASSED ##########"
