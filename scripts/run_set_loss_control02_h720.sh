#!/usr/bin/env bash
# TRACK-A-SET-LOSS-CONTROL02 -- ETTh1_720, 4 arms, one at a time, GPU1
# only. Only run this AFTER ETTh1_96's batch-order gate has passed (spec
# section 9: "H96의 모든 hash와 결과를 검증한 뒤 H720으로 진행하라").
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

H96_GATE_OK="results/TRACK-A-SET-LOSS-CONTROL02/ETTh1_96/.batch_order_gate_passed"
if [[ ! -f "$H96_GATE_OK" ]]; then
  echo "[ISSUE][ABORT] ETTh1_96 batch-order gate has not been confirmed passed yet " \
       "($H96_GATE_OK missing) -- refusing to start H720 before H96 verification (spec section 9)." >&2
  exit 1
fi

S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
OUT="results/TRACK-A-SET-LOSS-CONTROL02"
INIT="logs/TRACK-A-SET-LOSS-CONTROL02/shared_init_ETTh1_H720_seed0.pth"

mkdir -p logs/TRACK-A-SET-LOSS-CONTROL02 "$OUT/ETTh1_720"

ARMS=(A0_hard_choice A1_adaptive_multipos A2_srm A3_setutility_softce)

for arm in "${ARMS[@]}"; do
  marker="$OUT/ETTh1_720/DONE_${arm}.marker"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_720] $arm already done, skipping ----"
    continue
  fi

  init_flag="--shared_init_in $INIT"
  [[ "$arm" == "A0_hard_choice" && ! -f "$INIT" ]] && init_flag="--shared_init_out $INIT"

  echo "---- [ETTh1_720] Stage-1 (CONTROL02): $arm ----"
  python -u scripts/train_set_loss_control02.py \
    --reference_ckpt "$S1_720" --stage2_host "$S2_720" --arm "$arm" \
    --pred_len 720 --cell ETTh1_720 --init_seed 0 --loader_seed 0 \
    $init_flag \
    2>&1 | tee "logs/TRACK-A-SET-LOSS-CONTROL02/stage1_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done

echo "---- batch-order cross-arm gate (spec section 5) ----"
python -u scripts/gate_set_loss_control02_batch_order.py "$OUT/ETTh1_720"
gate_status=$?
if [[ $gate_status -ne 0 ]]; then
  echo "[ISSUE][ABORT] batch-order gate FAILED for ETTh1_720 -- see output above." >&2
  exit 1
fi
touch "$OUT/ETTh1_720/.batch_order_gate_passed"

echo "########## TRACK-A-SET-LOSS-CONTROL02 ETTh1_720: all 4 arms complete, batch-order gate PASSED ##########"
