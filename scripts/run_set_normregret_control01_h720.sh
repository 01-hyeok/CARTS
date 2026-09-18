#!/usr/bin/env bash
# TRACK-A-SET-NORMREGRET-CONTROL01 -- ETTh1_720, 4 arms, one at a time,
# GPU1 only. Run only after H96's 4 arms are DONE with finite loss/gradient
# and 7-channel processing confirmed.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

for arm in A0_hard_choice A1_raw_wce A2_normregret_softce A3_normregret_srm; do
  if [[ ! -f "results/TRACK-A-SET-NORMREGRET-CONTROL01/ETTh1_96/DONE_${arm}.marker" ]]; then
    echo "[ISSUE][ABORT] ETTh1_96/$arm not complete yet -- refusing to start H720." >&2
    exit 1
  fi
done

S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
OUT="results/TRACK-A-SET-NORMREGRET-CONTROL01"
INIT="logs/TRACK-A-SET-NORMREGRET-CONTROL01/shared_init_ETTh1_720.pth"

mkdir -p logs/TRACK-A-SET-NORMREGRET-CONTROL01 "$OUT/ETTh1_720"

ARMS=(A0_hard_choice A1_raw_wce A2_normregret_softce A3_normregret_srm)

first=1
for arm in "${ARMS[@]}"; do
  marker="$OUT/ETTh1_720/DONE_${arm}.marker"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_720] $arm already done, skipping ----"
    continue
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 && ! -f "$INIT" ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_720] Stage-1: $arm ----"
  python -u scripts/train_set_normregret_control01.py \
    --reference_ckpt "$S1_720" --stage2_host "$S2_720" --arm "$arm" \
    --pred_len 720 --cell ETTh1_720 --init_seed 0 --loader_seed 0 \
    $init_flag \
    2>&1 | tee "logs/TRACK-A-SET-NORMREGRET-CONTROL01/stage1_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## TRACK-A-SET-NORMREGRET-CONTROL01 ETTh1_720: all 4 arms complete ##########"
