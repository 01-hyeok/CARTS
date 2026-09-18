#!/usr/bin/env bash
# TRACK-A-SET-NORMREGRET-CONTROL01 -- ETTh1_96, 4 arms, one at a time,
# GPU1 only. Does NOT touch or wait on any other GPU1 job.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
OUT="results/TRACK-A-SET-NORMREGRET-CONTROL01"
INIT="logs/TRACK-A-SET-NORMREGRET-CONTROL01/shared_init_ETTh1_96.pth"

mkdir -p logs/TRACK-A-SET-NORMREGRET-CONTROL01 "$OUT/ETTh1_96"

ARMS=(A0_hard_choice A1_raw_wce A2_normregret_softce A3_normregret_srm)

first=1
for arm in "${ARMS[@]}"; do
  marker="$OUT/ETTh1_96/DONE_${arm}.marker"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_96] $arm already done, skipping ----"
    continue
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 && ! -f "$INIT" ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_96] Stage-1: $arm ----"
  python -u scripts/train_set_normregret_control01.py \
    --reference_ckpt "$S1_96" --stage2_host "$S2_96" --arm "$arm" \
    --pred_len 96 --cell ETTh1_96 --init_seed 0 --loader_seed 0 \
    $init_flag \
    2>&1 | tee "logs/TRACK-A-SET-NORMREGRET-CONTROL01/stage1_ETTh1_96_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## TRACK-A-SET-NORMREGRET-CONTROL01 ETTh1_96: all 4 arms complete ##########"
