#!/usr/bin/env bash
# EXP-ORACLE-WCE-CONTROL01 -- ETTh1_720, 2 arms, one at a time, GPU1 only.
# Run only AFTER ETTh1_96's two arms are DONE with finite loss/gradient
# and 7-channel processing confirmed (spec section 12).
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

H96_A="results/EXP-ORACLE-WCE-CONTROL01/ETTh1_96/DONE_individual_onpolicy_cosine_wce.marker"
H96_B="results/EXP-ORACLE-WCE-CONTROL01/ETTh1_96/DONE_set_onpolicy_cosine_wce.marker"
if [[ ! -f "$H96_A" || ! -f "$H96_B" ]]; then
  echo "[ISSUE][ABORT] ETTh1_96 arms not both complete yet -- refusing to start H720." >&2
  exit 1
fi

S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
OUT="results/EXP-ORACLE-WCE-CONTROL01"
INIT="logs/EXP-ORACLE-WCE-CONTROL01/shared_init_ETTh1_720.pth"

mkdir -p logs/EXP-ORACLE-WCE-CONTROL01 "$OUT/ETTh1_720"

ARMS=(individual_onpolicy_cosine_wce set_onpolicy_cosine_wce)

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

  echo "---- [ETTh1_720] Stage-1 (WCE): $arm ----"
  python -u scripts/train_oracle_wce_control01.py \
    --reference_ckpt "$S1_720" --stage2_host "$S2_720" --arm "$arm" \
    --pred_len 720 --cell ETTh1_720 --init_seed 0 --loader_seed 0 \
    $init_flag \
    2>&1 | tee "logs/EXP-ORACLE-WCE-CONTROL01/stage1_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## EXP-ORACLE-WCE-CONTROL01 ETTh1_720: both arms complete ##########"
