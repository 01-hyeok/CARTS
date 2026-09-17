#!/usr/bin/env bash
# EXP-ORACLE-WCE-CONTROL01 -- Stage-2 retraining, ETTh1_720, 2 WCE arms,
# one at a time, GPU1 only. Requires both H720 Stage-1 WCE arms to be
# DONE. Stage-2 artifacts under results/EXP-ORACLE-WCE-CONTROL01/stage2/
# (see run_oracle_wce_stage2_control01_h96.sh header for why).
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

STAGE1_OUT="results/EXP-ORACLE-WCE-CONTROL01"
S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
STAGE2_OUT="results/EXP-ORACLE-WCE-CONTROL01/stage2"
INIT="logs/EXP-ORACLE-WCE-CONTROL01/shared_stage2_init_ETTh1_720.pth"

mkdir -p logs/EXP-ORACLE-WCE-CONTROL01 "$STAGE2_OUT/ETTh1_720"

ARMS=(individual_onpolicy_cosine_wce set_onpolicy_cosine_wce)

first=1
for arm in "${ARMS[@]}"; do
  stage1_marker="$STAGE1_OUT/ETTh1_720/DONE_${arm}.marker"
  if [[ ! -f "$stage1_marker" ]]; then
    echo "---- [ETTh1_720] $arm Stage-1 not complete yet -- skipping for now ----"
    continue
  fi

  marker="$STAGE2_OUT/ETTh1_720/DONE_${arm}.marker"
  cache_dir="$STAGE2_OUT/cache/ETTh1_720/${arm}"
  rm_json="$STAGE1_OUT/ETTh1_720/retrieval_metrics_${arm}.json"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_720] $arm Stage-2 already done, skipping ----"
    continue
  fi

  if [[ ! -f "$cache_dir/train.pt" ]]; then
    echo "---- [ETTh1_720] building retrieval cache for $arm ----"
    python -u scripts/build_oracle_wce_retrieval_cache01.py \
      --cell ETTh1_720 --arm_name "$arm" --reference_ckpt "$S1_720" --stage2_host "$S2_720" \
      --pred_len 720 --top_k 10 --chunk_size 4096 \
      --factorial_out_dir "$STAGE1_OUT" --out_dir "$STAGE2_OUT" \
      2>&1 | tee "logs/EXP-ORACLE-WCE-CONTROL01/cache_ETTh1_720_${arm}.log"
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_720] Stage-2 retrain: $arm ----"
  python -u scripts/train_oracle_wce_stage2_control01.py \
    --cell ETTh1_720 --arm_name "$arm" --stage2_host "$S2_720" --cache_dir "$cache_dir" \
    --stage1_retrieval_metrics_json "$rm_json" --fragg_tolerance 0.02 \
    --checkpoints checkpoints/exp_oracle_wce_control01_stage2 --out_dir "$STAGE2_OUT" \
    --seed 0 --train_epochs 10 --patience 5 $init_flag \
    2>&1 | tee "logs/EXP-ORACLE-WCE-CONTROL01/stage2_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## EXP-ORACLE-WCE-CONTROL01 Stage2 ETTh1_720: all available arms complete ##########"
