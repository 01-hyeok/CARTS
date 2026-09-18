#!/usr/bin/env bash
# TRACK-A-SET-NORMREGRET-CONTROL01 -- Stage-2 retraining, ETTh1_96, 4 arms,
# one at a time, GPU1 only. Requires all 4 H96 Stage-1 arms to be DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

STAGE1_OUT="results/TRACK-A-SET-NORMREGRET-CONTROL01"
S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
STAGE2_OUT="results/TRACK-A-SET-NORMREGRET-CONTROL01/stage2"
INIT="logs/TRACK-A-SET-NORMREGRET-CONTROL01/shared_stage2_init_ETTh1_96.pth"

mkdir -p logs/TRACK-A-SET-NORMREGRET-CONTROL01 "$STAGE2_OUT/ETTh1_96"

ARMS=(A0_hard_choice A1_raw_wce A2_normregret_softce A3_normregret_srm)

first=1
for arm in "${ARMS[@]}"; do
  stage1_marker="$STAGE1_OUT/ETTh1_96/DONE_${arm}.marker"
  if [[ ! -f "$stage1_marker" ]]; then
    echo "---- [ETTh1_96] $arm Stage-1 not complete yet -- skipping for now ----"
    continue
  fi

  marker="$STAGE2_OUT/ETTh1_96/DONE_${arm}.marker"
  cache_dir="$STAGE2_OUT/cache/ETTh1_96/${arm}"
  rm_json="$STAGE1_OUT/ETTh1_96/retrieval_metrics_${arm}.json"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_96] $arm Stage-2 already done, skipping ----"
    continue
  fi

  if [[ ! -f "$cache_dir/train.pt" ]]; then
    echo "---- [ETTh1_96] building retrieval cache for $arm ----"
    python -u scripts/build_set_normregret_retrieval_cache01.py \
      --cell ETTh1_96 --arm_name "$arm" --reference_ckpt "$S1_96" --stage2_host "$S2_96" \
      --pred_len 96 --top_k 10 --chunk_size 4096 \
      --factorial_out_dir "$STAGE1_OUT" --out_dir "$STAGE2_OUT" \
      2>&1 | tee "logs/TRACK-A-SET-NORMREGRET-CONTROL01/cache_ETTh1_96_${arm}.log"
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_96] Stage-2 retrain: $arm ----"
  python -u scripts/train_set_normregret_stage2_control01.py \
    --cell ETTh1_96 --arm_name "$arm" --stage2_host "$S2_96" --cache_dir "$cache_dir" \
    --stage1_retrieval_metrics_json "$rm_json" --fragg_tolerance 0.02 \
    --checkpoints checkpoints/track_a_set_normregret_control01_stage2 --out_dir "$STAGE2_OUT" \
    --seed 0 --train_epochs 10 --patience 5 $init_flag \
    2>&1 | tee "logs/TRACK-A-SET-NORMREGRET-CONTROL01/stage2_ETTh1_96_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## TRACK-A-SET-NORMREGRET-CONTROL01 Stage2 ETTh1_96: all available arms complete ##########"
