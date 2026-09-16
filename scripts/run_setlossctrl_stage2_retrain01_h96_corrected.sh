#!/usr/bin/env bash
# EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED -- ETTh1_96, 4 loss arms (Hard CE /
# MultiPos / SRM / SoftCE), one at a time. CORRECTED (delta-space) cache +
# trainer -- see scripts/build_setlossctrl_retrieval_cache01.py and
# scripts/train_setlossctrl_stage2_retrain01.py module docstrings.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
STAGE1_OUT="results/TRACK-A-SET-LOSS-CONTROL01"
OUT="results/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED"
INIT="logs/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/shared_stage2_init_ETTh1_96.pth"

mkdir -p logs/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED "$OUT/ETTh1_96"

ARMS=(A0_hard_choice A2_srm A3_setutility_softce A1_adaptive_multipos)

first=1
for arm in "${ARMS[@]}"; do
  stage1_marker="$STAGE1_OUT/ETTh1_96/DONE_${arm}.marker"
  if [[ ! -f "$stage1_marker" ]]; then
    echo "---- [ETTh1_96] $arm Stage-1 not complete yet ($stage1_marker missing) -- skipping for now, not waiting ----"
    continue
  fi

  marker="$OUT/ETTh1_96/DONE_${arm}.marker"
  cache_dir="$OUT/cache/ETTh1_96/${arm}"
  rm_json="$STAGE1_OUT/ETTh1_96/retrieval_metrics_${arm}.json"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_96] $arm Stage-2 already done (corrected), skipping ----"
    continue
  fi

  if [[ ! -f "$cache_dir/train.pt" ]]; then
    echo "---- [ETTh1_96] building CORRECTED (delta-space) retrieval cache for $arm ----"
    python -u scripts/build_setlossctrl_retrieval_cache01.py \
      --cell ETTh1_96 --arm_name "$arm" --reference_ckpt "$S1_96" --stage2_host "$S2_96" \
      --pred_len 96 --top_k 10 --chunk_size 4096 --out_dir "$OUT" \
      2>&1 | tee "logs/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/cache_ETTh1_96_${arm}.log"
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_96] Stage-2 retrain (CORRECTED): $arm ----"
  python -u scripts/train_setlossctrl_stage2_retrain01.py \
    --cell ETTh1_96 --arm_name "$arm" --stage2_host "$S2_96" --cache_dir "$cache_dir" \
    --stage1_retrieval_metrics_json "$rm_json" --fragg_tolerance 0.02 \
    --checkpoints checkpoints/exp_set_loss_stage2_retrain01_corrected --out_dir "$OUT" \
    --seed 1 --train_epochs 10 --patience 5 $init_flag \
    2>&1 | tee "logs/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/stage2_ETTh1_96_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED ETTh1_96: all available arms complete ##########"
