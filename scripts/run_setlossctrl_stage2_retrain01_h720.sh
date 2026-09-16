#!/usr/bin/env bash
# EXP-SET-LOSS-STAGE2-RETRAIN01 -- ETTh1_720, 4 loss arms (Hard CE / MultiPos /
# SRM / SoftCE), one at a time. Requires TRACK-A-SET-LOSS-CONTROL01's H96
# Stage-1 arms to already be complete (DONE_<arm>.marker present under
# results/TRACK-A-SET-LOSS-CONTROL01/ETTh1_720/) -- this script does NOT
# start, modify, or wait on that Stage-1 experiment; it only reads its
# already-finished outputs.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
STAGE1_OUT="results/TRACK-A-SET-LOSS-CONTROL01"
OUT="results/EXP-SET-LOSS-STAGE2-RETRAIN01"
INIT="logs/EXP-SET-LOSS-STAGE2-RETRAIN01/shared_stage2_init_ETTh1_720.pth"

mkdir -p logs/EXP-SET-LOSS-STAGE2-RETRAIN01 "$OUT/ETTh1_720"

# execution order per spec section 10 (Hard CE, SRM, SoftCE, then any
# additional arm actually present in the Stage-1 experiment -- MultiPos)
ARMS=(A0_hard_choice A2_srm A3_setutility_softce A1_adaptive_multipos)

first=1
for arm in "${ARMS[@]}"; do
  stage1_marker="$STAGE1_OUT/ETTh1_720/DONE_${arm}.marker"
  if [[ ! -f "$stage1_marker" ]]; then
    echo "---- [ETTh1_720] $arm Stage-1 not complete yet ($stage1_marker missing) -- skipping for now, not waiting ----"
    continue
  fi

  marker="$OUT/ETTh1_720/DONE_${arm}.marker"
  cache_dir="$OUT/cache/ETTh1_720/${arm}"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_720] $arm Stage-2 already done, skipping ----"
    continue
  fi

  if [[ ! -f "$cache_dir/train.pt" ]]; then
    echo "---- [ETTh1_720] building retrieval cache for $arm ----"
    python -u scripts/build_setlossctrl_retrieval_cache01.py \
      --cell ETTh1_720 --arm_name "$arm" --reference_ckpt "$S1_96" --stage2_host "$S2_96" \
      --pred_len 720 --top_k 10 --chunk_size 4096 --out_dir "$OUT" \
      2>&1 | tee "logs/EXP-SET-LOSS-STAGE2-RETRAIN01/cache_ETTh1_720_${arm}.log"
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_720] Stage-2 retrain: $arm ----"
  python -u scripts/train_setlossctrl_stage2_retrain01.py \
    --cell ETTh1_720 --arm_name "$arm" --stage2_host "$S2_96" --cache_dir "$cache_dir" \
    --checkpoints checkpoints/exp_set_loss_stage2_retrain01 --out_dir "$OUT" \
    --seed 1 --train_epochs 10 --patience 5 $init_flag \
    2>&1 | tee "logs/EXP-SET-LOSS-STAGE2-RETRAIN01/stage2_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## EXP-SET-LOSS-STAGE2-RETRAIN01 ETTh1_720: all available arms complete ##########"
