#!/usr/bin/env bash
# TRACK-A-SET-LOSS-CONTROL01 -- sequential H96 arm chain (one arm at a
# time on GPU1, alongside the pre-existing Weather_96 job). Each arm's
# completion is detected via its DONE_<arm>.marker file; already-present
# markers are SKIPPED, not re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
INIT="logs/TRACK-A-SET-LOSS-CONTROL01/shared_init_ETTh1_96.pth"
CELL_DIR="results/TRACK-A-SET-LOSS-CONTROL01/ETTh1_96"

ARMS=(A0_hard_choice A1_adaptive_multipos A2_srm A3_setutility_softce)

mkdir -p logs/TRACK-A-SET-LOSS-CONTROL01 "$CELL_DIR"

for name in "${ARMS[@]}"; do
  marker="$CELL_DIR/DONE_${name}.marker"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_96] $name already done, skipping ----"
    continue
  fi
  if [[ "$name" != "A0_hard_choice" ]]; then
    until [[ -f "$INIT" ]]; do sleep 5; done
  fi
  init_flag="--shared_init_in $INIT"
  [[ "$name" == "A0_hard_choice" ]] && init_flag="--shared_init_out $INIT"

  echo "---- [ETTh1_96] $name ----"
  python -u scripts/train_set_loss_control01.py \
    --reference_ckpt "$S1_96" --stage2_host "$S2_96" \
    --arm "$name" --pred_len 96 --cell ETTh1_96 \
    --checkpoints checkpoints/track_a_set_loss_control01 --out_dir results/TRACK-A-SET-LOSS-CONTROL01 \
    --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 --weight_decay 0.0 \
    --batch_size 32 --seed 1 --chunk_size 4096 --test_epochs 1,5,10 \
    $init_flag 2>&1 | tee "logs/TRACK-A-SET-LOSS-CONTROL01/stage1_ETTh1_96_${name}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $name did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## TRACK-A-SET-LOSS-CONTROL01 ETTh1_96: all 4 arms complete ##########"
