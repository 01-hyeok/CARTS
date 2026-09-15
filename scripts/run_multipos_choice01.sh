#!/usr/bin/env bash
# TRACK-A-MULTIPOS-CHOICE01 orchestrator. Oracle=greedy_set, Prefix=onpolicy,
# Score=cosine fixed. Supervision varies: A0 one-hot Choice CE vs A1
# Adaptive Multi-positive Choice CE (alpha=0.50, pre-registered -- see
# scripts/train_multipos_choice01.py's module docstring).
#
# A2 (fixed top-3) is SKIPPED per spec S6's explicit permission ("자원이
# 부족하면 A2는 생략 가능합니다. A0/A1이 우선입니다.") -- GPU 1 is already
# running TRACK-A-FACTORIAL-E2E01's Weather cells in parallel.
#
#   scripts/run_multipos_choice01.sh <cell> [<cell> ...]
#   cell in: ETTh1_96 ETTh1_720

set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/TRACK-A-MULTIPOS-CHOICE01
CKPT=checkpoints/track_a_multipos_choice01
LOGS=logs/TRACK-A-MULTIPOS-CHOICE01
FACTORIAL_OUT=results/track_a_factorial_e2e
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1_ETTh1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_ETTh1_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_ETTh1_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"

ARMS=(A0_choice_ce A1_multipos)

run_cell() {
  local cell="$1"
  local pred_len s1 s2
  case "$cell" in
    ETTh1_96)  pred_len=96;  s1="$S1_ETTh1_96";  s2="$S2_ETTh1_96" ;;
    ETTh1_720) pred_len=720; s1="$S1_ETTh1_720"; s2="$S2_ETTh1_720" ;;
    *) echo "unsupported cell: $cell" >&2; return 2 ;;
  esac
  local init="$LOGS/shared_encoder_init_${cell}.pth"
  local base_ref="$LOGS/y_base_reference_${cell}.pt"
  local indep="$FACTORIAL_OUT/$cell/independent_base_only.json"
  mkdir -p "$OUT/$cell"

  [[ -f "$indep" ]] || { echo "[ABORT] $cell: Independent Base-only not found at $indep" >&2; return 1; }
  echo "########## cell $cell (pred_len=$pred_len) -- baseline $indep (read-only) ##########"

  local first=1
  for arm in "${ARMS[@]}"; do
    local init_flag
    if [[ $first -eq 1 ]]; then init_flag="--shared_init_out $init"
    else init_flag="--shared_init_in $init"; fi

    echo "---- [$cell] Stage-1 $arm ----"
    python -u scripts/train_multipos_choice01.py \
      --reference_ckpt "$s1" --stage2_host "$s2" \
      --arm "$arm" --pred_len "$pred_len" --cell "$cell" \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 \
      --weight_decay 0.0 --batch_size 32 --seed 0 --test_epochs 1,5,10 \
      $init_flag 2>&1 | tee "$LOGS/stage1_${cell}_${arm}.log"

    echo "---- [$cell] Stage-2 $arm ----"
    local ref_flags
    if [[ $first -eq 1 ]]; then ref_flags="--save_base_ref $base_ref"
    else ref_flags="--base_ref $base_ref"; fi
    python -u scripts/eval_multipos_choice01_stage2.py \
      --arm_checkpoint "$CKPT/$cell/$arm/checkpoint.pth" \
      --stage2_host "$s2" --cell "$cell" --arm_name "$arm" \
      --out_dir "$OUT" --top_k 10 \
      --independent_base_json "$indep" \
      $ref_flags 2>&1 | tee "$LOGS/stage2_${cell}_${arm}.log"

    first=0
  done
  echo "########## cell $cell complete ##########"
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <cell> [<cell> ...]   (ETTh1_96 ETTh1_720)" >&2
  exit 2
fi
for cell in "$@"; do
  run_cell "$cell"
done
echo "all requested cells complete: $*"
