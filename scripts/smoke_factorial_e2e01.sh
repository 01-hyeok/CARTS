#!/usr/bin/env bash
# TRACK-A-FACTORIAL-E2E01 smoke test: all 8 arms, ETTh1_96, 1 epoch,
# 3 batches per split. Writes ONLY under *_smoke paths so no real artifact
# can be created or overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/track_a_factorial_e2e_smoke
CKPT=checkpoints/track_a_factorial_e2e_smoke
LOGS=logs/track_a_factorial_e2e_smoke
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"

ARMS=(
  "individual_tf_cosine            individual tf       cosine"
  "individual_tf_asymmetric        individual tf       asymmetric"
  "individual_onpolicy_cosine      individual onpolicy cosine"
  "individual_onpolicy_asymmetric  individual onpolicy asymmetric"
  "set_tf_cosine                   greedy_set tf       cosine"
  "set_tf_asymmetric               greedy_set tf       asymmetric"
  "set_onpolicy_cosine             greedy_set onpolicy cosine"
  "set_onpolicy_asymmetric         greedy_set onpolicy asymmetric"
)

CELL=ETTh1_96
INIT="$LOGS/shared_encoder_init_${CELL}.pth"
BASEREF="$LOGS/y_base_reference_${CELL}.pt"
first=1
for spec in "${ARMS[@]}"; do
  read -r name target prefix scorer <<<"$spec"
  if [[ $first -eq 1 ]]; then init_flag="--shared_init_out $INIT"; ref_flag="--save_base_ref $BASEREF"
  else init_flag="--shared_init_in $INIT"; ref_flag="--base_ref $BASEREF"; fi

  echo "########## SMOKE Stage-1 $name ##########"
  python -u scripts/train_factorial_e2e01.py \
    --reference_ckpt "$S1" --stage2_host "$S2" \
    --target "$target" --prefix_policy "$prefix" --scorer "$scorer" \
    --pred_len 96 --cell "$CELL" --arm_name "$name" \
    --checkpoints "$CKPT" --out_dir "$OUT" \
    --train_epochs 1 --limit_batches 3 --test_epochs 1 \
    $init_flag 2>&1 | tee "$LOGS/stage1_${name}.log"

  echo "########## SMOKE Stage-2 $name ##########"
  python -u scripts/eval_factorial_e2e01_stage2.py \
    --arm_checkpoint "$CKPT/$CELL/$name/checkpoint.pth" \
    --stage2_host "$S2" --cell "$CELL" --arm_name "$name" \
    --out_dir "$OUT" --limit_batches 3 \
    $ref_flag 2>&1 | tee "$LOGS/stage2_${name}.log"
  first=0
done
echo "SMOKE COMPLETE"
