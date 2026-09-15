#!/usr/bin/env bash
# TRACK-A-WEATHER-SETSAFE01 -- SAFE-path orchestrator for TRACK-A-FACTORIAL-E2E01.
#
# Identical to scripts/run_factorial_e2e01.sh in every respect (same ARMS,
# same hyperparameters, same OUT/CKPT/LOGS paths, same Stage-1/Stage-2/gate
# sequence) with exactly ONE difference: every Stage-1 arm invocation adds
# `--oracle_compute_impl safe`.
#
# 'safe' resolves (per scripts/train_factorial_e2e01.py::resolve_oracle_compute_impl):
#   - Choice-CE (C): optimized (dead-code elimination, bit-exact vs reference)
#   - Individual Oracle (A): optimized_cache (reference formula, cached once
#     per query instead of recomputed every K step, bit-exact vs reference)
#   - Greedy Set Oracle (E): ALWAYS reference, unconditionally -- the OPT02/
#     OPT03 algebraic reformulation is NEVER used here (see
#     research/TRACK-A-WEATHER-OPT04.md: it does not preserve on-policy
#     training reproducibility).
#
# This is a SEPARATE FILE from run_factorial_e2e01.sh -- the original is
# never edited, so it is never at risk while any process might still be
# reading it. Use this script only for cells where the ORIGINAL
# orchestrator's own run has been confirmed fully stopped for that cell
# (never run both against the same cell/output directory concurrently).
#
#   scripts/run_factorial_e2e01_safe.sh <cell> [<cell> ...]

set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

OUT=results/track_a_factorial_e2e
CKPT=checkpoints/track_a_factorial_e2e
LOGS=logs/track_a_factorial_e2e
mkdir -p "$OUT" "$CKPT" "$LOGS"

S1_ETTh1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

S2_ETTh1_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_ETTh1_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_Weather_96="checkpoints/stage2/custom/seq96_pred96/stage2_carts_softset_s2_Weather_96_S0_wce_RelationStage2_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S2_Weather_720="checkpoints/stage2/custom/seq720_pred720/stage2_carts_softset_s2_Weather_720_S0_wce_RelationStage2_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

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

run_cell() {
  local cell="$1"
  local pred_len s1 s2
  case "$cell" in
    ETTh1_96)    pred_len=96;  s1="$S1_ETTh1_96";    s2="$S2_ETTh1_96" ;;
    ETTh1_720)   pred_len=720; s1="$S1_ETTh1_720";   s2="$S2_ETTh1_720" ;;
    Weather_96)  pred_len=96;  s1="$S1_Weather_96";  s2="$S2_Weather_96" ;;
    Weather_720) pred_len=720; s1="$S1_Weather_720"; s2="$S2_Weather_720" ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac
  local init="$LOGS/shared_encoder_init_${cell}.pth"
  local base_ref="$LOGS/y_base_reference_${cell}.pt"
  local indep="$OUT/$cell/independent_base_only.json"
  mkdir -p "$OUT/$cell"

  echo "########## cell $cell (pred_len=$pred_len) [SAFE oracle_compute_impl] ##########"

  # ---- STEP 0: Independent Base-only Forecaster = THE baseline ----
  # (no Oracle machinery involved -- --oracle_compute_impl not applicable)
  if [[ ! -f "$indep" ]]; then
    echo "---- [$cell] Independent Base-only Forecaster ----"
    python -u scripts/train_factorial_e2e01_base_only.py \
      --reference_ckpt "$s1" --stage2_host "$s2" \
      --pred_len "$pred_len" --cell "$cell" \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      2>&1 | tee "$LOGS/base_only_${cell}.log"
  else
    echo "---- [$cell] Independent Base-only already present, reusing $indep ----"
  fi
  [[ -f "$indep" ]] || { echo "[ABORT] $cell: Independent Base-only not produced" >&2; return 1; }

  local first=1
  for spec in "${ARMS[@]}"; do
    read -r name target prefix scorer <<<"$spec"
    local init_flag
    if [[ $first -eq 1 ]]; then init_flag="--shared_init_out $init"
    else init_flag="--shared_init_in $init"; fi

    echo "---- [$cell] Stage-1 $name [oracle_compute_impl=safe] ----"
    python -u scripts/train_factorial_e2e01.py \
      --reference_ckpt "$s1" --stage2_host "$s2" \
      --target "$target" --prefix_policy "$prefix" --scorer "$scorer" \
      --pred_len "$pred_len" --cell "$cell" --arm_name "$name" \
      --checkpoints "$CKPT" --out_dir "$OUT" \
      --top_k 10 --train_epochs 10 --patience 5 --learning_rate 0.001 \
      --weight_decay 0.0 --batch_size 32 --seed 0 --chunk_size 4096 \
      --test_epochs 1,5,10 --oracle_compute_impl safe \
      $init_flag 2>&1 | tee "$LOGS/stage1_${cell}_${name}.log"

    echo "---- [$cell] Stage-2 $name ----"
    local ref_flags
    if [[ $first -eq 1 ]]; then ref_flags="--save_base_ref $base_ref"
    else ref_flags="--base_ref $base_ref"; fi
    python -u scripts/eval_factorial_e2e01_stage2.py \
      --arm_checkpoint "$CKPT/$cell/$name/checkpoint.pth" \
      --stage2_host "$s2" --cell "$cell" --arm_name "$name" \
      --out_dir "$OUT" --top_k 10 --chunk_size 4096 \
      --independent_base_json "$indep" \
      $ref_flags 2>&1 | tee "$LOGS/stage2_${cell}_${name}.log"

    first=0
  done
  echo "---- [$cell] cell gate ----"
  python -u scripts/validate_factorial_e2e01_cell.py "$cell" "$OUT" \
    2>&1 | tee "$LOGS/gate_${cell}.log"
  echo "########## cell $cell complete and validated [SAFE] ##########"
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <cell> [<cell> ...]   (ETTh1_96 ETTh1_720 Weather_96 Weather_720)" >&2
  exit 2
fi
for cell in "$@"; do
  if ! run_cell "$cell"; then
    echo "[STOP] cell $cell failed its gate -- not proceeding to the remaining cells." >&2
    exit 1
  fi
done
python -u scripts/analyze_factorial_e2e01.py --root "$OUT" 2>&1 | tee "$LOGS/analyze_safe.log"
echo "all requested cells complete (SAFE): $*"
