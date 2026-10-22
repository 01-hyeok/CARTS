#!/usr/bin/env bash
# TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 -- 4 cells (ETTh1/Weather x
# H96/H720) x 4 arms (V1/V5/PH1/PH5), seed0, Shared-P100, sequential,
# GPU1 only. MLP-V1/V5 canonical baselines are NOT retrained here --
# read-only reuse of TRACK-V-SHARED-TOP100-SIGNFIX01's own artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"

TRAIN_EPOCHS=10

run_arm () {
  local ds="$1" h="$2" ref="$3" base="$4" patch_len="$5" arm="$6"
  local cell="${ds}_${h}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local ckpt_root="checkpoints/track_transformer_patch_query_heads01/${cell}"
  local result_root="results/TRACK-TRANSFORMER-PATCH-QUERY-HEADS01/${cell}"
  local arm_ckpt_dir="${ckpt_root}/${arm}"
  local arm_result_dir="${result_root}/${arm}"
  mkdir -p "${arm_result_dir}"

  if [ ! -f "${pool}/val.pt" ]; then
    echo "[ISSUE][ABORT] shared Top-100 pool cache missing at ${pool}"
    exit 1
  fi

  echo "=== [${cell}/${arm}] Stage-1 train (fixed ${TRAIN_EPOCHS} epochs, no early stopping) ==="
  if [ ! -f "${arm_ckpt_dir}/checkpoint_epoch${TRAIN_EPOCHS}.pth" ]; then
    python -u scripts/train_transformer_patch_query_heads01.py \
      --reference_ckpt "$ref" --arm "$arm" --cell "$cell" --pred_len "$h" --seq_len "$h" \
      --patch_len "$patch_len" --candidate_pool_size 100 --candidate_pool_cache "$pool" \
      --train_epochs "$TRAIN_EPOCHS" \
      --out_dir "$result_root" --checkpoints "$ckpt_root"
  else
    echo "---- [${cell}/${arm}] Stage-1 already complete, skipping ----"
  fi

  echo "=== [${cell}/${arm}] post-hoc checkpoint selection ==="
  if [ ! -f "${arm_result_dir}/checkpoint_selection/selected_checkpoint.json" ]; then
    local epochs_csv
    epochs_csv=$(seq -s, 1 "$TRAIN_EPOCHS")
    python -u scripts/eval_transformer_patch_query_meanmix_selection01.py \
      --reference_ckpt "$ref" --arm "$arm" --ckpt_dir "$arm_ckpt_dir" --epochs "$epochs_csv" \
      --pred_len "$h" --seq_len "$h" --patch_len "$patch_len" --seed 0 \
      --candidate_pool_cache "$pool" --out_dir "${arm_result_dir}/checkpoint_selection"
  else
    echo "---- [${cell}/${arm}] checkpoint selection already done, skipping ----"
  fi
  local selected_ckpt
  selected_ckpt=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
  echo "---- [${cell}/${arm}] selected ckpt=${selected_ckpt} ----"

  echo "=== [${cell}/${arm}] cache build (Mean-Mixture for n_slots=5, ordinary Top-K for n_slots=1) ==="
  if [ ! -f "${arm_result_dir}/cache/test.pt" ]; then
    python -u scripts/build_transformer_patch_query_cache01.py \
      --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --patch_len "$patch_len" --seed 0 \
      --candidate_pool_cache "$pool" --retriever_checkpoint "$selected_ckpt" --out_dir "${arm_result_dir}/cache"
  else
    echo "---- [${cell}/${arm}] cache already built, skipping ----"
  fi

  echo "=== [${cell}/${arm}] Stage-2: Original CARTS Trainable Global Lambda ==="
  if [ ! -f "${arm_result_dir}/stage2_lambda_original/metrics.json" ]; then
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "${arm_result_dir}/cache" \
      --out_dir "${arm_result_dir}/stage2_lambda_original"
  else
    echo "---- [${cell}/${arm}] Stage-2 already done, skipping ----"
  fi
  echo "=== [${cell}/${arm}] complete ==="
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4" patch_len="$5"
  for arm in V1 V5 PH1 PH5; do
    run_arm "$ds" "$h" "$ref" "$base" "$patch_len" "$arm"
  done
}

run_cell ETTh1 96 "${REF_ETTH1_96}" "${BASE_ETTH1_96}" 16
run_cell ETTh1 720 "${REF_ETTH1_720}" "${BASE_ETTH1_720}" 120
run_cell Weather 96 "${REF_WEATHER_96}" "${BASE_WEATHER_96}" 16
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}" 120

echo "=== TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 (4 cells x 4 arms) complete ==="
