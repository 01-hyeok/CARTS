#!/usr/bin/env bash
# TRACK-AGGREGATE-ROUTER-FROZEN01 -- builds the retrieval cache for the
# BEST variant decided in phase2_decision.json (BEST_BACKBONE=Hard,
# BEST_ROUTER_TARGET=R1, BEST_ROUTING_MODE=Soft) and runs Stage-2
# (Original CARTS Trainable Global Lambda, via
# eval_r_stage2_lambda_signfix01.py, reused unmodified) on all 8
# cells. Weather cells ALSO get the validation-only beta-grid CONTROL
# (diagnostic only, never primary).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_192="checkpoints/soft_set_mse/stage1/ETTh1/seq192_pred192/stage1_carts_softset_ETTh1_192_S0_wce_RelationStage1_ETTh1_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl192_pl192_0/checkpoint.pth"
REF_ETTH1_336="checkpoints/soft_set_mse/stage1/ETTh1/seq336_pred336/stage1_carts_softset_ETTh1_336_S0_wce_RelationStage1_ETTh1_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl336_pl336_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_192="checkpoints/soft_set_mse/stage1/custom/seq192_pred192/stage1_carts_softset_Weather_192_S0_wce_RelationStage1_custom_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl192_pl192_0/checkpoint.pth"
REF_WEATHER_336="checkpoints/soft_set_mse/stage1/custom/seq336_pred336/stage1_carts_softset_Weather_336_S0_wce_RelationStage1_custom_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl336_pl336_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_192="checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/base/checkpoint.pth"
BASE_ETTH1_336="checkpoints/track_v_multiquery_generalization01/ETTh1/H336/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_192="checkpoints/track_v_multiquery_generalization01/Weather/H192/seed0/base/checkpoint.pth"
BASE_WEATHER_336="checkpoints/track_v_multiquery_generalization01/Weather/H336/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"

ALLH01_ROOT="results/TRACK-HARD-EXPERT-V5-P100-ALLH01"
ROUTER_ROOT="results/TRACK-AGGREGATE-ROUTER-FROZEN01"
BACKBONE="Hard"

process_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local cell="${ds}_${h}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local sel_json="${ALLH01_ROOT}/${ds}/H${h}/${cell}/${BACKBONE}/checkpoint_selection/selected_checkpoint.json"
  local ckpt tau_s
  ckpt=$(python3 -c "import json; print(json.load(open('${sel_json}'))['selected_checkpoint_path'])")
  tau_s=$(python3 -c "import json; print(json.load(open('${sel_json}'))['tau_s'])")
  local r1_ckpt
  r1_ckpt=$(python3 -c "import json; print(json.load(open('${ROUTER_ROOT}/${ds}/H${h}/${BACKBONE}/R1/checkpoint_selection.json'))['selected_checkpoint_path'])")

  local out="${ROUTER_ROOT}/${ds}/H${h}/${BACKBONE}/best_variant_cache"
  if [ ! -f "${out}/test.pt" ]; then
    echo "=== [${cell}/${BACKBONE}/R1-Soft] build retrieval cache ==="
    python -u scripts/build_router_retrieval_cache01.py \
      --reference_ckpt "$ref" --h "$h" --retriever_checkpoint "$ckpt" --router_checkpoint "$r1_ckpt" \
      --routing_mode soft --candidate_pool_cache "$pool" --tau_s "$tau_s" --out_dir "$out"
  else
    echo "---- [${cell}] cache already built, skipping ----"
  fi

  local stage2_out="${ROUTER_ROOT}/${ds}/H${h}/${BACKBONE}/best_variant_stage2"
  if [ ! -f "${stage2_out}/metrics.json" ]; then
    echo "=== [${cell}/${BACKBONE}/R1-Soft] Stage-2: Original CARTS Trainable Global Lambda ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py \
      --reference_ckpt "$ref" --arm Hard --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "$out" --gpu_index 1 --out_dir "$stage2_out"
  else
    echo "---- [${cell}] Stage-2 already done, skipping ----"
  fi

  if [ "$ds" = "Weather" ]; then
    local beta_out="${ROUTER_ROOT}/${ds}/H${h}/${BACKBONE}/best_variant_beta_control"
    if [ ! -f "${beta_out}/metrics.json" ]; then
      echo "=== [${cell}/${BACKBONE}/R1-Soft] Weather beta-control CONTROL diagnostic ==="
      python -u scripts/eval_professor_style_fusion01.py \
        --reference_ckpt "$ref" --arm Hard --pred_len "$h" --seq_len "$h" --seed 0 \
        --base_checkpoint "$base" --cache_dir "$out" --candidate_support pool_top100 --gpu_index 1 \
        --out_dir "$beta_out"
    else
      echo "---- [${cell}] beta-control already done, skipping ----"
    fi
  fi
}

process_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
process_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
process_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
process_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
process_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
process_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
process_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
process_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-AGGREGATE-ROUTER-FROZEN01 best-variant Stage-2 (8 cells) complete ==="
