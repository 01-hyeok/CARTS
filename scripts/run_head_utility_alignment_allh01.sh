#!/usr/bin/env bash
# TRACK-HEAD-UTILITY-ALIGNMENT01 -- read-only diagnostic over all 24
# existing TRACK-HARD-EXPERT-V5-P100-ALLH01 settings (8 cells x 3 arms:
# V5/Soft/Hard). No retraining; checkpoints and P100 pools are never
# modified. Resolves each arm's selected checkpoint / tau_s directly
# from that arm's own checkpoint_selection.json (works transparently
# for V5's H96/H720 cells, which were themselves REUSED from
# TRACK-V-SHARED-TOP100-SIGNFIX01 rather than retrained).
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

TRACK_RESULT_ROOT="results/TRACK-HEAD-UTILITY-ALIGNMENT01"
SRC_ROOT="results/TRACK-HARD-EXPERT-V5-P100-ALLH01"

run_arm () {
  local ds="$1" h="$2" ref="$3" base="$4" arm="$5"
  local cell="${ds}_${h}"
  local src_dir="${SRC_ROOT}/${ds}/H${h}/${cell}/${arm}"
  local out_dir="${TRACK_RESULT_ROOT}/${ds}/H${h}/${arm}"
  if [ -f "${out_dir}/resource_metrics.json" ]; then
    echo "---- [${cell}/${arm}] already done, skipping ----"
    return
  fi
  mkdir -p "$out_dir"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local sel_json="${src_dir}/checkpoint_selection/selected_checkpoint.json"
  if [ ! -f "$sel_json" ]; then
    echo "[ISSUE][ABORT] missing ${sel_json}"
    exit 1
  fi
  local ckpt tau_s
  ckpt=$(python3 -c "import json; print(json.load(open('${sel_json}'))['selected_checkpoint_path'])")
  tau_s=$(python3 -c "import json; print(json.load(open('${sel_json}'))['tau_s'])")
  if [ ! -f "$ckpt" ]; then
    echo "[ISSUE][ABORT] selected checkpoint missing on disk: ${ckpt}"
    exit 1
  fi
  echo "=== [${cell}/${arm}] Head-Utility-Alignment diagnostic (read-only) ==="
  python -u scripts/run_head_utility_alignment01.py \
    --reference_ckpt "$ref" --ds "$ds" --h "$h" --arm "$arm" \
    --retriever_checkpoint "$ckpt" --base_checkpoint "$base" \
    --arm_cache_dir "${src_dir}/cache" --arm_stage2_metrics "${src_dir}/stage2/metrics.json" \
    --candidate_pool_cache "$pool" --tau_s "$tau_s" --gpu_index 1 \
    --out_dir "$out_dir"
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  for arm in V5 Soft Hard; do
    run_arm "$ds" "$h" "$ref" "$base" "$arm"
  done
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
run_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
run_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-HEAD-UTILITY-ALIGNMENT01 (8 cells x 3 arms) complete ==="
