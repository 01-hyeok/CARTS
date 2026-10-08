#!/usr/bin/env bash
# TRACK-V-PROFESSOR-FUSION01 -- Shared-Top-100 (P100), 16 cells (2
# datasets x 2 horizons x 4 methods). ONLY H96/H720 -- audited via
# filesystem before this script was written: P100 checkpoints/caches
# for H192/H336 do NOT exist for either dataset (never previously
# trained) and are NOT created here; they are simply out of scope.
# Same Professor-style evaluator as the Full-memory orchestrator,
# never train_r_stage2_lambda01.py. V2/V5 use the
# TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 P100 corrected caches.
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

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local cell="${ds}_${h}"
  local out_root="results/TRACK-V-PROFESSOR-FUSION01/${ds}/H${h}/seed0/pool_top100"
  for arm in V0 V1 V2 V5; do
    local cache_dir
    if [ "$arm" = "V0" ] || [ "$arm" = "V1" ]; then
      cache_dir="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/${cell}/${arm}/cache"
    elif [ "$arm" = "V2" ]; then
      cache_dir="results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/${ds}/H${h}/seed0/pool_top100/V2/cache"
    else
      cache_dir="results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/${ds}/H${h}/seed0/pool_top100/cache"
    fi
    if [ ! -f "${cache_dir}/test.pt" ]; then
      echo "[ISSUE][ABORT] missing P100 cache for ${cell}/${arm} at ${cache_dir}/test.pt"
      exit 1
    fi
    echo "=== [${cell}/P100/${arm}] Professor-style fusion eval (cache: ${cache_dir}) ==="
    python -u scripts/eval_professor_style_fusion01.py \
      --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "$cache_dir" --candidate_support pool_top100 --gpu_index 1 \
      --out_dir "${out_root}/${arm}"
  done
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-V-PROFESSOR-FUSION01 P100 (16 cells) complete ==="
