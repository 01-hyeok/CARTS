#!/usr/bin/env bash
# TRACK-V-SHARED-TOP100-SIGNFIX01 -- Stage-2 re-evaluation using the
# ORIGINAL CARTS trainable global-lambda gate (Y_final = B + lambda*(R-B),
# lambda=sigmoid(a), scripts/eval_r_stage2_lambda_signfix01.py which
# reuses train_r_stage2_lambda01.py's load_tensors/formula UNMODIFIED) --
# NOT the Professor-paper-style validation-only beta fusion.
#
# Reuses ONLY the already sign-fixed P100 Stage-1 caches under
# results/TRACK-V-SHARED-TOP100-SIGNFIX01/<ds>/H<h>/seed0/pool_top100/<cell>/<arm>/cache/
# -- aborts if a cache is missing rather than falling back to the old
# bugged caches. Writes to a SEPARATE
# .../stage2_lambda_original/ directory per cell/arm; never touches or
# overwrites the existing .../stage2_professor/ results.
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
  local result_root="results/TRACK-V-SHARED-TOP100-SIGNFIX01/${ds}/H${h}/seed0/pool_top100"
  for arm in V0 V1 V2 V5; do
    local arm_dir="${result_root}/${cell}/${arm}"
    local cache_dir="${arm_dir}/cache"
    if [ ! -f "${cache_dir}/test.pt" ]; then
      echo "[ISSUE][ABORT] missing sign-fixed P100 cache for ${cell}/${arm} at ${cache_dir}/test.pt -- NOT falling back to old bugged cache"
      exit 1
    fi
    local out_dir="${arm_dir}/stage2_lambda_original"
    if [ -f "${out_dir}/metrics.json" ]; then
      echo "---- [${cell}/${arm}] stage2_lambda_original already done, skipping ----"
      continue
    fi
    echo "=== [${cell}/P100/${arm}] Stage-2: Original CARTS Trainable Global Lambda (sign-fixed cache) ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py \
      --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "$cache_dir" --gpu_index 1 \
      --out_dir "$out_dir"
  done
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-V-SHARED-TOP100-SIGNFIX01 Stage-2 (Original CARTS lambda) complete for all available cells ==="
