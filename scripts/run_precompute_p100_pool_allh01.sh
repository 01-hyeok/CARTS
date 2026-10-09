#!/usr/bin/env bash
# TRACK-HARD-EXPERT-V5-P100-ALLH01 -- generate the missing Shared-Top-100
# (P100) candidate pools for H192/H336 (ETTh1 and Weather), using the
# EXISTING UNMODIFIED precompute_candidate_pool01.py (bug-independent --
# never calls normalized_teacher_prob, confirmed during the
# TRACK-V-SHARED-TOP100-SIGNFIX01 audit). Writes to the SAME convention
# already used for H96/H720's pools
# (results/TRACK-V-MULTIQUERY-GENERALIZATION01/<ds>/H<h>/seed0/pool_top100/shared_candidate_pool)
# so every downstream script that expects that path works unmodified.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

REF_ETTH1_192="checkpoints/soft_set_mse/stage1/ETTh1/seq192_pred192/stage1_carts_softset_ETTh1_192_S0_wce_RelationStage1_ETTh1_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl192_pl192_0/checkpoint.pth"
REF_ETTH1_336="checkpoints/soft_set_mse/stage1/ETTh1/seq336_pred336/stage1_carts_softset_ETTh1_336_S0_wce_RelationStage1_ETTh1_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl336_pl336_0/checkpoint.pth"
REF_WEATHER_192="checkpoints/soft_set_mse/stage1/custom/seq192_pred192/stage1_carts_softset_Weather_192_S0_wce_RelationStage1_custom_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl192_pl192_0/checkpoint.pth"
REF_WEATHER_336="checkpoints/soft_set_mse/stage1/custom/seq336_pred336/stage1_carts_softset_Weather_336_S0_wce_RelationStage1_custom_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl336_pl336_0/checkpoint.pth"

gen_pool () {
  local ds="$1" h="$2" ref="$3"
  local out="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  if [ -f "${out}/val.pt" ]; then
    echo "---- ${ds}_H${h} pool already exists, skipping ----"
    return
  fi
  echo "=== generating ${ds}_H${h} Shared-Top-100 pool ==="
  python -u scripts/precompute_candidate_pool01.py \
    --reference_ckpt "$ref" --dataset "$ds" --seq_len "$h" --pred_len "$h" \
    --candidate_pool_size 100 --query_chunk_size 256 --out_dir "$out"
}

gen_pool ETTh1 192 "$REF_ETTH1_192"
gen_pool ETTh1 336 "$REF_ETTH1_336"
gen_pool Weather 192 "$REF_WEATHER_192"
gen_pool Weather 336 "$REF_WEATHER_336"

echo "=== all P100 pools ready ==="
