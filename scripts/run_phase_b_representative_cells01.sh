#!/usr/bin/env bash
# TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase B.
# 5 representative (cell, method) pairs x 2 arms = 10 RNG-corrected
# runs, using the new *_rngfixed.py trainers (Phase A1 fix applied).
# Same Base checkpoint / candidate pool / init_seed / loader_seed / 10
# fixed epochs as the ORIGINAL track's corresponding pair -- the ONLY
# intended difference from the original artifact is the RNG fix.
#
# Representative cells (spec section B1):
#   ETTh1_192  / V5     -- QA degraded in the original track
#   ETTh1_336  / Hard   -- QA improved a lot in the original track
#   Weather_720/ V5     -- QA's largest improvement anywhere
#   Weather_336/ Joint  -- Router + QA combined effect
#   Solar_96   / V5     -- Solar reproducibility
#
# GPU1 only, strictly sequential, resumable (skips if
# checkpoint_selection.json already exists). Must only be launched
# once the ORIGINAL TRACK-P100-SOFTTOPK-QUALITY-AWARE01 grid has
# finished (no parallel GPU jobs) -- this script does NOT wait/poll
# for that itself; the caller is responsible for only invoking it once
# GPU1 is free.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

echo "### nvidia-smi before starting (must show physical GPU 1, idle) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

RESULTS_ROOT="results/TRACK-P100-QA-CAUSAL-VALIDATION01/rng_corrected"
CKPT_ROOT="checkpoints/track_p100_qa_causal_validation01/rng_corrected"

REF_ETTH1_192="checkpoints/soft_set_mse/stage1/ETTh1/seq192_pred192/stage1_carts_softset_ETTh1_192_S0_wce_RelationStage1_ETTh1_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl192_pl192_0/checkpoint.pth"
REF_ETTH1_336="checkpoints/soft_set_mse/stage1/ETTh1/seq336_pred336/stage1_carts_softset_ETTh1_336_S0_wce_RelationStage1_ETTh1_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl336_pl336_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_336="checkpoints/soft_set_mse/stage1/custom/seq336_pred336/stage1_carts_softset_Weather_336_S0_wce_RelationStage1_custom_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl336_pl336_0/checkpoint.pth"
REF_SOLAR_96="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"

BASE_ETTH1_192="checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/base/checkpoint.pth"
BASE_ETTH1_336="checkpoints/track_v_multiquery_generalization01/ETTh1/H336/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_336="checkpoints/track_v_multiquery_generalization01/Weather/H336/seed0/base/checkpoint.pth"
BASE_SOLAR_96="checkpoints/track_p100_softtopk_quality_aware01/Solar_base/H96/base/checkpoint.pth"

POOL_ETTH1_192="results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H192/seed0/pool_top100/shared_candidate_pool"
POOL_ETTH1_336="results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H336/seed0/pool_top100/shared_candidate_pool"
POOL_WEATHER_720="results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/pool_top100/shared_candidate_pool"
POOL_WEATHER_336="results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H336/seed0/pool_top100/shared_candidate_pool"
POOL_SOLAR_96="results/TRACK-V-MULTIQUERY-GENERALIZATION01/Solar/H96/seed0/pool_top100/shared_candidate_pool"

run_v5 () {
  local cell="$1" ref="$2" base="$3" pool="$4" h="$5" arm="$6"
  local sel="${RESULTS_ROOT}/${cell}/V5/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then echo "---- [${cell}/V5/${arm}] already done ----"; return; fi
  echo "=== [${cell}/V5/${arm}] Phase B RNG-corrected ==="
  python -u scripts/train_softtopk_e2e_v02_rngfixed.py \
    --reference_ckpt "$ref" --num_query_views 5 --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

run_hard () {
  local cell="$1" ref="$2" base="$3" pool="$4" h="$5" arm="$6"
  local sel="${RESULTS_ROOT}/${cell}/Hard/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then echo "---- [${cell}/Hard/${arm}] already done ----"; return; fi
  echo "=== [${cell}/Hard/${arm}] Phase B RNG-corrected ==="
  python -u scripts/train_softtopk_e2e_hard02_rngfixed.py \
    --reference_ckpt "$ref" --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

run_joint () {
  local cell="$1" ref="$2" base="$3" pool="$4" h="$5" arm="$6"
  local sel="${RESULTS_ROOT}/${cell}/Joint/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then echo "---- [${cell}/Joint/${arm}] already done ----"; return; fi
  echo "=== [${cell}/Joint/${arm}] Phase B RNG-corrected ==="
  python -u scripts/train_softtopk_e2e_joint02_rngfixed.py \
    --reference_ckpt "$ref" --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

for arm in NOQA QA; do run_v5   "ETTh1_192"   "$REF_ETTH1_192"   "$BASE_ETTH1_192"   "$POOL_ETTH1_192"   192 "$arm"; done
for arm in NOQA QA; do run_hard "ETTh1_336"   "$REF_ETTH1_336"   "$BASE_ETTH1_336"   "$POOL_ETTH1_336"   336 "$arm"; done
for arm in NOQA QA; do run_v5   "Weather_720" "$REF_WEATHER_720" "$BASE_WEATHER_720" "$POOL_WEATHER_720" 720 "$arm"; done
for arm in NOQA QA; do run_joint "Weather_336" "$REF_WEATHER_336" "$BASE_WEATHER_336" "$POOL_WEATHER_336" 336 "$arm"; done
for arm in NOQA QA; do run_v5   "Solar_96"    "$REF_SOLAR_96"    "$BASE_SOLAR_96"    "$POOL_SOLAR_96"    96  "$arm"; done

echo "### Phase B representative-cell runs complete. Verifying RNG parity for each pair... ###"
python -u scripts/verify_rngfixed_pairs01.py --results_root "$RESULTS_ROOT" --cell ETTh1_192   --method V5
python -u scripts/verify_rngfixed_pairs01.py --results_root "$RESULTS_ROOT" --cell ETTh1_336   --method Hard
python -u scripts/verify_rngfixed_pairs01.py --results_root "$RESULTS_ROOT" --cell Weather_720 --method V5
python -u scripts/verify_rngfixed_pairs01.py --results_root "$RESULTS_ROOT" --cell Weather_336 --method Joint
python -u scripts/verify_rngfixed_pairs01.py --results_root "$RESULTS_ROOT" --cell Solar_96    --method V5
echo "### All Phase B pairs verified RNG-clean. ###"
