#!/usr/bin/env bash
# TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- Phase C+D master orchestrator.
# For every one of the 12 cells (ETTh1/Weather/Solar x 96/192/336/720)
# and 6 methods (V0/V1/V2/V5/Hard/Joint), runs BOTH arms
# (SoftTopK-NOQA then SoftTopK-QA) back to back, paired on: same Base
# checkpoint, same candidate pool, same init_seed/loader_seed, same
# fixed 10 epochs. V0/V1/V2/V5 share one trainer
# (`train_softtopk_e2e_v01.py`, dispatched by --num_query_views); Hard
# and Joint each have their own trainer file (different retrieval loss
# / router structure, see those files' docstrings).
#
# GPU1 only, strictly sequential (every cell/method/arm one after the
# other -- no parallel GPU jobs, per standing rule). Resumable: each
# run is skipped if its `checkpoint_selection.json` already exists.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

RESULTS_ROOT="results/TRACK-P100-SOFTTOPK-QUALITY-AWARE01"
CKPT_ROOT="checkpoints/track_p100_softtopk_quality_aware01"

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_192="checkpoints/soft_set_mse/stage1/ETTh1/seq192_pred192/stage1_carts_softset_ETTh1_192_S0_wce_RelationStage1_ETTh1_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl192_pl192_0/checkpoint.pth"
REF_ETTH1_336="checkpoints/soft_set_mse/stage1/ETTh1/seq336_pred336/stage1_carts_softset_ETTh1_336_S0_wce_RelationStage1_ETTh1_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl336_pl336_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_192="checkpoints/soft_set_mse/stage1/custom/seq192_pred192/stage1_carts_softset_Weather_192_S0_wce_RelationStage1_custom_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl192_pl192_0/checkpoint.pth"
REF_WEATHER_336="checkpoints/soft_set_mse/stage1/custom/seq336_pred336/stage1_carts_softset_Weather_336_S0_wce_RelationStage1_custom_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl336_pl336_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
REF_SOLAR_96="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"
REF_SOLAR_192="checkpoints/soft_set_mse/stage1/Solar/seq192_pred192/stage1_carts_softset_Solar_192_S0_wce_RelationStage1_Solar_ftM_sl192_ll0_pl192_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl192_pl192_0/checkpoint.pth"
REF_SOLAR_336="checkpoints/soft_set_mse/stage1/Solar/seq336_pred336/stage1_carts_softset_Solar_336_S0_wce_RelationStage1_Solar_ftM_sl336_ll0_pl336_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl336_pl336_0/checkpoint.pth"
REF_SOLAR_720="checkpoints/soft_set_mse/stage1/Solar/seq720_pred720/stage1_carts_softset_Solar_720_S0_wce_RelationStage1_Solar_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_192="checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/base/checkpoint.pth"
BASE_ETTH1_336="checkpoints/track_v_multiquery_generalization01/ETTh1/H336/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_192="checkpoints/track_v_multiquery_generalization01/Weather/H192/seed0/base/checkpoint.pth"
BASE_WEATHER_336="checkpoints/track_v_multiquery_generalization01/Weather/H336/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"
BASE_SOLAR_96="checkpoints/track_p100_softtopk_quality_aware01/Solar_base/H96/base/checkpoint.pth"
BASE_SOLAR_192="checkpoints/track_p100_softtopk_quality_aware01/Solar_base/H192/base/checkpoint.pth"
BASE_SOLAR_336="checkpoints/track_p100_softtopk_quality_aware01/Solar_base/H336/base/checkpoint.pth"
BASE_SOLAR_720="checkpoints/track_p100_softtopk_quality_aware01/Solar_base/H720/base/checkpoint.pth"

pool_for () {
  local ds="$1" h="$2"
  if [ "$ds" = "Solar" ]; then
    echo "results/TRACK-V-MULTIQUERY-GENERALIZATION01/Solar/H${h}/seed0/pool_top100/shared_candidate_pool"
  else
    echo "results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  fi
}

run_v_family () {
  local ds="$1" h="$2" ref="$3" base="$4" pool="$5" nqv="$6" arm="$7"
  local cell="${ds}_${h}"
  local sel="${RESULTS_ROOT}/${cell}/V${nqv}/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then
    echo "---- [${cell}/V${nqv}/${arm}] already done ----"; return
  fi
  echo "=== [${cell}/V${nqv}/${arm}] SoftTopK E2E ==="
  python -u scripts/train_softtopk_e2e_v01.py \
    --reference_ckpt "$ref" --num_query_views "$nqv" --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

run_hard () {
  local ds="$1" h="$2" ref="$3" base="$4" pool="$5" arm="$6"
  local cell="${ds}_${h}"
  local sel="${RESULTS_ROOT}/${cell}/Hard/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then
    echo "---- [${cell}/Hard/${arm}] already done ----"; return
  fi
  echo "=== [${cell}/Hard/${arm}] SoftTopK E2E ==="
  python -u scripts/train_softtopk_e2e_hard01.py \
    --reference_ckpt "$ref" --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

run_joint () {
  local ds="$1" h="$2" ref="$3" base="$4" pool="$5" arm="$6"
  local cell="${ds}_${h}"
  local sel="${RESULTS_ROOT}/${cell}/Joint/${arm}/checkpoint_selection.json"
  if [ -f "$sel" ]; then
    echo "---- [${cell}/Joint/${arm}] already done ----"; return
  fi
  echo "=== [${cell}/Joint/${arm}] SoftTopK E2E ==="
  python -u scripts/train_softtopk_e2e_joint01.py \
    --reference_ckpt "$ref" --arm "$arm" --cell "$cell" \
    --pred_len "$h" --seq_len "$h" --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --out_dir "$RESULTS_ROOT" --checkpoints "$CKPT_ROOT"
}

process_cell () {
  local ds="$1" h="$2"
  local ref_var="REF_${ds^^}_${h}" base_var="BASE_${ds^^}_${h}"
  local ref="${!ref_var}" base="${!base_var}"
  local pool=$(pool_for "$ds" "$h")
  if [ ! -f "$ref" ]; then echo "[ISSUE][ABORT] missing reference ckpt: $ref"; exit 1; fi
  if [ ! -f "$base" ]; then echo "[ISSUE][ABORT] missing base ckpt: $base"; exit 1; fi
  if [ ! -f "${pool}/train.pt" ]; then echo "[ISSUE][ABORT] missing P100 pool: $pool"; exit 1; fi

  for nqv in 0 1 2 5; do
    run_v_family "$ds" "$h" "$ref" "$base" "$pool" "$nqv" NOQA
    run_v_family "$ds" "$h" "$ref" "$base" "$pool" "$nqv" QA
  done
  run_hard "$ds" "$h" "$ref" "$base" "$pool" NOQA
  run_hard "$ds" "$h" "$ref" "$base" "$pool" QA
  run_joint "$ds" "$h" "$ref" "$base" "$pool" NOQA
  run_joint "$ds" "$h" "$ref" "$base" "$pool" QA
  echo "=== [${ds}_${h}] all 6 methods x 2 arms complete ==="
}

for ds in ETTh1 Weather Solar; do
  for h in 96 192 336 720; do
    process_cell "$ds" "$h"
  done
done
echo "=== TRACK-P100-SOFTTOPK-QUALITY-AWARE01 Phase C+D (12 cells x 6 methods x 2 arms) complete ==="
