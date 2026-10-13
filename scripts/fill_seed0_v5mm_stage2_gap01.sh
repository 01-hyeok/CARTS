#!/usr/bin/env bash
set -euo pipefail
cd /data/pjh_workspace/CARTS
export CUDA_VISIBLE_DEVICES=1
source /data/pjh_workspace/ts-env/bin/activate

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
OUT_ROOT="results/TRACK-JOINT-ROUTER-GENERALIZATION01/P100/seed0_v5mm_fill"
mkdir -p "$OUT_ROOT"

do_cell () {
  local ds="$1" h="$2"
  local ref_var="REF_${ds^^}_${h}"
  local base_var="BASE_${ds^^}_${h}"
  local ref="${!ref_var}"
  local base="${!base_var}"
  local cell="${ds}_${h}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local sel_json="${ALLH01_ROOT}/${ds}/H${h}/${cell}/V5/checkpoint_selection/selected_checkpoint.json"
  local ckpt tau_s
  ckpt=$(python3 -c "import json; print(json.load(open('${sel_json}'))['selected_checkpoint_path'])")
  tau_s=$(python3 -c "import json; print(json.load(open('${sel_json}'))['tau_s'])")
  local cell_root="${OUT_ROOT}/${cell}"
  mkdir -p "$cell_root"
  local out="${cell_root}/V5_mm_cache"
  if [ ! -f "${out}/test.pt" ]; then
    echo "=== [seed0_fill/${cell}/V5_mm] cache build ==="
    python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views 5 \
      --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" --candidate_pool_cache "$pool" \
      --retriever_checkpoint "$ckpt" --out_dir "$out"
  fi
  local s2out="${cell_root}/V5_mm_stage2"
  if [ ! -f "${s2out}/metrics.json" ]; then
    echo "=== [seed0_fill/${cell}/V5_mm] Stage-2 ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py \
      --reference_ckpt "$ref" --arm V5 --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "$out" --gpu_index 1 --out_dir "$s2out"
  fi
  echo "=== [seed0_fill/${cell}] test_mse: $(python3 -c "import json; print(json.load(open('${s2out}/metrics.json'))['test_mse'])") ==="
}

do_cell ETTh1 96
do_cell ETTh1 192
do_cell ETTh1 336
do_cell ETTh1 720
do_cell Weather 96
do_cell Weather 192
do_cell Weather 336
do_cell Weather 720

echo "=== seed0 V5_mm fill-in complete ==="
