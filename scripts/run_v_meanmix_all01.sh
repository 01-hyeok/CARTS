#!/usr/bin/env bash
# TRACK-V-MEANMIX-INFERENCE01 -- 32-cell execution (2 datasets x 4
# horizons x 4 arms). NEVER retrains Stage-1 -- loads the EXISTING
# canonical TRACK-V-MULTIQUERY-GENERALIZATION01 checkpoints read-only,
# rebuilds the retrieval cache with `mean_mixture_topk_selection`
# instead of `round_robin_topk_selection`, and reruns the EXISTING,
# UNMODIFIED `train_r_stage2_lambda01.py`. tau_s is read from each
# arm's own historical config.json -- never a new default.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  for arm in V0 V1 V2 V5; do
    if [ "$arm" = "V0" ]; then
      ckpt="checkpoints/track_v_multiquery_generalization01/${ds}/H${h}/seed0/V0/stage1/${ds}_${h}/checkpoint.pth"
      cfg="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/V0/stage1/${ds}_${h}/config.json"
    else
      ckpt="checkpoints/track_v_multiquery_generalization01/${ds}/H${h}/seed0/${arm}/stage1/checkpoint.pth"
      cfg="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/${arm}/stage1/config.json"
    fi
    tau_s=$(python3 -c "import json; print(json.load(open('${cfg}'))['tau_s'])")
    out="results/TRACK-V-MEANMIX-INFERENCE01/${ds}/H${h}/seed0/${arm}"
    echo "=== [${ds}_${h}/${arm}] MeanMix cache (tau_s=${tau_s}, read from ${cfg}) ==="
    python -u scripts/build_v_meanmix_cache01.py --arm "${arm}" --reference_ckpt "${ref}" \
      --retriever_checkpoint "${ckpt}" --pred_len "${h}" --seq_len "${h}" --seed 0 --tau_s "${tau_s}" \
      --out_dir "${out}"
    echo "=== [${ds}_${h}/${arm}] MeanMix Stage-2 (reused unmodified train_r_stage2_lambda01.py) ==="
    python -u scripts/train_r_stage2_lambda01.py --reference_ckpt "${ref}" --pred_len "${h}" --seq_len "${h}" --seed 0 \
      --base_checkpoint "${base}" --cache_dir "${out}" --out_dir "${out}/stage2"
  done
}

echo "### nvidia-smi before starting ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
run_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
run_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-V-MEANMIX-INFERENCE01 (32 cells) complete ==="
