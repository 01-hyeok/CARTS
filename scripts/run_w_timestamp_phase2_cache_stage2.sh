#!/usr/bin/env bash
# TRACK-W-TIMESTAMP-FUSION01 -- Phase 2 cache + Stage2, run AFTER
# run_w_timestamp_fusion01.sh's Phase1+Phase2 Stage1 queue finishes.
# Sequential, GPU1 only, one active child of this track at a time.
set -e
cd /data/pjh_workspace/CARTS
export CUDA_VISIBLE_DEVICES=1

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"

CKPT_ROOT="checkpoints/track_w_timestamp_fusion01"
RESULTS_ROOT="results/TRACK-W-TIMESTAMP-FUSION01"

cache_and_stage2 () {
  local mode="$1" num_slots="$2" arm="$3" cell="$4" pred_len="$5" seq_len="$6" ref="$7" base_ckpt="$8" ckpt_path="$9"
  echo "=== TRACK-W-TIMESTAMP-FUSION01 cache: ${cell}/${arm} ==="
  python -u scripts/build_w_timestamp_cache01.py \
    --reference_ckpt "${ref}" --mode "${mode}" --num_slots "${num_slots}" --arm_name "${arm}" --cell "${cell}" \
    --pred_len "${pred_len}" --seq_len "${seq_len}" \
    --retriever_checkpoint "${ckpt_path}" \
    --out_dir "${RESULTS_ROOT}/${cell}/${arm}/cache"
  echo "=== TRACK-W-TIMESTAMP-FUSION01 stage2: ${cell}/${arm} ==="
  python -u scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "${ref}" --pred_len "${pred_len}" --seq_len "${seq_len}" --seed 0 \
    --base_checkpoint "${base_ckpt}" \
    --cache_dir "${RESULTS_ROOT}/${cell}/${arm}/cache" \
    --out_dir "${RESULTS_ROOT}/${cell}/${arm}/stage2"
}

for CELL_SPEC in "ETTh1_96 96 96 ${REF_ETTH1_96} ${BASE_ETTH1_96}" \
                "ETTh1_720 720 720 ${REF_ETTH1_720} ${BASE_ETTH1_720}" \
                "Weather_96 96 96 ${REF_WEATHER_96} ${BASE_WEATHER_96}" \
                "Weather_720 720 720 ${REF_WEATHER_720} ${BASE_WEATHER_720}"; do
  read -r CELL PRED SEQ REF BASE <<< "${CELL_SPEC}"
  # T-V0 = Phase1's C1 checkpoint, reused verbatim (NOT retrained -- spec section 11)
  cache_and_stage2 zero_proj 1 T-V0 "${CELL}" "${PRED}" "${SEQ}" "${REF}" "${BASE}" \
    "${CKPT_ROOT}/${CELL}/C1/checkpoint.pth"
  cache_and_stage2 slots 1 T-V1 "${CELL}" "${PRED}" "${SEQ}" "${REF}" "${BASE}" \
    "${CKPT_ROOT}/${CELL}/T-V1/checkpoint.pth"
  cache_and_stage2 slots 2 T-V2 "${CELL}" "${PRED}" "${SEQ}" "${REF}" "${BASE}" \
    "${CKPT_ROOT}/${CELL}/T-V2/checkpoint.pth"
  cache_and_stage2 slots 5 T-V5 "${CELL}" "${PRED}" "${SEQ}" "${REF}" "${BASE}" \
    "${CKPT_ROOT}/${CELL}/T-V5/checkpoint.pth"
done

echo "=== TRACK-W-TIMESTAMP-FUSION01: Phase 2 cache+Stage2 (16 runs) complete ==="
