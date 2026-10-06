#!/usr/bin/env bash
# TRACK-W-TIMESTAMP-FUSION01 -- sequential runner, GPU1 only, exactly one
# active child of THIS track at a time (never run two of its own steps
# in parallel -- see AUDIT.md PART 0). Never touches
# TRACK-W-CHECKPOINT-CRITERION-CORRECTION01's own files/checkpoints/GPU
# usage.
set -e
cd /data/pjh_workspace/CARTS
export CUDA_VISIBLE_DEVICES=1

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

run_stage1 () {
  local mode="$1" num_slots="$2" time_mode="$3" arm="$4" cell="$5" pred_len="$6" seq_len="$7" ref="$8"
  echo "=== TRACK-W-TIMESTAMP-FUSION01 Stage1: ${cell}/${arm} (mode=${mode} num_slots=${num_slots} time_mode=${time_mode}) ==="
  python -u scripts/train_w_timestamp_stage1_01.py \
    --reference_ckpt "${ref}" --mode "${mode}" --num_slots "${num_slots}" --time_mode "${time_mode}" \
    --arm_name "${arm}" --cell "${cell}" --pred_len "${pred_len}" --seq_len "${seq_len}"
}

# ---- Phase 1: C0/C1/C2, 4 cells x 3 time_modes = 12 runs ----
for CELL_SPEC in "ETTh1_96 96 96 ${REF_ETTH1_96}" "ETTh1_720 720 720 ${REF_ETTH1_720}" \
                "Weather_96 96 96 ${REF_WEATHER_96}" "Weather_720 720 720 ${REF_WEATHER_720}"; do
  read -r CELL PRED SEQ REF <<< "${CELL_SPEC}"
  run_stage1 zero_proj 1 no_time       C0 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
  run_stage1 zero_proj 1 real_time     C1 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
  run_stage1 zero_proj 1 shuffled_time C2 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
done

echo "=== TRACK-W-TIMESTAMP-FUSION01: Phase 1 (12 runs) complete ==="

# ---- Phase 2: T-V1/T-V2/T-V5, 4 cells x 3 num_slots = 12 new runs ----
# (T-V0 is reused from Phase 1's C1 checkpoint -- NOT retrained, per spec section 11)
for CELL_SPEC in "ETTh1_96 96 96 ${REF_ETTH1_96}" "ETTh1_720 720 720 ${REF_ETTH1_720}" \
                "Weather_96 96 96 ${REF_WEATHER_96}" "Weather_720 720 720 ${REF_WEATHER_720}"; do
  read -r CELL PRED SEQ REF <<< "${CELL_SPEC}"
  run_stage1 slots 1 real_time T-V1 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
  run_stage1 slots 2 real_time T-V2 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
  run_stage1 slots 5 real_time T-V5 "${CELL}" "${PRED}" "${SEQ}" "${REF}"
done

echo "=== TRACK-W-TIMESTAMP-FUSION01: Phase 2 Stage1 (12 new runs) complete ==="
echo "=== TRACK-W-TIMESTAMP-FUSION01: all Stage1 runs done -- cache/Stage2 steps run separately ==="
