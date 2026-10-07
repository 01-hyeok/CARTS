#!/usr/bin/env bash
# Driver: TRACK-HARD-EXPERT-V5-FULL01, ETTh1 ONLY (H96, H720), per
# explicit spec -- Weather is NOT in scope for this track. Sequential,
# GPU1. Reference/base checkpoint paths copied VERBATIM from
# scripts/run_expert_v5_full_all01.sh (the canonical Soft Expert
# driver) -- never guessed.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"

bash scripts/run_hard_expert_v5_full01.sh ETTh1_96  "${REF_ETTH1_96}"  96  96  "${BASE_ETTH1_96}"
bash scripts/run_hard_expert_v5_full01.sh ETTh1_720 "${REF_ETTH1_720}" 720 720 "${BASE_ETTH1_720}"

echo "=== TRACK-HARD-EXPERT-V5-FULL01 (ETTh1 H96/H720 only) complete ==="
