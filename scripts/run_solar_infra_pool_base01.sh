#!/usr/bin/env bash
# TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- builds, per Solar horizon
# (96/192/336/720), the two infra pieces every downstream method needs
# and that have NO dependency on trained retrieval weights (confirmed
# by reading `build_experiment`/`precompute_candidate_pool01.py`/
# `train_r_base_forecaster01.py`: all three only read the reference
# checkpoint's SAVED ARGS, never its weights):
#   1. Shared Top-100 (P100) candidate pool (future-blind, cosine-based)
#   2. Base Forecaster (per-channel-linear, trained fresh, frozen
#      everywhere downstream)
# Both reuse the EXISTING canonical scripts UNMODIFIED
# (`precompute_candidate_pool01.py`, `train_r_base_forecaster01.py`),
# run with the SAME protocol/hyperparameters as ETTh1/Weather's own.
#
# Requires each horizon's Solar canonical Stage-1 reference checkpoint
# to already exist (H96 pre-existing; H192/H336/H720 built by
# `run_solar_softset_host_stage1_longh01.sh`) -- this script does NOT
# build those itself, and aborts if missing.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

REF_ROOT="checkpoints/soft_set_mse/stage1/Solar"
POOL_OUT_ROOT="results/TRACK-V-MULTIQUERY-GENERALIZATION01/Solar"  # same convention ETTh1/Weather pools live under
BASE_OUT_ROOT="results/TRACK-P100-SOFTTOPK-QUALITY-AWARE01/Solar_base"
BASE_CKPT_ROOT="checkpoints/track_p100_softtopk_quality_aware01/Solar_base"

ref_ckpt_for () {
  local h="$1"
  echo "${REF_ROOT}/seq${h}_pred${h}/stage1_carts_softset_Solar_${h}_S0_wce_RelationStage1_Solar_ftM_sl${h}_ll0_pl${h}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl${h}_pl${h}_0/checkpoint.pth"
}

build_pool () {
  local h="$1" ref="$2"
  local out="${POOL_OUT_ROOT}/H${h}/seed0/pool_top100/shared_candidate_pool"
  if [ -f "${out}/val.pt" ]; then
    echo "---- Solar_${h} P100 pool already exists, skipping ----"
    return
  fi
  echo "=== Solar_${h} P100 pool precompute ==="
  python -u scripts/precompute_candidate_pool01.py \
    --reference_ckpt "$ref" --dataset Solar --seq_len "$h" --pred_len "$h" \
    --candidate_pool_size 100 --query_chunk_size 256 --out_dir "$out"
}

build_base () {
  local h="$1" ref="$2"
  local ckpt="${BASE_CKPT_ROOT}/H${h}/base/checkpoint.pth"
  if [ -f "$ckpt" ]; then
    echo "---- Solar_${h} base forecaster already exists, skipping ----"
    return
  fi
  echo "=== Solar_${h} base forecaster ==="
  python -u scripts/train_r_base_forecaster01.py \
    --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed 0 \
    --out_dir "${BASE_OUT_ROOT}/H${h}/base" --checkpoint_path "$ckpt"
}

for h in 96 192 336 720; do
  ref=$(ref_ckpt_for "$h")
  if [ ! -f "$ref" ]; then
    echo "[ISSUE][ABORT] Solar H${h} Stage-1 reference checkpoint not found: $ref"
    exit 1
  fi
  build_pool "$h" "$ref"
  build_base "$h" "$ref"
done
echo "=== Solar P100 pool + base forecaster infra complete (all 4 horizons) ==="
