#!/usr/bin/env bash
# TRACK-AGGREGATE-ROUTER-FROZEN01 (Phase 2) -- 8 cells x 2 backbones
# (V5, Hard) x 2 targets (R1 CE, R2 Soft w/ 4 tau values) = 80 router
# training runs, plus one full real-backbone evaluation pass per
# (cell, backbone). Encoder/SlotHeads/Retriever are FROZEN throughout
# (loaded read-only from TRACK-HARD-EXPERT-V5-P100-ALLH01's
# checkpoint_selection.json, uniform path for both V5 and Hard, all
# horizons). Router input granularity = CHANNEL_AWARE per
# TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01's phase1_decision.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

DECISION_JSON="results/TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01/phase1_decision.json"
MODE=$(python3 -c "import json; print(json.load(open('${DECISION_JSON}'))['NEXT_ROUTER_MODE'])")
GO=$(python3 -c "import json; print(json.load(open('${DECISION_JSON}'))['PHASE1_GO'])")
if [ "$GO" != "True" ]; then
  echo "[ISSUE][ABORT] PHASE1_GO is not True -- Phase 2 must not run"
  exit 1
fi
if [ "$MODE" != "CHANNEL_AWARE" ]; then
  echo "[ISSUE][ABORT] unexpected NEXT_ROUTER_MODE=${MODE} -- this orchestrator only implements CHANNEL_AWARE"
  exit 1
fi
echo "### Phase 1 decision confirmed: PHASE1_GO=${GO} NEXT_ROUTER_MODE=${MODE} ###"

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
PHASE1_ROOT="results/TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01"
TRACK_RESULT_ROOT="results/TRACK-AGGREGATE-ROUTER-FROZEN01"
TRACK_CKPT_ROOT="checkpoints/track_aggregate_router_frozen01"

process_cell_backbone () {
  local ds="$1" h="$2" ref="$3" base="$4" backbone="$5"
  local cell="${ds}_${h}"
  local src_dir="${ALLH01_ROOT}/${ds}/H${h}/${cell}/${backbone}"
  local sel_json="${src_dir}/checkpoint_selection/selected_checkpoint.json"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  if [ ! -f "$sel_json" ]; then
    echo "[ISSUE][ABORT] missing ${sel_json}"
    exit 1
  fi
  local ckpt tau_s
  ckpt=$(python3 -c "import json; print(json.load(open('${sel_json}'))['selected_checkpoint_path'])")
  tau_s=$(python3 -c "import json; print(json.load(open('${sel_json}'))['tau_s'])")

  local feat_dir="${TRACK_RESULT_ROOT}/${ds}/H${h}/${backbone}/features"
  if [ ! -f "${feat_dir}/train.pt" ]; then
    echo "=== [${cell}/${backbone}] precompute router features (z_qc, U_agg, U_ind) ==="
    python -u scripts/precompute_router_features01.py \
      --reference_ckpt "$ref" --ds "$ds" --h "$h" --backbone "$backbone" \
      --retriever_checkpoint "$ckpt" --candidate_pool_cache "$pool" --tau_s "$tau_s" \
      --out_dir "$feat_dir"
  else
    echo "---- [${cell}/${backbone}] features already cached, skipping precompute ----"
  fi

  # ---- R1: Aggregate Winner CE ----
  local r1_out="${TRACK_RESULT_ROOT}/${ds}/H${h}/${backbone}/R1"
  local r1_ckpt_dir="${TRACK_CKPT_ROOT}/${ds}/H${h}/${backbone}/R1"
  if [ ! -f "${r1_out}/checkpoint_selection.json" ]; then
    echo "=== [${cell}/${backbone}] train R1 (Aggregate Winner CE) ==="
    python -u scripts/train_aggregate_router01.py --features_dir "$feat_dir" --target R1 --seed 0 \
      --out_dir "$r1_out" --checkpoints "$r1_ckpt_dir"
  else
    echo "---- [${cell}/${backbone}] R1 already trained, skipping ----"
  fi

  # ---- R2: Aggregate Soft Teacher, tau sweep, val-selected ----
  local best_r2_val_agg="inf" best_r2_tau="" best_r2_ckpt=""
  for tau in 0.25 0.5 1.0 2.0; do
    local r2_out="${TRACK_RESULT_ROOT}/${ds}/H${h}/${backbone}/R2_tau${tau}"
    local r2_ckpt_dir="${TRACK_CKPT_ROOT}/${ds}/H${h}/${backbone}/R2_tau${tau}"
    if [ ! -f "${r2_out}/checkpoint_selection.json" ]; then
      echo "=== [${cell}/${backbone}] train R2 tau=${tau} (Aggregate Soft Teacher) ==="
      python -u scripts/train_aggregate_router01.py --features_dir "$feat_dir" --target R2 --tau_r "$tau" --seed 0 \
        --out_dir "$r2_out" --checkpoints "$r2_ckpt_dir"
    else
      echo "---- [${cell}/${backbone}] R2 tau=${tau} already trained, skipping ----"
    fi
    local this_val
    this_val=$(python3 -c "import json; print(json.load(open('${r2_out}/checkpoint_selection.json'))['selected_val_agg_mse'])")
    local is_better
    is_better=$(python3 -c "print(1 if float('${this_val}') < float('${best_r2_val_agg}') else 0)")
    if [ "$is_better" = "1" ]; then
      best_r2_val_agg="$this_val"
      best_r2_tau="$tau"
      best_r2_ckpt=$(python3 -c "import json; print(json.load(open('${r2_out}/checkpoint_selection.json'))['selected_checkpoint_path'])")
    fi
  done
  echo "---- [${cell}/${backbone}] best R2 tau=${best_r2_tau} val_agg_mse=${best_r2_val_agg} ----"
  python3 -c "
import json
json.dump({'best_tau_r': ${best_r2_tau}, 'best_val_agg_mse': ${best_r2_val_agg},
          'best_checkpoint_path': '${best_r2_ckpt}'},
         open('${TRACK_RESULT_ROOT}/${ds}/H${h}/${backbone}/R2_best_tau_selection.json', 'w'), indent=2)
"

  # ---- full real-backbone evaluation pass ----
  local eval_out="${TRACK_RESULT_ROOT}/${ds}/H${h}/${backbone}"
  if [ ! -f "${eval_out}/full_evaluation.json" ]; then
    local r1_ckpt r2_ckpt
    r1_ckpt=$(python3 -c "import json; print(json.load(open('${r1_out}/checkpoint_selection.json'))['selected_checkpoint_path'])")
    r2_ckpt="$best_r2_ckpt"
    echo "=== [${cell}/${backbone}] full real-backbone evaluation (all baselines + R1/R2 Hard/Soft) ==="
    python -u scripts/eval_aggregate_router_full01.py \
      --reference_ckpt "$ref" --ds "$ds" --h "$h" --backbone "$backbone" \
      --retriever_checkpoint "$ckpt" --base_checkpoint "$base" \
      --arm_stage2_metrics "${src_dir}/stage2/metrics.json" \
      --candidate_pool_cache "$pool" --tau_s "$tau_s" \
      --features_dir "$feat_dir" --oracle_headroom_json "${PHASE1_ROOT}/${ds}/H${h}/${backbone}/oracle_headroom.json" \
      --r1_checkpoint "$r1_ckpt" --r2_checkpoint "$r2_ckpt" \
      --out_dir "$eval_out"
  else
    echo "---- [${cell}/${backbone}] full evaluation already done, skipping ----"
  fi
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  process_cell_backbone "$ds" "$h" "$ref" "$base" V5
  process_cell_backbone "$ds" "$h" "$ref" "$base" Hard
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
run_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
run_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-AGGREGATE-ROUTER-FROZEN01 (8 cells x 2 backbones) complete ==="
