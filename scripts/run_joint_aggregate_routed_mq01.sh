#!/usr/bin/env bash
# TRACK-JOINT-AGGREGATE-ROUTED-MQ01 (Phase 3) -- 8 cells x 3 alpha
# values (0, 0.1, 0.5) = 24 joint-training runs, initialized from
# Phase 2's BEST_BACKBONE (Hard), Soft routing throughout. Encoder +
# SlotHeads + Router ALL trainable jointly (no freezing). Fixed 10
# epochs, every checkpoint saved, post-hoc selection via validation
# Soft-routed Aggregate MSE (computed via a real forward pass each
# epoch -- Phase 2's caching trick does not apply once the backbone
# updates). After each cell's best alpha is chosen, builds a real
# retrieval cache from that joint checkpoint and runs Stage-2
# (Original CARTS Trainable Global Lambda); Weather also gets the
# validation-only beta-grid CONTROL.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

P2_DECISION="results/TRACK-AGGREGATE-ROUTER-FROZEN01/phase2_decision.json"
P2_GO=$(python3 -c "import json; print(json.load(open('${P2_DECISION}'))['PHASE2_GO'])")
BACKBONE=$(python3 -c "import json; print(json.load(open('${P2_DECISION}'))['BEST_BACKBONE'])")
if [ "$P2_GO" != "True" ]; then
  echo "[ISSUE][ABORT] PHASE2_GO is not True -- Phase 3 must not run"
  exit 1
fi
echo "### Phase 2 decision confirmed: PHASE2_GO=${P2_GO} BEST_BACKBONE=${BACKBONE} ###"

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
TRACK_RESULT_ROOT="results/TRACK-JOINT-AGGREGATE-ROUTED-MQ01"
TRACK_CKPT_ROOT="checkpoints/track_joint_aggregate_routed_mq01"

process_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local cell="${ds}_${h}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local sel_json="${ALLH01_ROOT}/${ds}/H${h}/${cell}/${BACKBONE}/checkpoint_selection/selected_checkpoint.json"
  local init_ckpt tau_s
  init_ckpt=$(python3 -c "import json; print(json.load(open('${sel_json}'))['selected_checkpoint_path'])")
  tau_s=$(python3 -c "import json; print(json.load(open('${sel_json}'))['tau_s'])")
  local tau_r
  tau_r=$(python3 -c "import json; print(json.load(open('results/TRACK-AGGREGATE-ROUTER-FROZEN01/${ds}/H${h}/${BACKBONE}/R2_best_tau_selection.json'))['best_tau_r'])")
  local fixed_head_json="${PHASE1_ROOT}/${ds}/H${h}/${BACKBONE}/oracle_headroom.json"

  local best_alpha="" best_val="inf"
  for alpha in 0.0 0.1 0.5; do
    local out="${TRACK_RESULT_ROOT}"
    local ckpt_root="${TRACK_CKPT_ROOT}"
    local sel_out="${out}/${cell}/alpha${alpha}/checkpoint_selection.json"
    if [ ! -f "$sel_out" ]; then
      echo "=== [${cell}/alpha=${alpha}] joint training (10 epochs, Encoder+SlotHeads+Router) ==="
      python -u scripts/train_joint_aggregate_routed_mq01.py \
        --reference_ckpt "$ref" --cell "$cell" --pred_len "$h" --seq_len "$h" --tau_s "$tau_s" --tau_r "$tau_r" \
        --alpha "$alpha" --init_checkpoint "$init_ckpt" --fixed_head_json "$fixed_head_json" \
        --candidate_pool_cache "$pool" --out_dir "$out" --checkpoints "$ckpt_root"
    else
      echo "---- [${cell}/alpha=${alpha}] already trained, skipping ----"
    fi
    local this_val
    this_val=$(python3 -c "import json; print(json.load(open('${sel_out}'))['selected_val_agg_mse'])")
    local is_better
    is_better=$(python3 -c "print(1 if float('${this_val}') < float('${best_val}') else 0)")
    if [ "$is_better" = "1" ]; then
      best_val="$this_val"
      best_alpha="$alpha"
    fi
  done
  echo "---- [${cell}] best alpha=${best_alpha} val_agg_mse=${best_val} ----"
  python3 -c "
import json
json.dump({'best_alpha': ${best_alpha}, 'best_val_agg_mse': ${best_val}},
         open('${TRACK_RESULT_ROOT}/${cell}/best_alpha_selection.json', 'w'), indent=2)
"

  local joint_ckpt="${TRACK_CKPT_ROOT}/${cell}/alpha${best_alpha}"
  local best_ep
  best_ep=$(python3 -c "import json; print(json.load(open('${TRACK_RESULT_ROOT}/${cell}/alpha${best_alpha}/checkpoint_selection.json'))['selected_epoch'])")
  local joint_ckpt_path="${joint_ckpt}/checkpoint_epoch${best_ep}.pth"

  local eval_out="${TRACK_RESULT_ROOT}/${cell}/best_alpha_eval"
  if [ ! -f "${eval_out}/test_evaluation.json" ]; then
    echo "=== [${cell}] full test evaluation (best alpha=${best_alpha}) ==="
    python -u scripts/eval_joint_router_full01.py \
      --reference_ckpt "$ref" --cell "$cell" --pred_len "$h" --seq_len "$h" --tau_s "$tau_s" \
      --joint_checkpoint "$joint_ckpt_path" --fixed_head_json "$fixed_head_json" --candidate_pool_cache "$pool" \
      --out_dir "$eval_out"
  else
    echo "---- [${cell}] full test evaluation already done, skipping ----"
  fi

  local cache_out="${TRACK_RESULT_ROOT}/${cell}/best_alpha_cache"
  if [ ! -f "${cache_out}/test.pt" ]; then
    echo "=== [${cell}] build retrieval cache (best alpha=${best_alpha}) ==="
    python -u scripts/build_router_retrieval_cache01.py \
      --reference_ckpt "$ref" --h "$h" --retriever_checkpoint "$joint_ckpt_path" \
      --router_checkpoint "$joint_ckpt_path" --routing_mode soft --candidate_pool_cache "$pool" --tau_s "$tau_s" \
      --out_dir "$cache_out"
  else
    echo "---- [${cell}] cache already built, skipping ----"
  fi

  local stage2_out="${TRACK_RESULT_ROOT}/${cell}/best_alpha_stage2"
  if [ ! -f "${stage2_out}/metrics.json" ]; then
    echo "=== [${cell}] Stage-2: Original CARTS Trainable Global Lambda ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py \
      --reference_ckpt "$ref" --arm Hard --pred_len "$h" --seq_len "$h" --seed 0 \
      --base_checkpoint "$base" --cache_dir "$cache_out" --gpu_index 1 --out_dir "$stage2_out"
  else
    echo "---- [${cell}] Stage-2 already done, skipping ----"
  fi

  if [ "$ds" = "Weather" ]; then
    local beta_out="${TRACK_RESULT_ROOT}/${cell}/best_alpha_beta_control"
    if [ ! -f "${beta_out}/metrics.json" ]; then
      echo "=== [${cell}] Weather beta-control CONTROL diagnostic ==="
      python -u scripts/eval_professor_style_fusion01.py \
        --reference_ckpt "$ref" --arm Hard --pred_len "$h" --seq_len "$h" --seed 0 \
        --base_checkpoint "$base" --cache_dir "$cache_out" --candidate_support pool_top100 --gpu_index 1 \
        --out_dir "$beta_out"
    else
      echo "---- [${cell}] beta-control already done, skipping ----"
    fi
  fi
}

process_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
process_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
process_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
process_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
process_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
process_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
process_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
process_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-JOINT-AGGREGATE-ROUTED-MQ01 (8 cells x 3 alpha) complete ==="
