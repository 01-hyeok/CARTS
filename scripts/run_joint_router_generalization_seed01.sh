#!/usr/bin/env bash
# TRACK-JOINT-ROUTER-GENERALIZATION01 (Phase 4, P100 portion) -- runs
# the ENTIRE Phase1->Phase3 P100 chain for ONE given seed, across all
# 8 cells, reusing EVERY existing script unmodified (only
# --seed/--init_seed/--loader_seed are threaded through). Per cell:
#   1. V5 P100 Stage-1 (seed-varied) -> Mean-Mixture checkpoint
#      selection -> Mean-Mixture cache -> Stage-2
#   2. Hard P100 Stage-1 (seed-varied) -> Mean-Mixture checkpoint
#      selection -> Mean-Mixture cache -> Stage-2
#   3. Frozen Router (R1, Hard backbone, seed-varied; BEST_ROUTER_TARGET
#      per phase2_decision.json) -> Soft-routed cache -> Stage-2
#   4. Joint Router (alpha sweep 0/0.1/0.5, seed-varied, initialized
#      from THIS seed's own Hard P100 checkpoint) -> Soft-routed cache
#      -> Stage-2
#   5. Weather beta-control (diagnostic) for every cache built above
#
# Base Forecaster is intentionally NOT retrained per seed (fixed,
# shared across the seed ablation -- documented in REPORT.md, not
# hidden). P100 pool/architecture/alpha/tau are reused unmodified from
# Phase 1-3 (never re-tuned here, per spec).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${1:?usage: run_joint_router_generalization_seed01.sh <seed>}"

echo "### nvidia-smi before starting (must show physical GPU 1) ### seed=${SEED}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

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

ROOT="results/TRACK-JOINT-ROUTER-GENERALIZATION01/P100/seed${SEED}"
CKPT_ROOT="checkpoints/track_joint_router_generalization01/P100/seed${SEED}"
PHASE1_ROOT="results/TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01"  # fixed-head reference, reused across seeds

stage1_arm () {
  # args: ds h ref arm cell_root pool
  local ds="$1" h="$2" ref="$3" arm="$4" cell_root="$5" pool="$6" cell="${ds}_${h}"
  local arm_ckpt_dir="${CKPT_ROOT}/${cell}/${arm}"
  local arm_result_dir="${cell_root}/${arm}_stage1"
  if [ -f "${arm_result_dir}/checkpoint_selection/selected_checkpoint.json" ]; then
    echo "---- [seed${SEED}/${cell}/${arm}] Stage-1 already done ----"
    return
  fi
  echo "=== [seed${SEED}/${cell}/${arm}] P100 Stage-1 (fixed epochs, seed-varied) ==="
  local raw_dir="${cell_root}/${arm}_stage1_raw"
  local tau_s epochs_csv
  if [ "$arm" = "V5" ]; then
    python -u scripts/train_v_sharedtop100_01.py \
      --reference_ckpt "$ref" --num_query_views 5 --cell "$cell" --pred_len "$h" --seq_len "$h" \
      --candidate_pool_size 100 --candidate_pool_cache "$pool" --train_epochs 10 --disable_early_stopping \
      --init_seed "$SEED" --loader_seed "$SEED" --out_dir "$raw_dir" --checkpoints "$arm_ckpt_dir"
    tau_s=$(python3 -c "import json; print(json.load(open('${raw_dir}/${cell}/V5/config.json'))['tau_s'])")
    arm_ckpt_dir="${arm_ckpt_dir}/${cell}/V5"
    epochs_csv=$(seq -s, 1 10)
  else
    python -u scripts/train_hard_expert_v5_p100_allh01.py \
      --reference_ckpt "$ref" --cell "$cell" --pred_len "$h" --seq_len "$h" \
      --candidate_pool_cache "$pool" --train_epochs 10 --init_seed "$SEED" --loader_seed "$SEED" \
      --out_dir "$raw_dir" --checkpoints "$arm_ckpt_dir"
    tau_s=$(python3 -c "import json; print(json.load(open('${raw_dir}/${cell}/Hard/config.json'))['tau_s'])")
    arm_ckpt_dir="${arm_ckpt_dir}/${cell}/Hard"
    epochs_csv=$(seq -s, 0 10)
  fi
  mkdir -p "${arm_result_dir}/checkpoint_selection"
  echo "=== [seed${SEED}/${cell}/${arm}] Mean-Mixture checkpoint selection ==="
  python -u scripts/eval_v5_meanmix_checkpoint_selection_pool01.py \
    --reference_ckpt "$ref" --ckpt_dir "$arm_ckpt_dir" --epochs "$epochs_csv" \
    --num_query_views 5 --pred_len "$h" --seq_len "$h" --seed "$SEED" --tau_s "$tau_s" \
    --candidate_pool_cache "$pool" --out_dir "${arm_result_dir}/checkpoint_selection"
  echo "{\"tau_s\": ${tau_s}}" > "${arm_result_dir}/tau_s.json"
}

build_cache_and_stage2 () {
  # args: ds h ref base arm cache_source_ckpt tau_s pool cell_root out_name routing_script_extra
  local ds="$1" h="$2" ref="$3" base="$4" arm="$5" ckpt="$6" tau_s="$7" pool="$8" cell_root="$9" out_name="${10}"
  local out="${cell_root}/${out_name}_cache"
  if [ ! -f "${out}/test.pt" ]; then
    echo "=== [seed${SEED}/${ds}_${h}/${out_name}] Mean-Mixture cache build ==="
    python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views 5 \
      --pred_len "$h" --seq_len "$h" --seed "$SEED" --tau_s "$tau_s" --candidate_pool_cache "$pool" \
      --retriever_checkpoint "$ckpt" --out_dir "$out"
  fi
  local s2out="${cell_root}/${out_name}_stage2"
  if [ ! -f "${s2out}/metrics.json" ]; then
    echo "=== [seed${SEED}/${ds}_${h}/${out_name}] Stage-2 ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py \
      --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed "$SEED" \
      --base_checkpoint "$base" --cache_dir "$out" --gpu_index 1 --out_dir "$s2out"
  fi
  if [ "$ds" = "Weather" ]; then
    local betaout="${cell_root}/${out_name}_beta_control"
    if [ ! -f "${betaout}/metrics.json" ]; then
      echo "=== [seed${SEED}/${ds}_${h}/${out_name}] Weather beta-control ==="
      python -u scripts/eval_professor_style_fusion01.py \
        --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed "$SEED" \
        --base_checkpoint "$base" --cache_dir "$out" --candidate_support pool_top100 --gpu_index 1 \
        --out_dir "$betaout"
    fi
  fi
}

process_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local cell="${ds}_${h}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local cell_root="${ROOT}/${cell}"
  mkdir -p "$cell_root"

  stage1_arm "$ds" "$h" "$ref" V5 "$cell_root" "$pool"
  stage1_arm "$ds" "$h" "$ref" Hard "$cell_root" "$pool"

  local v5_ckpt v5_tau hard_ckpt hard_tau
  v5_ckpt=$(python3 -c "import json; print(json.load(open('${cell_root}/V5_stage1/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
  v5_tau=$(python3 -c "import json; print(json.load(open('${cell_root}/V5_stage1/tau_s.json'))['tau_s'])")
  hard_ckpt=$(python3 -c "import json; print(json.load(open('${cell_root}/Hard_stage1/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
  hard_tau=$(python3 -c "import json; print(json.load(open('${cell_root}/Hard_stage1/tau_s.json'))['tau_s'])")

  build_cache_and_stage2 "$ds" "$h" "$ref" "$base" V5 "$v5_ckpt" "$v5_tau" "$pool" "$cell_root" "V5_mm"
  build_cache_and_stage2 "$ds" "$h" "$ref" "$base" Hard "$hard_ckpt" "$hard_tau" "$pool" "$cell_root" "Hard_mm"

  # ---- Frozen Router (R1, Hard backbone) ----
  local feat_dir="${cell_root}/frozen_router_features"
  if [ ! -f "${feat_dir}/train.pt" ]; then
    echo "=== [seed${SEED}/${cell}] precompute frozen-router features ==="
    python -u scripts/precompute_router_features01.py \
      --reference_ckpt "$ref" --ds "$ds" --h "$h" --backbone Hard --retriever_checkpoint "$hard_ckpt" \
      --candidate_pool_cache "$pool" --tau_s "$hard_tau" --out_dir "$feat_dir"
  fi
  local r1_out="${cell_root}/frozen_router_R1"
  local r1_ckpt_dir="${CKPT_ROOT}/${cell}/frozen_router_R1"
  if [ ! -f "${r1_out}/checkpoint_selection.json" ]; then
    echo "=== [seed${SEED}/${cell}] train Frozen Router R1 ==="
    python -u scripts/train_aggregate_router01.py --features_dir "$feat_dir" --target R1 --seed "$SEED" \
      --out_dir "$r1_out" --checkpoints "$r1_ckpt_dir"
  fi
  local r1_ckpt
  r1_ckpt=$(python3 -c "import json; print(json.load(open('${r1_out}/checkpoint_selection.json'))['selected_checkpoint_path'])")
  local frozen_cache="${cell_root}/frozen_router_cache"
  if [ ! -f "${frozen_cache}/test.pt" ]; then
    echo "=== [seed${SEED}/${cell}] build Frozen Router (Soft) cache ==="
    python -u scripts/build_router_retrieval_cache01.py --reference_ckpt "$ref" --h "$h" \
      --retriever_checkpoint "$hard_ckpt" --router_checkpoint "$r1_ckpt" --routing_mode soft \
      --candidate_pool_cache "$pool" --tau_s "$hard_tau" --out_dir "$frozen_cache"
  fi
  local fr_s2="${cell_root}/frozen_router_stage2"
  if [ ! -f "${fr_s2}/metrics.json" ]; then
    echo "=== [seed${SEED}/${cell}] Frozen Router Stage-2 ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py --reference_ckpt "$ref" --arm Hard --pred_len "$h" \
      --seq_len "$h" --seed "$SEED" --base_checkpoint "$base" --cache_dir "$frozen_cache" --gpu_index 1 \
      --out_dir "$fr_s2"
  fi
  if [ "$ds" = "Weather" ]; then
    local fr_beta="${cell_root}/frozen_router_beta_control"
    if [ ! -f "${fr_beta}/metrics.json" ]; then
      python -u scripts/eval_professor_style_fusion01.py --reference_ckpt "$ref" --arm Hard --pred_len "$h" \
        --seq_len "$h" --seed "$SEED" --base_checkpoint "$base" --cache_dir "$frozen_cache" \
        --candidate_support pool_top100 --gpu_index 1 --out_dir "$fr_beta"
    fi
  fi

  # ---- Joint Router (alpha sweep, initialized from THIS seed's Hard checkpoint) ----
  local tau_r
  tau_r=$(python3 -c "import json; print(json.load(open('results/TRACK-AGGREGATE-ROUTER-FROZEN01/${ds}/H${h}/Hard/R2_best_tau_selection.json'))['best_tau_r'])")
  local fixed_head_json="${PHASE1_ROOT}/${ds}/H${h}/Hard/oracle_headroom.json"
  local best_alpha="" best_val="inf"
  for alpha in 0.0 0.1 0.5; do
    local jout="${cell_root}/joint_alpha${alpha}"
    local jckpt_root="${CKPT_ROOT}/${cell}/joint_alpha${alpha}"
    if [ ! -f "${jout}/checkpoint_selection.json" ]; then
      echo "=== [seed${SEED}/${cell}/joint alpha=${alpha}] joint training ==="
      python -u scripts/train_joint_aggregate_routed_mq01.py \
        --reference_ckpt "$ref" --cell "$cell" --pred_len "$h" --seq_len "$h" --tau_s "$hard_tau" \
        --tau_r "$tau_r" --alpha "$alpha" --init_checkpoint "$hard_ckpt" --fixed_head_json "$fixed_head_json" \
        --candidate_pool_cache "$pool" --init_seed "$SEED" --loader_seed "$SEED" \
        --out_dir "${ROOT}_joint_raw/${cell}" --checkpoints "$jckpt_root"
      mkdir -p "$jout"
      cp "${ROOT}_joint_raw/${cell}/${cell}/alpha${alpha}/checkpoint_selection.json" "${jout}/checkpoint_selection.json"
    fi
    local this_val
    this_val=$(python3 -c "import json; print(json.load(open('${jout}/checkpoint_selection.json'))['selected_val_agg_mse'])")
    local is_better
    is_better=$(python3 -c "print(1 if float('${this_val}') < float('${best_val}') else 0)")
    if [ "$is_better" = "1" ]; then
      best_val="$this_val"; best_alpha="$alpha"
    fi
  done
  python3 -c "import json; json.dump({'best_alpha': ${best_alpha}, 'best_val_agg_mse': ${best_val}}, open('${cell_root}/joint_best_alpha_selection.json','w'), indent=2)"
  local joint_ckpt_path
  joint_ckpt_path=$(python3 -c "import json; print(json.load(open('${cell_root}/joint_alpha${best_alpha}/checkpoint_selection.json'))['selected_checkpoint_path'])")
  local joint_cache="${cell_root}/joint_cache"
  if [ ! -f "${joint_cache}/test.pt" ]; then
    echo "=== [seed${SEED}/${cell}] build Joint Router (Soft) cache (best alpha=${best_alpha}) ==="
    python -u scripts/build_router_retrieval_cache01.py --reference_ckpt "$ref" --h "$h" \
      --retriever_checkpoint "$joint_ckpt_path" --router_checkpoint "$joint_ckpt_path" --routing_mode soft \
      --candidate_pool_cache "$pool" --tau_s "$hard_tau" --out_dir "$joint_cache"
  fi
  local joint_s2="${cell_root}/joint_stage2"
  if [ ! -f "${joint_s2}/metrics.json" ]; then
    echo "=== [seed${SEED}/${cell}] Joint Router Stage-2 ==="
    python -u scripts/eval_r_stage2_lambda_signfix01.py --reference_ckpt "$ref" --arm Hard --pred_len "$h" \
      --seq_len "$h" --seed "$SEED" --base_checkpoint "$base" --cache_dir "$joint_cache" --gpu_index 1 \
      --out_dir "$joint_s2"
  fi
  if [ "$ds" = "Weather" ]; then
    local joint_beta="${cell_root}/joint_beta_control"
    if [ ! -f "${joint_beta}/metrics.json" ]; then
      python -u scripts/eval_professor_style_fusion01.py --reference_ckpt "$ref" --arm Hard --pred_len "$h" \
        --seq_len "$h" --seed "$SEED" --base_checkpoint "$base" --cache_dir "$joint_cache" \
        --candidate_support pool_top100 --gpu_index 1 --out_dir "$joint_beta"
    fi
  fi
  echo "=== [seed${SEED}/${cell}] fully complete ==="
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  process_cell "$ds" "$h" "$ref" "$base"
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
run_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
run_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-JOINT-ROUTER-GENERALIZATION01 P100 seed${SEED} (8 cells) complete ==="
