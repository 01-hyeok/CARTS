#!/usr/bin/env bash
# TRACK-HARD-EXPERT-V5-P100-ALLH01 -- V5 Original / Soft Expert / Hard
# Expert comparison under Shared-Top-100 (P100) candidate support, all
# 8 dataset/horizon cells (ETTh1/Weather x H96/H192/H336/H720).
#
# - V5 for H96/H720: REUSED from TRACK-V-SHARED-TOP100-SIGNFIX01 (same
#   sign-fixed trainer, same pool, same Mean-Mixture selection -- no
#   retrain, only the specialization/diagnostic report is computed
#   fresh since SIGNFIX01 itself never ran the per-head report).
# - V5 for H192/H336: fresh train via the SAME sign-fixed
#   train_v_sharedtop100_01.py (num_query_views=5).
# - Soft/Hard, all 8 cells: fresh train via the new
#   train_expert_v5_p100_allh01.py / train_hard_expert_v5_p100_allh01.py.
# - ALL arms: checkpoint selection = validation Mean-Mixture RetMSE@10
#   (eval_v5_meanmix_checkpoint_selection_pool01.py, reused unmodified),
#   cache = Mean-Mixture (build_v_meanmix_cache_pool01.py, reused
#   unmodified), Stage-2 = Original CARTS Trainable Global Lambda
#   (eval_r_stage2_lambda_signfix01.py -- NEVER the Professor-style beta
#   fusion), specialization report = run_specialization_report_pool01.py.
# - Fixed 10 epochs, no early stopping, every epoch checkpoint saved.
# - GPU1 only, strictly sequential (one cell/arm at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
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

TRACK_RESULT_ROOT="results/TRACK-HARD-EXPERT-V5-P100-ALLH01"
TRACK_CKPT_ROOT="checkpoints/track_hard_expert_v5_p100_allh01"

run_v5 () {
  local ds="$1" h="$2" ref="$3" pool="$4"
  local cell="${ds}_${h}"
  local arm_result_dir="${TRACK_RESULT_ROOT}/${ds}/H${h}/${cell}/V5"
  local arm_ckpt_dir="${TRACK_CKPT_ROOT}/${ds}/H${h}/${cell}/V5"
  mkdir -p "$arm_result_dir"
  local t0 t1 stage1_s=0 sel_s=0 cache_s=0

  if [ "$h" = "96" ] || [ "$h" = "720" ]; then
    local src_dir="results/TRACK-V-SHARED-TOP100-SIGNFIX01/${ds}/H${h}/seed0/pool_top100/${cell}/V5"
    if [ ! -f "${src_dir}/cache/test.pt" ]; then
      echo "[ISSUE][ABORT] expected reusable sign-fixed V5 P100 cache at ${src_dir}/cache missing"
      exit 1
    fi
    echo "=== [${cell}/V5] REUSING sign-fixed TRACK-V-SHARED-TOP100-SIGNFIX01 checkpoint/cache (no retrain) ==="
    mkdir -p "${arm_result_dir}/cache" "${arm_result_dir}/checkpoint_selection"
    cp "${src_dir}/cache/"*.pt "${arm_result_dir}/cache/" 2>/dev/null || true
    cp "${src_dir}/cache/stage1_metrics.json" "${arm_result_dir}/cache/"
    cp "${src_dir}/checkpoint_selection/selected_checkpoint.json" "${arm_result_dir}/checkpoint_selection/"
    python3 -c "
import json
sel = json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))
sel['provenance'] = 'REUSED from TRACK-V-SHARED-TOP100-SIGNFIX01 (identical sign-fixed trainer, identical pool, identical Mean-Mixture selection) -- no retrain for H96/H720'
json.dump(sel, open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json', 'w'), indent=2)
"
  else
    local epochs_csv
    epochs_csv=$(seq -s, 1 10)
    t0=$(date +%s)
    echo "=== [${cell}/V5] Stage-1 (sign-fixed trainer, fresh train for H${h}) ==="
    python -u scripts/train_v_sharedtop100_01.py \
      --reference_ckpt "$ref" --num_query_views 5 --cell "$cell" \
      --pred_len "$h" --seq_len "$h" --candidate_pool_size 100 \
      --candidate_pool_cache "$pool" --train_epochs 10 --disable_early_stopping \
      --out_dir "${TRACK_RESULT_ROOT}/${ds}/H${h}/stage1_tmp" --checkpoints "$arm_ckpt_dir"
    t1=$(date +%s); stage1_s=$((t1 - t0))

    t0=$(date +%s)
    local tau_s
    tau_s=$(python3 -c "import json; print(json.load(open('${TRACK_RESULT_ROOT}/${ds}/H${h}/stage1_tmp/${cell}/V5/config.json'))['tau_s'])")
    echo "=== [${cell}/V5] checkpoint selection (Mean-Mixture val RetMSE@10) ==="
    python -u scripts/eval_v5_meanmix_checkpoint_selection_pool01.py \
      --reference_ckpt "$ref" --ckpt_dir "${arm_ckpt_dir}/${cell}/V5" --epochs "$epochs_csv" \
      --num_query_views 5 --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
      --candidate_pool_cache "$pool" --out_dir "${arm_result_dir}/checkpoint_selection"
    t1=$(date +%s); sel_s=$((t1 - t0))

    local selected_ckpt
    selected_ckpt=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
    t0=$(date +%s)
    echo "=== [${cell}/V5] Mean-Mixture cache build ==="
    python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views 5 \
      --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
      --candidate_pool_cache "$pool" --retriever_checkpoint "$selected_ckpt" --out_dir "${arm_result_dir}/cache"
    t1=$(date +%s); cache_s=$((t1 - t0))
  fi
  python3 -c "
import json
json.dump({'stage1_train_seconds': $stage1_s, 'checkpoint_selection_seconds': $sel_s, 'cache_build_seconds': $cache_s},
          open('${arm_result_dir}/resource_metrics.json', 'w'), indent=2)
"
}

run_soft_or_hard () {
  local arm="$1" ds="$2" h="$3" ref="$4" pool="$5"
  local cell="${ds}_${h}"
  local arm_result_dir="${TRACK_RESULT_ROOT}/${ds}/H${h}/${cell}/${arm}"
  local arm_ckpt_dir="${TRACK_CKPT_ROOT}/${ds}/H${h}/${cell}/${arm}"
  mkdir -p "$arm_result_dir"
  local script_name epochs_csv="1,2,3,4,5,6,7,8,9,10"
  if [ "$arm" = "Soft" ]; then
    script_name="train_expert_v5_p100_allh01.py"
  else
    script_name="train_hard_expert_v5_p100_allh01.py"
    epochs_csv="0,1,2,3,4,5,6,7,8,9,10"
  fi

  local t0 t1 stage1_s sel_s cache_s
  t0=$(date +%s)
  echo "=== [${cell}/${arm}] Stage-1 (fixed 10 epochs, P100) ==="
  python -u "scripts/${script_name}" \
    --reference_ckpt "$ref" --cell "$cell" --pred_len "$h" --seq_len "$h" \
    --candidate_pool_cache "$pool" --train_epochs 10 \
    --out_dir "${TRACK_RESULT_ROOT}/${ds}/H${h}/stage1_tmp" --checkpoints "$arm_ckpt_dir"
  t1=$(date +%s); stage1_s=$((t1 - t0))

  t0=$(date +%s)
  local tau_s
  tau_s=$(python3 -c "import json; print(json.load(open('${TRACK_RESULT_ROOT}/${ds}/H${h}/stage1_tmp/${cell}/${arm}/config.json'))['tau_s'])")
  echo "=== [${cell}/${arm}] checkpoint selection (Mean-Mixture val RetMSE@10, never Round-Robin) ==="
  python -u scripts/eval_v5_meanmix_checkpoint_selection_pool01.py \
    --reference_ckpt "$ref" --ckpt_dir "${arm_ckpt_dir}/${cell}/${arm}" --epochs "$epochs_csv" \
    --num_query_views 5 --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
    --candidate_pool_cache "$pool" --out_dir "${arm_result_dir}/checkpoint_selection"
  t1=$(date +%s); sel_s=$((t1 - t0))

  local selected_ckpt best_epoch
  selected_ckpt=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
  best_epoch=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_epoch'])")

  t0=$(date +%s)
  echo "=== [${cell}/${arm}] Mean-Mixture cache build ==="
  python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views 5 \
    --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
    --candidate_pool_cache "$pool" --retriever_checkpoint "$selected_ckpt" --out_dir "${arm_result_dir}/cache"
  t1=$(date +%s); cache_s=$((t1 - t0))

  python3 -c "
import json
json.dump({'stage1_train_seconds': $stage1_s, 'checkpoint_selection_seconds': $sel_s, 'cache_build_seconds': $cache_s,
          'selected_epoch': $best_epoch}, open('${arm_result_dir}/resource_metrics.json', 'w'), indent=2)
"
}

run_stage2_and_report () {
  local ds="$1" h="$2" arm="$3" ref="$4" base="$5" pool="$6"
  local cell="${ds}_${h}"
  local arm_result_dir="${TRACK_RESULT_ROOT}/${ds}/H${h}/${cell}/${arm}"
  local selected_ckpt best_epoch
  selected_ckpt=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
  best_epoch=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_epoch'])")

  local t0 t1 stage2_s report_s
  t0=$(date +%s)
  echo "=== [${cell}/${arm}] Stage-2: Original CARTS Trainable Global Lambda ==="
  python -u scripts/eval_r_stage2_lambda_signfix01.py \
    --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed 0 \
    --base_checkpoint "$base" --cache_dir "${arm_result_dir}/cache" --gpu_index 1 \
    --out_dir "${arm_result_dir}/stage2"
  t1=$(date +%s); stage2_s=$((t1 - t0))

  t0=$(date +%s)
  echo "=== [${cell}/${arm}] specialization/per-head report (P100) ==="
  python -u scripts/run_specialization_report_pool01.py \
    --reference_ckpt "$ref" --retriever_checkpoint "$selected_ckpt" --pred_len "$h" --seq_len "$h" --seed 0 \
    --candidate_pool_cache "$pool" --best_epoch "$best_epoch" --out_dir "$arm_result_dir"
  t1=$(date +%s); report_s=$((t1 - t0))

  python3 -c "
import json
p = '${arm_result_dir}/resource_metrics.json'
d = json.load(open(p))
d['stage2_seconds'] = $stage2_s
d['specialization_report_seconds'] = $report_s
d['total_wall_clock_seconds'] = sum(d.get(k, 0) for k in ('stage1_train_seconds','checkpoint_selection_seconds','cache_build_seconds','stage2_seconds','specialization_report_seconds'))
json.dump(d, open(p, 'w'), indent=2)
"
  echo "=== [${cell}/${arm}] cell/arm complete ==="
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  if [ ! -f "${pool}/val.pt" ]; then
    echo "[ISSUE][ABORT] P100 pool missing for ${ds}_${h} at ${pool}"
    exit 1
  fi
  run_v5 "$ds" "$h" "$ref" "$pool"
  run_stage2_and_report "$ds" "$h" V5 "$ref" "$base" "$pool"
  run_soft_or_hard Soft "$ds" "$h" "$ref" "$pool"
  run_stage2_and_report "$ds" "$h" Soft "$ref" "$base" "$pool"
  run_soft_or_hard Hard "$ds" "$h" "$ref" "$pool"
  run_stage2_and_report "$ds" "$h" Hard "$ref" "$base" "$pool"
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 192   "${REF_ETTH1_192}"   "${BASE_ETTH1_192}"
run_cell ETTh1 336   "${REF_ETTH1_336}"   "${BASE_ETTH1_336}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 192 "${REF_WEATHER_192}" "${BASE_WEATHER_192}"
run_cell Weather 336 "${REF_WEATHER_336}" "${BASE_WEATHER_336}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-HARD-EXPERT-V5-P100-ALLH01 (8 cells x 3 arms) complete ==="
