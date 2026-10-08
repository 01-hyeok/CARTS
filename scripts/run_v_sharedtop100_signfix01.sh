#!/usr/bin/env bash
# TRACK-V-SHARED-TOP100-SIGNFIX01 -- retrain all 16 Shared-Top-100 (P100)
# Stage-1 cells (ETTh1/Weather x H96/H720 x V0/V1/V2/V5) after fixing the
# teacher-sign double-negation bug in scripts/train_v_sharedtop100_01.py
# (normalized_teacher_prob(-d_pool,...) -> normalized_teacher_prob(d_pool,...)).
#
# - Reuses the EXISTING shared Top-100 candidate pool cache unmodified
#   (results/TRACK-V-MULTIQUERY-GENERALIZATION01/<ds>/H<h>/seed0/pool_top100/shared_candidate_pool)
#   -- precompute_candidate_pool01.py never calls normalized_teacher_prob,
#   confirmed independent of the bug, so the pool's candidate identity does
#   not need regeneration.
# - Fixed-epoch policy: --disable_early_stopping, train_epochs=10, every
#   epoch checkpoint saved (standing policy, research/RESEARCH_DECISIONS.md D-0015).
# - V0/V1: single-distribution selection == Mean-Mixture (proven elsewhere);
#   use the trainer's own checkpoint_best_retmse.pth directly, cache via the
#   existing build_retrieval_cache_pool01.py (round-robin, correct for S<=1).
# - V2/V5: post-hoc Mean-Mixture checkpoint re-selection across all 10 epoch
#   checkpoints (eval_v5_meanmix_checkpoint_selection_pool01.py, generic over
#   --num_query_views), then Mean-Mixture cache build
#   (build_v_meanmix_cache_pool01.py) -- NEVER Round-Robin, NEVER the test split.
# - Stage-2: Professor-paper-style Validation-Only Scalar Trust Fusion
#   (scripts/eval_professor_style_fusion01.py) ONLY -- train_r_stage2_lambda01.py
#   is NOT used for this track's final numbers.
# - Sequential, GPU1 only.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "### nvidia-smi before starting (must show physical GPU 1) ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"

TRAIN_EPOCHS=10

run_arm () {
  local ds="$1" h="$2" ref="$3" base="$4" s="$5"
  local cell="${ds}_${h}"
  local arm="V${s}"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local ckpt_root="checkpoints/track_v_sharedtop100_signfix01/${ds}/H${h}/seed0/pool_top100"
  local result_root="results/TRACK-V-SHARED-TOP100-SIGNFIX01/${ds}/H${h}/seed0/pool_top100"
  local arm_ckpt_dir="${ckpt_root}/${cell}/${arm}"
  local arm_result_dir="${result_root}/${cell}/${arm}"
  mkdir -p "${arm_result_dir}"

  if [ ! -f "${pool}/val.pt" ]; then
    echo "[ISSUE][ABORT] shared Top-100 pool cache missing at ${pool} -- must exist from the original (bug-independent) precompute step"
    exit 1
  fi

  local t0 t1 stage1_seconds stage2_seconds selection_seconds cache_seconds
  t0=$(date +%s)
  echo "=== [${cell}/P100/${arm}] (SIGNFIX01) Stage-1 retrain (sign-fixed, fixed-epoch) ==="
  if [ ! -f "${arm_ckpt_dir}/checkpoint_epoch${TRAIN_EPOCHS}.pth" ]; then
    python -u scripts/train_v_sharedtop100_01.py \
      --reference_ckpt "$ref" --num_query_views "$s" --cell "$cell" \
      --pred_len "$h" --seq_len "$h" --candidate_pool_size 100 \
      --candidate_pool_cache "$pool" --train_epochs "$TRAIN_EPOCHS" --disable_early_stopping \
      --out_dir "$result_root" --checkpoints "$ckpt_root"
  else
    echo "---- [${cell}/${arm}] Stage-1 already complete (epoch ${TRAIN_EPOCHS} checkpoint exists), skipping ----"
  fi
  t1=$(date +%s); stage1_seconds=$((t1 - t0))

  local tau_s
  tau_s=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/config.json'))['tau_s'])")

  t0=$(date +%s)
  local selected_ckpt selected_epoch
  if [ "$arm" = "V0" ] || [ "$arm" = "V1" ]; then
    echo "=== [${cell}/P100/${arm}] checkpoint selection: single-distribution == Mean-Mixture (proven), reuse checkpoint_best_retmse.pth ==="
    selected_ckpt="${arm_ckpt_dir}/checkpoint_best_retmse.pth"
    selected_epoch=$(python3 -c "import torch; print(torch.load('${selected_ckpt}', map_location='cpu')['epoch'])")
    mkdir -p "${arm_result_dir}/checkpoint_selection"
    python3 - "$arm_result_dir" "$selected_ckpt" "$selected_epoch" <<'PYEOF'
import json, sys
out_dir, ckpt, epoch = sys.argv[1:4]
(json.dump({
    'selection_criterion': 'min validation single-distribution RetMSE@10 (== Mean-Mixture for S<=1, proven elsewhere); test split NEVER used',
    'selected_epoch': int(epoch), 'selected_checkpoint_path': ckpt,
}, open(f'{out_dir}/checkpoint_selection/selected_checkpoint.json', 'w'), indent=2))
PYEOF
  else
    echo "=== [${cell}/P100/${arm}] checkpoint selection: post-hoc Mean-Mixture val RetMSE@10 across all ${TRAIN_EPOCHS} epochs (never Round-Robin, never test) ==="
    local epochs_csv
    epochs_csv=$(seq -s, 1 "$TRAIN_EPOCHS")
    python -u scripts/eval_v5_meanmix_checkpoint_selection_pool01.py \
      --reference_ckpt "$ref" --ckpt_dir "$arm_ckpt_dir" --epochs "$epochs_csv" \
      --num_query_views "$s" --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
      --candidate_pool_cache "$pool" --out_dir "${arm_result_dir}/checkpoint_selection"
    selected_ckpt=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")
    selected_epoch=$(python3 -c "import json; print(json.load(open('${arm_result_dir}/checkpoint_selection/selected_checkpoint.json'))['selected_epoch'])")
  fi
  t1=$(date +%s); selection_seconds=$((t1 - t0))
  echo "---- [${cell}/${arm}] selected epoch=${selected_epoch} ckpt=${selected_ckpt} ----"

  t0=$(date +%s)
  echo "=== [${cell}/P100/${arm}] cache build from selected (sign-fixed) checkpoint ==="
  if [ "$arm" = "V0" ] || [ "$arm" = "V1" ]; then
    python -u scripts/build_retrieval_cache_pool01.py \
      --reference_ckpt "$ref" --num_query_views "$s" --pred_len "$h" --seq_len "$h" --seed 0 \
      --candidate_pool_mode coarse_topk --candidate_pool_size 100 --candidate_pool_cache "$pool" \
      --retriever_checkpoint "$selected_ckpt" --out_dir "${arm_result_dir}/cache"
  else
    python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views "$s" \
      --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
      --candidate_pool_cache "$pool" --retriever_checkpoint "$selected_ckpt" --out_dir "${arm_result_dir}/cache"
  fi
  t1=$(date +%s); cache_seconds=$((t1 - t0))

  t0=$(date +%s)
  echo "=== [${cell}/P100/${arm}] Stage-2: Professor-style Validation-Only Scalar Trust Fusion ==="
  python -u scripts/eval_professor_style_fusion01.py \
    --reference_ckpt "$ref" --arm "$arm" --pred_len "$h" --seq_len "$h" --seed 0 \
    --base_checkpoint "$base" --cache_dir "${arm_result_dir}/cache" --candidate_support pool_top100 --gpu_index 1 \
    --out_dir "${arm_result_dir}/stage2_professor"
  t1=$(date +%s); stage2_seconds=$((t1 - t0))

  python3 - "$arm_result_dir" "$stage1_seconds" "$selection_seconds" "$cache_seconds" "$stage2_seconds" "$selected_epoch" <<'PYEOF'
import json, sys
out_dir, s1, sel, cache, s2, ep = sys.argv[1:7]
json.dump({
    'stage1_train_seconds': int(s1), 'checkpoint_selection_seconds': int(sel),
    'cache_build_seconds': int(cache), 'stage2_professor_seconds': int(s2),
    'selected_epoch': int(ep), 'disable_early_stopping': True, 'train_epochs': 10,
}, open(f'{out_dir}/resource_metrics.json', 'w'), indent=2)
PYEOF
  echo "=== [${cell}/P100/${arm}] SIGNFIX01 cell complete (stage1=${stage1_seconds}s selection=${selection_seconds}s cache=${cache_seconds}s stage2=${stage2_seconds}s) ==="
}

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  for s in 0 1 2 5; do
    run_arm "$ds" "$h" "$ref" "$base" "$s"
  done
}

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-V-SHARED-TOP100-SIGNFIX01 (16 cells) complete ==="
