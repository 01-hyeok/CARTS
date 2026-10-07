#!/usr/bin/env bash
# TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 -- V2 Shared-Top-100 (P100)
# scope expansion, mirroring the V5 P100 orchestrator exactly. All 4
# P100 V2 cells already have epoch1-6 checkpoints (filesystem-audited,
# early-stopped at patience=5 like V5's P100 cells) -- no retrain.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

REF_ETTH1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
REF_ETTH1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
REF_WEATHER_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
REF_WEATHER_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

BASE_ETTH1_96="checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth"
BASE_ETTH1_720="checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth"
BASE_WEATHER_96="checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth"
BASE_WEATHER_720="checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/checkpoint.pth"

NUM_SLOTS=2

run_cell () {
  local ds="$1" h="$2" ref="$3" base="$4"
  local cell="${ds}_${h}"
  local hist_ckpt_dir="checkpoints/track_v_multiquery_generalization01/${ds}/H${h}/seed0/pool_top100/${cell}/V2"
  local pool="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/shared_candidate_pool"
  local cfg="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${ds}/H${h}/seed0/pool_top100/${cell}/V2/config.json"
  local tau_s
  tau_s=$(python3 -c "import json; print(json.load(open('${cfg}'))['tau_s'])")
  local out="results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/${ds}/H${h}/seed0/pool_top100/V2"
  mkdir -p "$out"

  local epochs="1,2,3,4,5,6"

  echo "=== [${cell}/P100/V2] checkpoint selection (Mean-Mixture val RetMSE@10 within Top-100, tau_s=${tau_s}) ==="
  python -u scripts/eval_v5_meanmix_checkpoint_selection_pool01.py \
    --reference_ckpt "$ref" --ckpt_dir "$hist_ckpt_dir" --epochs "$epochs" \
    --num_query_views "$NUM_SLOTS" --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
    --candidate_pool_cache "$pool" --out_dir "${out}/checkpoint_selection"

  local selected_ckpt
  selected_ckpt=$(python3 -c "import json; print(json.load(open('${out}/checkpoint_selection/selected_checkpoint.json'))['selected_checkpoint_path'])")

  echo "=== [${cell}/P100/V2] Mean-Mixture cache within Top-100 pool (selected checkpoint: ${selected_ckpt}) ==="
  python -u scripts/build_v_meanmix_cache_pool01.py --reference_ckpt "$ref" --num_query_views "$NUM_SLOTS" \
    --pred_len "$h" --seq_len "$h" --seed 0 --tau_s "$tau_s" \
    --candidate_pool_cache "$pool" --retriever_checkpoint "$selected_ckpt" --out_dir "${out}/cache"

  echo "=== [${cell}/P100/V2] Stage2 (reused unmodified train_r_stage2_lambda01.py) ==="
  python -u scripts/train_r_stage2_lambda01.py --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed 0 \
    --base_checkpoint "$base" --cache_dir "${out}/cache" --out_dir "${out}/stage2"

  python3 - "$ds" "$h" "$out" <<'PYEOF'
import json
import sys
ds, h, out = sys.argv[1:4]
sel = json.load(open(f'{out}/checkpoint_selection/checkpoint_selection.json'))
stage1 = json.load(open(f'{out}/cache/stage1_metrics.json'))
stage2 = json.load(open(f'{out}/stage2/metrics.json'))
summary = {
    'dataset': ds, 'horizon': int(h), 'arm': 'V2', 'candidate_support': 'pool_top100',
    'old_rr_best_epoch': sel['old_rr_best_epoch'],
    'new_mean_best_epoch': sel['new_mean_best_epoch'],
    'epoch_changed': sel['epoch_changed'],
    'old_rr_best_val_retmse10': sel['old_rr_best_val_retmse10'],
    'new_mean_best_val_retmse10': sel['new_mean_best_val_retmse10'],
    'new_mean_test_retmse10': stage1['retmse10'],
    'new_mean_test_top10_overlap_vs_roundrobin': stage1['top10_overlap_vs_roundrobin'],
    'corrected_stage2_test_mse': stage2['test_mse'],
    'corrected_stage2_test_mae': stage2.get('test_mae'),
    'corrected_stage2_lambda': stage2.get('test_lambda', stage2.get('lambda')),
}
with open(f'{out}/final_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'[run_v2_meanmix_checkpoint_correction_pool] wrote {out}/final_summary.json: {summary}')
PYEOF

  echo "=== [${cell}/P100/V2] checkpoint-correction cell complete ==="
}

echo "### nvidia-smi before starting ###"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

run_cell ETTh1 96    "${REF_ETTH1_96}"    "${BASE_ETTH1_96}"
run_cell ETTh1 720   "${REF_ETTH1_720}"   "${BASE_ETTH1_720}"
run_cell Weather 96  "${REF_WEATHER_96}"  "${BASE_WEATHER_96}"
run_cell Weather 720 "${REF_WEATHER_720}" "${BASE_WEATHER_720}"

echo "=== TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 V2 P100 (4 cells) complete ==="
