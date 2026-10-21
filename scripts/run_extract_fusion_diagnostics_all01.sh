#!/usr/bin/env bash
# Runs extract_fusion_diagnostics_raw01.py for all 5 Phase B
# representative pairs, using their RNG-corrected checkpoints.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

RNG_ROOT="results/TRACK-P100-QA-CAUSAL-VALIDATION01/rng_corrected"
OUT_DIR="results/TRACK-P100-QA-CAUSAL-VALIDATION01/phase_c_raw"

ckpt_path () {
  python3 -c "import json; print(json.load(open('$1'))['checkpoint_path'])"
}

run_one () {
  local cell="$1" method="$2" ref="$3" base="$4" pool="$5" h="$6"
  local noqa_sel="${RNG_ROOT}/${cell}/${method}/NOQA/checkpoint_selection.json"
  local qa_sel="${RNG_ROOT}/${cell}/${method}/QA/checkpoint_selection.json"
  local noqa_ckpt; noqa_ckpt=$(ckpt_path "$noqa_sel")
  local qa_ckpt; qa_ckpt=$(ckpt_path "$qa_sel")
  if [ -f "${OUT_DIR}/${cell}/${method}/qa_raw.pt" ]; then
    echo "---- [${cell}/${method}] already extracted, skip ----"
    return
  fi
  echo "=== [${cell}/${method}] extracting raw fusion diagnostics tensors ==="
  python -u scripts/extract_fusion_diagnostics_raw01.py \
    --reference_ckpt "$ref" --method "$method" --cell "$cell" --pred_len "$h" --seq_len "$h" \
    --base_checkpoint "$base" --candidate_pool_cache "$pool" \
    --noqa_checkpoint "$noqa_ckpt" --qa_checkpoint "$qa_ckpt" \
    --out_dir "$OUT_DIR"
}

ref_ckpt () { python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.qa_test_eval_paths01 import reference_ckpt
print(reference_ckpt('$1', $2))
"; }
pool_cache_dir () { python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.qa_test_eval_paths01 import candidate_pool_cache
print(candidate_pool_cache('$1', $2))
"; }
base_ck () { python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.qa_test_eval_paths01 import base_checkpoint
print(base_checkpoint('$1', $2))
"; }

run_one "ETTh1_192"   "V5"    "$(ref_ckpt ETTh1 192)"   "$(base_ck ETTh1 192)"   "$(pool_cache_dir ETTh1 192)"   192
run_one "ETTh1_336"   "Hard"  "$(ref_ckpt ETTh1 336)"   "$(base_ck ETTh1 336)"   "$(pool_cache_dir ETTh1 336)"   336
run_one "Weather_720" "V5"    "$(ref_ckpt Weather 720)" "$(base_ck Weather 720)" "$(pool_cache_dir Weather 720)" 720
run_one "Weather_336" "Joint" "$(ref_ckpt Weather 336)" "$(base_ck Weather 336)" "$(pool_cache_dir Weather 336)" 336
run_one "Solar_96"    "V5"    "$(ref_ckpt Solar 96)"    "$(base_ck Solar 96)"    "$(pool_cache_dir Solar 96)"    96

echo "### Phase C raw extraction for all 5 pairs complete. ###"
