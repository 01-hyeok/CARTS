#!/usr/bin/env bash
# P100-restricted 3-seed Raw Cosine / Original KL baselines for
# ETTh1+Weather x H{96,192,336,720} x seed{0,1,2}, directly comparable
# to this session's P100 SoftTopK E2E results (same Shared-Top-100
# pool). Supersedes run_baseline_3seed_etth1_weather01.sh's Full-memory
# version for this purpose (that script's OWN outputs are left
# untouched -- this writes to separate "*_p100" subdirectories).
#
# Base Forecaster is UNAFFECTED by P100 vs Full-memory (no candidates
# involved at all) -- reused as-is from the Full-memory sweep's
# checkpoints, not retrained.
#
# Original KL's Stage1 ENCODER is also reused from the (now-stopped)
# Full-memory sweep where already trained (Stage1 training itself does
# not depend on P100 vs Full-memory -- that choice only affects the
# SCORING/retrieval-cache step, exactly like the main SoftTopK E2E
# track's own `coarse_topk` mode). Trains a fresh Stage1 only for
# settings the Full-memory sweep did not reach (Weather entirely,
# ETTh1_720 seed2).
#
# GPU1 only. Sequential within this script (one external process at a
# time), to avoid adding further concurrent GPU1 load on top of the
# main grid + Phase B jobs already running -- same reasoning as the
# (now-superseded) Full-memory version. Resumable: skips any step
# whose output already exists.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

R_ROOT_BASE="results/TRACK-R-FINAL-METHOD-GENERALIZATION01"
R_CK_BASE="checkpoints/track_r_final_method_generalization01"
S_ROOT_BASE="results/TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01"
S_CK_BASE="checkpoints/track_s_kl_contribution_decomposition01"

ref_ckpt () {
  python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.qa_test_eval_paths01 import reference_ckpt
print(reference_ckpt('$1', $2))
"
}
pool_cache_dir () {
  python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.qa_test_eval_paths01 import candidate_pool_cache
print(candidate_pool_cache('$1', $2))
"
}

run_one () {
  local ds="$1" h="$2" seed="$3"
  local ref; ref=$(ref_ckpt "$ds" "$h")
  local pool; pool=$(pool_cache_dir "$ds" "$h")
  local r_ck="${R_CK_BASE}/${ds}/H${h}/seed${seed}"
  local r_root="${R_ROOT_BASE}/${ds}/H${h}/seed${seed}"
  local s_root="${S_ROOT_BASE}/${ds}/H${h}/seed${seed}"
  local s_ck="${S_CK_BASE}/${ds}/H${h}/seed${seed}"
  mkdir -p "$r_ck/base" "$r_root/cosine_p100/cache" "$r_root/cosine_p100/stage2" \
          "$s_root/original_kl/stage1" "$s_root/original_kl_p100/cache" "$s_root/original_kl_p100/stage2" \
          "${s_ck}/original_kl"

  echo "=== [${ds}_${h} seed${seed}] Base Forecaster (P100-agnostic, reuse-or-train) ==="
  if [ -f "$r_ck/base/checkpoint.pth" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/train_r_base_forecaster01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --out_dir "$r_root/base" --checkpoint_path "$r_ck/base/checkpoint.pth"
  fi

  echo "=== [${ds}_${h} seed${seed}] Raw Cosine P100 cache + Stage2 ==="
  if [ -f "$r_root/cosine_p100/stage2/metrics.json" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/build_r_retrieval_cache_p100_01.py \
      --retriever cosine --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --candidate_pool_cache "$pool" --out_dir "$r_root/cosine_p100/cache"
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --base_checkpoint "$r_ck/base/checkpoint.pth" \
      --cache_dir "$r_root/cosine_p100/cache" --out_dir "$r_root/cosine_p100/stage2"
  fi

  echo "=== [${ds}_${h} seed${seed}] Original KL Stage1 (reused from Full-memory sweep if present) ==="
  local kl_ckpt="${s_ck}/original_kl/${ds}_${h}/checkpoint.pth"
  if [ -f "$kl_ckpt" ]; then
    echo "---- already exists (reused), skip ----"
  else
    python -u scripts/train_j_shared_encoder_drift01.py \
      --reference_ckpt "$ref" --cell "${ds}_${h}" \
      --pred_len "$h" --seq_len "$h" --init_seed "$seed" --loader_seed "$seed" \
      --out_dir "$s_root/original_kl/stage1" --checkpoints "${s_ck}/original_kl"
  fi

  echo "=== [${ds}_${h} seed${seed}] Original KL P100 cache + Stage2 ==="
  if [ -f "$s_root/original_kl_p100/stage2/metrics.json" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/build_r_retrieval_cache_p100_01.py \
      --retriever original_kl --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --retriever_checkpoint "$kl_ckpt" --candidate_pool_cache "$pool" --out_dir "$s_root/original_kl_p100/cache"
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --base_checkpoint "$r_ck/base/checkpoint.pth" \
      --cache_dir "$s_root/original_kl_p100/cache" --out_dir "$s_root/original_kl_p100/stage2"
  fi

  echo "=== [${ds}_${h} seed${seed}] DONE (P100) ==="
}

for ds in ETTh1 Weather; do
  for h in 96 192 336 720; do
    for seed in 0 1 2; do
      run_one "$ds" "$h" "$seed"
    done
  done
done

echo "### ALL 3-seed P100 baseline settings complete. ###"
