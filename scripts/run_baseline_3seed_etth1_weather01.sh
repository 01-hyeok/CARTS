#!/usr/bin/env bash
# Ad-hoc follow-up: 3-seed Base Forecaster / Raw Cosine / Original KL
# baselines for ETTh1+Weather x H{96,192,336,720} x seed{0,1,2}.
#
# Reuses EXISTING canonical scripts unmodified, same methodology as
# the original TRACK-R-FINAL-METHOD-GENERALIZATION01 /
# TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01 tracks this extends (not a
# new protocol): train_r_base_forecaster01.py (Base),
# build_r_retrieval_cache01.py --retriever cosine (Raw Cosine, no
# training) + train_r_stage2_lambda01.py (Stage2), and
# train_j_shared_encoder_drift01.py (Original KL Stage1 training) +
# build_r_retrieval_cache01.py --retriever original_kl +
# train_r_stage2_lambda01.py. These are the SAME established-protocol
# scripts from the already-closed TRACK-R/TRACK-S tracks -- this is an
# extension (more seeds/cells), not a new experiment design.
#
# GPU1 only, per standing rule (the original run_s_original_kl01.sh
# used GPU2 "to run parallel to TRACK-R" -- overridden here to honor
# the current project-wide GPU1-only rule). Runs strictly SEQUENTIALLY
# within this script (one external process at a time) specifically to
# avoid adding MORE concurrent GPU1 load on top of the already-running
# main grid + Phase B jobs -- by explicit user decision to proceed
# anyway, accepting some slowdown, but not multiplying it further.
# Resumable: skips any step whose output already exists.
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

run_one () {
  local ds="$1" h="$2" seed="$3"
  local ref; ref=$(ref_ckpt "$ds" "$h")
  local r_root="${R_ROOT_BASE}/${ds}/H${h}/seed${seed}"
  local r_ck="${R_CK_BASE}/${ds}/H${h}/seed${seed}"
  local s_root="${S_ROOT_BASE}/${ds}/H${h}/seed${seed}"
  local s_ck="${S_CK_BASE}/${ds}/H${h}/seed${seed}"
  mkdir -p "$r_root/base" "$r_root/cosine/cache" "$r_root/cosine/stage2" \
          "$s_root/original_kl/stage1" "$s_root/original_kl/cache" "$s_root/original_kl/stage2" \
          "$s_ck/original_kl"

  echo "=== [${ds}_${h} seed${seed}] Base Forecaster ==="
  if [ -f "$r_ck/base/checkpoint.pth" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/train_r_base_forecaster01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --out_dir "$r_root/base" --checkpoint_path "$r_ck/base/checkpoint.pth"
  fi

  echo "=== [${ds}_${h} seed${seed}] Raw Cosine cache + Stage2 ==="
  if [ -f "$r_root/cosine/stage2/metrics.json" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/build_r_retrieval_cache01.py \
      --retriever cosine --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --out_dir "$r_root/cosine/cache"
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --base_checkpoint "$r_ck/base/checkpoint.pth" \
      --cache_dir "$r_root/cosine/cache" --out_dir "$r_root/cosine/stage2"
  fi

  echo "=== [${ds}_${h} seed${seed}] Original KL Stage1 ==="
  local kl_ckpt="${s_ck}/original_kl/${ds}_${h}/checkpoint.pth"
  if [ -f "$kl_ckpt" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/train_j_shared_encoder_drift01.py \
      --reference_ckpt "$ref" --cell "${ds}_${h}" \
      --pred_len "$h" --seq_len "$h" --init_seed "$seed" --loader_seed "$seed" \
      --out_dir "$s_root/original_kl/stage1" --checkpoints "${s_ck}/original_kl"
  fi

  echo "=== [${ds}_${h} seed${seed}] Original KL cache + Stage2 ==="
  if [ -f "$s_root/original_kl/stage2/metrics.json" ]; then
    echo "---- already exists, skip ----"
  else
    python -u scripts/build_r_retrieval_cache01.py \
      --retriever original_kl --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --retriever_checkpoint "$kl_ckpt" --out_dir "$s_root/original_kl/cache"
    python -u scripts/train_r_stage2_lambda01.py \
      --reference_ckpt "$ref" --pred_len "$h" --seq_len "$h" --seed "$seed" \
      --base_checkpoint "$r_ck/base/checkpoint.pth" \
      --cache_dir "$s_root/original_kl/cache" --out_dir "$s_root/original_kl/stage2"
  fi

  echo "=== [${ds}_${h} seed${seed}] DONE ==="
}

for ds in ETTh1 Weather; do
  for h in 96 192 336 720; do
    for seed in 0 1 2; do
      run_one "$ds" "$h" "$seed"
    done
  done
done

echo "### ALL 3-seed baseline settings complete. ###"
