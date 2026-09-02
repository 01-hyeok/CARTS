#!/usr/bin/env bash
# Shared candidate pool + teacher scores for the first-wave settings.
set -u
GPU=${GPU:-1}
POOL=${POOL:-100}
for dataset in ETTh1 ETTm1; do
  for pred in ${PREDS:-96}; do
    ckpt=$(ls -d checkpoints/stage2/${dataset}/seq${pred}_pred${pred}/*s2_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "[miss] ${dataset}/${pred}"; continue; }
    echo "[run ] ${dataset}/${pred}"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/precompute_utility_teacher.py \
      --checkpoint "$ckpt" --pool_m "$POOL" --splits train,val,test \
      > logs/utility_teacher_precompute_${dataset}_${pred}.log 2>&1
    echo "[done] ${dataset}/${pred} exit=$? $(grep -c '\[done\]' logs/utility_teacher_precompute_${dataset}_${pred}.log) splits"
  done
done
