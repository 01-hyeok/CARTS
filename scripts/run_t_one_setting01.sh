#!/usr/bin/env bash
# TRACK-T-PURE-MULTISLOT-VALIDATION01 -- one (dataset, horizon, seed)
# setting: trains T0/T1/T2/T3 (S=1/2/4/10), builds their retrieval
# caches, and runs Stage2 for each, reusing TRACK-R's frozen base
# forecaster checkpoint READ-ONLY (PART 16: "Base는 해당
# dataset/horizon/seed에서 하나만 학습하고 모든 T0/T1/T2/T3이 공유한다" --
# TRACK-R already trained exactly that base for this setting; this track
# must never retrain or modify it).
#
# Usage: run_t_one_setting01.sh <Dataset> <reference_ckpt> <pred_len> <seed>
set -euo pipefail

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
SEQ_LEN="$PRED_LEN"

R_BASE_CKPT="checkpoints/track_r_final_method_generalization01/${DATASET}/H${PRED_LEN}/seed${SEED}/base/checkpoint.pth"
if [ ! -f "$R_BASE_CKPT" ]; then
  echo "[ISSUE][ABORT] TRACK-R base checkpoint not found: $R_BASE_CKPT -- TRACK-T requires TRACK-R's base to already exist for this setting (PART 16 common-base reuse). Not training a new one."
  exit 1
fi

CELL="${DATASET}_${PRED_LEN}"
OUT_ROOT="results/TRACK-T-PURE-MULTISLOT-VALIDATION01/${DATASET}/H${PRED_LEN}/seed${SEED}"
CK_ROOT="checkpoints/track_t_pure_multislot_validation01/${DATASET}/H${PRED_LEN}/seed${SEED}"

for S in 1 2 4 10; do
  ARM="S${S}"
  echo "=== TRACK-T ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : Stage1 training ==="
  python scripts/train_t_pure_multislot01.py \
    --reference_ckpt "$REF_CKPT" --num_slots "$S" --cell "$CELL" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
    --skip_init_hash_check \
    --out_dir "${OUT_ROOT}/${ARM}/stage1" \
    --checkpoints "${CK_ROOT}/${ARM}/stage1"

  echo "=== TRACK-T ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : retrieval cache ==="
  python scripts/build_t_multislot_cache01.py \
    --reference_ckpt "$REF_CKPT" --num_slots "$S" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --retriever_checkpoint "${CK_ROOT}/${ARM}/stage1/checkpoint.pth" \
    --out_dir "${OUT_ROOT}/${ARM}/cache"

  echo "=== TRACK-T ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : Stage2 ==="
  python scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --base_checkpoint "$R_BASE_CKPT" \
    --cache_dir "${OUT_ROOT}/${ARM}/cache" \
    --out_dir "${OUT_ROOT}/${ARM}/stage2"
done

echo "=== TRACK-T ${DATASET} H${PRED_LEN} seed${SEED} : all 4 arms (S1/S2/S4/S10) complete ==="
