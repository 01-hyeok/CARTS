#!/usr/bin/env bash
# TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01 -- one (dataset, horizon,
# seed) setting: trains ALL FOUR arms (U0 True Original KL, U1 Shared
# Symmetric Projection, U2 Query-only Asymmetric, U3 Key-only
# Asymmetric) fresh, builds their retrieval caches, and runs Stage2 for
# each, reusing TRACK-R's frozen base forecaster checkpoint READ-ONLY.
#
# All four arms are retrained fresh in THIS track (unlike TRACK-T2,
# which reused T1-T10 read-only) because U1/U2/U3 require an IDENTICAL
# initial projection weight tensor across arms (PART 6) -- a stricter
# control than anything the existing TRACK-T/TRACK-T2 checkpoints were
# built to satisfy.
#
# Usage: run_u_one_setting01.sh <Dataset> <reference_ckpt> <pred_len> <seed>
set -euo pipefail

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
SEQ_LEN="$PRED_LEN"

R_BASE_CKPT="checkpoints/track_r_final_method_generalization01/${DATASET}/H${PRED_LEN}/seed${SEED}/base/checkpoint.pth"
if [ ! -f "$R_BASE_CKPT" ]; then
  echo "[ISSUE][ABORT] TRACK-R base checkpoint not found: $R_BASE_CKPT"
  exit 1
fi

CELL="${DATASET}_${PRED_LEN}"
OUT_ROOT="results/TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01/${DATASET}/H${PRED_LEN}/seed${SEED}"
CK_ROOT="checkpoints/track_u_asymmetry_capacity_decomposition01/${DATASET}/H${PRED_LEN}/seed${SEED}"

for ARM in U0 U1 U2 U3; do
  echo "=== TRACK-U ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : Stage1 training ==="
  python scripts/train_u_asymmetry_capacity01.py \
    --reference_ckpt "$REF_CKPT" --arm "$ARM" --cell "$CELL" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
    --out_dir "${OUT_ROOT}/${ARM}/stage1" \
    --checkpoints "${CK_ROOT}/${ARM}/stage1"

  echo "=== TRACK-U ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : retrieval cache ==="
  python scripts/build_u_cache01.py \
    --reference_ckpt "$REF_CKPT" --arm "$ARM" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --retriever_checkpoint "${CK_ROOT}/${ARM}/stage1/checkpoint.pth" \
    --out_dir "${OUT_ROOT}/${ARM}/cache"

  echo "=== TRACK-U ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : Stage2 ==="
  python scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --base_checkpoint "$R_BASE_CKPT" \
    --cache_dir "${OUT_ROOT}/${ARM}/cache" \
    --out_dir "${OUT_ROOT}/${ARM}/stage2"
done

echo "=== TRACK-U ${DATASET} H${PRED_LEN} seed${SEED} : all 4 arms (U0/U1/U2/U3) complete ==="
