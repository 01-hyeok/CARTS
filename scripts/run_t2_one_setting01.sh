#!/usr/bin/env bash
# TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01 -- one (dataset, horizon,
# seed) setting: trains ONLY T0 (TRUE Original KL, zero SlotHeads
# parameters, via `train_j_shared_encoder_drift01.py` completely
# UNMODIFIED -- the actual reference implementation, not a
# reimplementation), builds its retrieval cache, and runs Stage2.
#
# T1/T2/T4/T10 are NOT retrained here (PART 18: "기존 TRACK-T 결과 처리
# ... 기존 결과를 삭제하거나 덮어쓰지 마라") -- they are read directly,
# read-only, from `results/TRACK-T-PURE-MULTISLOT-VALIDATION01/` (S1,
# S2, S4, S10 respectively) by the summary-building step, not by this
# script.
#
# Usage: run_t2_one_setting01.sh <Dataset> <reference_ckpt> <pred_len> <seed>
set -euo pipefail

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
SEQ_LEN="$PRED_LEN"

R_BASE_CKPT="checkpoints/track_r_final_method_generalization01/${DATASET}/H${PRED_LEN}/seed${SEED}/base/checkpoint.pth"
if [ ! -f "$R_BASE_CKPT" ]; then
  echo "[ISSUE][ABORT] TRACK-R base checkpoint not found: $R_BASE_CKPT -- TRACK-T2 requires TRACK-R's base to already exist for this setting (common-base reuse, PART 11/18)."
  exit 1
fi

CELL="${DATASET}_${PRED_LEN}"
OUT_ROOT="results/TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01/${DATASET}/H${PRED_LEN}/seed${SEED}"
CK_ROOT="checkpoints/track_t2_projection_multislot_decomposition01/${DATASET}/H${PRED_LEN}/seed${SEED}"

echo "=== TRACK-T2 ${DATASET} H${PRED_LEN} seed${SEED} T0 : TRUE Original KL Stage1 training (train_j_shared_encoder_drift01.py, UNMODIFIED) ==="
python scripts/train_j_shared_encoder_drift01.py \
  --reference_ckpt "$REF_CKPT" --cell "$CELL" \
  --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
  --instrument off \
  --out_dir "${OUT_ROOT}/T0/stage1" \
  --checkpoints "${CK_ROOT}/T0/stage1"

echo "=== TRACK-T2 ${DATASET} H${PRED_LEN} seed${SEED} T0 : retrieval cache ==="
python scripts/build_t2_true_original_kl_cache01.py \
  --reference_ckpt "$REF_CKPT" \
  --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
  --retriever_checkpoint "${CK_ROOT}/T0/stage1/${CELL}/checkpoint.pth" \
  --out_dir "${OUT_ROOT}/T0/cache"

echo "=== TRACK-T2 ${DATASET} H${PRED_LEN} seed${SEED} T0 : Stage2 ==="
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
  --base_checkpoint "$R_BASE_CKPT" \
  --cache_dir "${OUT_ROOT}/T0/cache" \
  --out_dir "${OUT_ROOT}/T0/stage2"

echo "=== TRACK-T2 ${DATASET} H${PRED_LEN} seed${SEED} : T0 complete ==="
