#!/usr/bin/env bash
# TRACK-V-MULTIQUERY-GENERALIZATION01 -- one (dataset, horizon, seed)
# setting: trains V0 (TRUE Original KL, num_query_views=0, via
# train_j_shared_encoder_drift01.py UNMODIFIED) and V1/V2/V5 (via
# train_t_pure_multislot01.py UNMODIFIED, --num_slots 1/2/5 -- this
# script IS TRACK-T's own Pure Multi-Slot trainer; V1/V2 are
# architecturally IDENTICAL to TRACK-T's S1/S2, V5 is a new
# --num_slots value the script already supports generically via
# SlotHeads' own per-slot-index seeded init, which is exactly what
# gives W_1(S=1)==W_1(S=2)==W_1(S=5) and W_2(S=2)==W_2(S=5) for free
# -- PART 4's cross-arm initialization-fairness requirement holds by
# construction, verified by this track's own unit tests).
#
# Usage: run_v_one_setting01.sh <Dataset> <reference_ckpt> <pred_len> <seed> <base_checkpoint> [full|top100]
set -euo pipefail

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
BASE_CKPT="$5"
CANDIDATE_POOL_MODE="${6:-${CANDIDATE_POOL_MODE:-full}}"
SEQ_LEN="$PRED_LEN"

if [[ "$CANDIDATE_POOL_MODE" != "full" && "$CANDIDATE_POOL_MODE" != "top100" ]]; then
  echo "[ISSUE][ABORT] candidate pool mode must be full or top100: $CANDIDATE_POOL_MODE"
  exit 1
fi

if [ ! -f "$BASE_CKPT" ]; then
  echo "[ISSUE][ABORT] base checkpoint not found: $BASE_CKPT"
  exit 1
fi

CELL="${DATASET}_${PRED_LEN}"
OUT_ROOT="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${DATASET}/H${PRED_LEN}/seed${SEED}"
CK_ROOT="checkpoints/track_v_multiquery_generalization01/${DATASET}/H${PRED_LEN}/seed${SEED}"
POOL_CACHE="${OUT_ROOT}/shared_candidate_pool/top100"
POOL_ARGS=(--candidate_pool_mode "$CANDIDATE_POOL_MODE")
if [[ "$CANDIDATE_POOL_MODE" == "top100" ]]; then
  echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} : build shared Top-100 pool ONCE ==="
  python scripts/build_v_shared_candidate_pool01.py \
    --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --out_dir "$POOL_CACHE"
  POOL_ARGS+=(--candidate_pool_cache_dir "$POOL_CACHE")
fi

echo "[track_v] candidate_pool_mode=$CANDIDATE_POOL_MODE"

echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} V0 (S=0, TRUE Original KL) : Stage1 ==="
python scripts/train_j_shared_encoder_drift01.py \
  --reference_ckpt "$REF_CKPT" --cell "$CELL" \
  --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
  --instrument off "${POOL_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/V0/stage1" \
  --checkpoints "${CK_ROOT}/V0/stage1"

echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} V0 : retrieval cache ==="
python scripts/build_t2_true_original_kl_cache01.py \
  --reference_ckpt "$REF_CKPT" \
  --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
  --retriever_checkpoint "${CK_ROOT}/V0/stage1/${CELL}/checkpoint.pth" "${POOL_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/V0/cache"

echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} V0 : Stage2 ==="
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
  --base_checkpoint "$BASE_CKPT" \
  --cache_dir "${OUT_ROOT}/V0/cache" \
  --out_dir "${OUT_ROOT}/V0/stage2"

for S in 1 2 5; do
  ARM="V${S}"
  echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} (S=${S}) : Stage1 ==="
  python scripts/train_t_pure_multislot01.py \
    --reference_ckpt "$REF_CKPT" --num_slots "$S" --cell "$CELL" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
    --skip_init_hash_check "${POOL_ARGS[@]}" \
    --out_dir "${OUT_ROOT}/${ARM}/stage1" \
    --checkpoints "${CK_ROOT}/${ARM}/stage1"

  echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : retrieval cache ==="
  python scripts/build_t_multislot_cache01.py \
    --reference_ckpt "$REF_CKPT" --num_slots "$S" \
    --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --retriever_checkpoint "${CK_ROOT}/${ARM}/stage1/checkpoint.pth" "${POOL_ARGS[@]}" \
    --out_dir "${OUT_ROOT}/${ARM}/cache"

  echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} ${ARM} : Stage2 ==="
  python scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$SEQ_LEN" --seed "$SEED" \
    --base_checkpoint "$BASE_CKPT" \
    --cache_dir "${OUT_ROOT}/${ARM}/cache" \
    --out_dir "${OUT_ROOT}/${ARM}/stage2"
done

echo "=== TRACK-V ${DATASET} H${PRED_LEN} seed${SEED} : all 4 arms (V0/V1/V2/V5) complete ==="
