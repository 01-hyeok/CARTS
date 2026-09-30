#!/usr/bin/env bash
# TRACK-R-FINAL-METHOD-GENERALIZATION01 -- runs ONE full (dataset, horizon,
# seed) setting end-to-end: base forecaster -> J1 -> J1 relevance reference
# -> M2 -> 3 retrieval caches (cosine/J1/M2) -> 3 Stage2 global-lambda
# trainings (B1/B2/B3). GPU1 only, strictly sequential (never run two
# invocations of this script concurrently).
#
# Usage: run_r_one_setting01.sh <dataset_name_for_paths> <reference_ckpt> <pred_len> <seed>
#   dataset_name_for_paths: "ETTh1" or "Weather" (used only for output paths)
set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
H="H${PRED_LEN}"

ROOT="results/TRACK-R-FINAL-METHOD-GENERALIZATION01/${DATASET}/${H}/seed${SEED}"
CKROOT="checkpoints/track_r_final_method_generalization01/${DATASET}/${H}/seed${SEED}"
mkdir -p "$ROOT" "$CKROOT"

echo "########## TRACK-R setting: ${DATASET} ${H} seed${SEED} =========="

echo "---- base forecaster ----"
python scripts/train_r_base_forecaster01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --out_dir "$ROOT/base" --checkpoint_path "$CKROOT/base/checkpoint.pth"

echo "---- J1 Stage1 ----"
python scripts/train_j2_key_update_decomposition01.py \
  --reference_ckpt "$REF_CKPT" --arm J1_stopgrad_key --cell "${DATASET}_${PRED_LEN}" \
  --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
  --out_dir "$ROOT/J1/stage1" --checkpoints "$CKROOT/J1" --skip_init_hash_check
J1_CKPT="$CKROOT/J1/${DATASET}_${PRED_LEN}/J1_stopgrad_key/checkpoint.pth"

echo "---- J1 relevance reference (train-only) ----"
python scripts/precompute_r_j1_reference01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --j1_checkpoint "$J1_CKPT" --out_dir "$ROOT/M2/j1_reference"

echo "---- M2 Stage1 ----"
python scripts/train_m_relevance_constrained_multislot01.py \
  --reference_ckpt "$REF_CKPT" --arm M2_relevance_budget --cell "${DATASET}_${PRED_LEN}" \
  --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
  --j1_ref_dir "$ROOT/M2/j1_reference" \
  --out_dir "$ROOT/M2/stage1" --checkpoints "$CKROOT/M2" --skip_init_hash_check
M2_CKPT="$CKROOT/M2/${DATASET}_${PRED_LEN}/M2_relevance_budget/checkpoint.pth"

echo "---- retrieval caches: cosine / J1 / M2 ----"
python scripts/build_r_retrieval_cache01.py \
  --retriever cosine --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --out_dir "$ROOT/cosine"
python scripts/build_r_retrieval_cache01.py \
  --retriever j1 --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --retriever_checkpoint "$J1_CKPT" --out_dir "$ROOT/J1/cache"
python scripts/build_r_retrieval_cache01.py \
  --retriever m2 --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --retriever_checkpoint "$M2_CKPT" --out_dir "$ROOT/M2/cache"

echo "---- Stage2 global lambda: B1 (cosine) / B2 (J1) / B3 (M2) ----"
BASE_CKPT="$CKROOT/base/checkpoint.pth"
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --base_checkpoint "$BASE_CKPT" --cache_dir "$ROOT/cosine" --out_dir "$ROOT/cosine/stage2"
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --base_checkpoint "$BASE_CKPT" --cache_dir "$ROOT/J1/cache" --out_dir "$ROOT/J1/stage2"
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --base_checkpoint "$BASE_CKPT" --cache_dir "$ROOT/M2/cache" --out_dir "$ROOT/M2/stage2"

echo "########## TRACK-R setting ${DATASET} ${H} seed${SEED}: ALL DONE =========="
