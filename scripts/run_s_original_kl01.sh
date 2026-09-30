#!/usr/bin/env bash
# TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01 -- runs the ONE new arm (S1,
# Original KL) for a single (dataset, horizon, seed) setting, then
# evaluates it through the SAME Stage2 (Uniform + Mixture + Trainable
# Global Lambda) as TRACK-R. Reuses TRACK-R's base forecaster
# read-only. Independent output/checkpoint directories throughout --
# never touches TRACK-R's files. Runs on GPU2 (free GPU), never GPU1
# (TRACK-R).
#
# Usage: run_s_original_kl01.sh <dataset_name_for_paths> <reference_ckpt> <pred_len> <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=2

DATASET="$1"
REF_CKPT="$2"
PRED_LEN="$3"
SEED="$4"
H="H${PRED_LEN}"

S_ROOT="results/TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01/${DATASET}/${H}/seed${SEED}"
S_CKROOT="checkpoints/track_s_kl_contribution_decomposition01/${DATASET}/${H}/seed${SEED}"
R_ROOT="results/TRACK-R-FINAL-METHOD-GENERALIZATION01/${DATASET}/${H}/seed${SEED}"
R_CKROOT="checkpoints/track_r_final_method_generalization01/${DATASET}/${H}/seed${SEED}"
mkdir -p "$S_ROOT/original_kl" "$S_CKROOT"

echo "########## TRACK-S setting: ${DATASET} ${H} seed${SEED} (GPU2, parallel to TRACK-R/GPU1) =========="

# sanity: TRACK-R's base forecaster must already exist (read-only reuse, PART 9)
test -f "$R_CKROOT/base/checkpoint.pth" || { echo "[ISSUE][ABORT] TRACK-R base checkpoint not found: $R_CKROOT/base/checkpoint.pth"; exit 1; }

echo "---- S1 Original KL Stage1 (new training) ----"
python scripts/train_j_shared_encoder_drift01.py \
  --reference_ckpt "$REF_CKPT" --cell "${DATASET}_${PRED_LEN}" \
  --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --init_seed "$SEED" --loader_seed "$SEED" \
  --out_dir "$S_ROOT/original_kl/stage1" --checkpoints "$S_CKROOT/original_kl"
KL_CKPT="$S_CKROOT/original_kl/${DATASET}_${PRED_LEN}/checkpoint.pth"

echo "---- S1 Original KL retrieval cache (Uniform aggregate) ----"
python scripts/build_r_retrieval_cache01.py \
  --retriever original_kl --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --retriever_checkpoint "$KL_CKPT" --out_dir "$S_ROOT/original_kl/cache"

echo "---- S1 Original KL Stage2 (global lambda, base reused read-only from TRACK-R) ----"
python scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "$REF_CKPT" --pred_len "$PRED_LEN" --seq_len "$PRED_LEN" --seed "$SEED" \
  --base_checkpoint "$R_CKROOT/base/checkpoint.pth" \
  --cache_dir "$S_ROOT/original_kl/cache" --out_dir "$S_ROOT/original_kl/stage2"

echo "########## TRACK-S setting ${DATASET} ${H} seed${SEED}: ALL DONE =========="
