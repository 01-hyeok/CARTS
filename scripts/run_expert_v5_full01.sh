#!/usr/bin/env bash
# TRACK-EXPERT-V5-FULL01, one cell at a time:
#   bash scripts/run_expert_v5_full01.sh <cell> <ref_ckpt> <pred_len> <seq_len> <base_ckpt> <current_v5_ckpt>
# Full-Candidate, fresh-every-step, candidate-gradient-ON throughout --
# NO efficiency change of any kind (never touches
# utils/full_candidate_bank.py or the R100 trainers). Only the loss
# changes relative to the canonical Full V5
# (`train_t_pure_multislot01.py --num_slots 5`), which this script
# never modifies -- it is loaded read-only for the standalone-head
# comparison step.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; PRED_LEN="$3"; SEQ_LEN="$4"; BASE_CKPT="$5"; CURRENT_V5_CKPT="$6"

OUT_ROOT="results/TRACK-EXPERT-V5-FULL01/${CELL}"
CKPT_ROOT="checkpoints/track_expert_v5_full01/${CELL}"

echo "=== [${CELL}] Expert-V5 Stage-1 (trains + writes its own per-head report) ==="
python -u scripts/train_expert_v5_full01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --skip_init_hash_check \
  --out_dir "results/TRACK-EXPERT-V5-FULL01" --checkpoints "checkpoints/track_expert_v5_full01"

echo "=== [${CELL}] Current-V5 standalone-head evaluation (existing checkpoint, read-only) ==="
python -u scripts/evaluate_v5_standalone_heads01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --current_v5_checkpoint "${CURRENT_V5_CKPT}" \
  --out_dir "${OUT_ROOT}/current_v5_standalone"

echo "=== [${CELL}] Expert-V5 retrieval cache (reused unmodified build_t_multislot_cache01.py, round-robin) ==="
python -u scripts/build_t_multislot_cache01.py \
  --reference_ckpt "${REF}" --num_slots 5 --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
  --retriever_checkpoint "${CKPT_ROOT}/checkpoint.pth" \
  --out_dir "${OUT_ROOT}/cache"

echo "=== [${CELL}] Expert-V5 Stage-2 (reused unmodified train_r_stage2_lambda01.py) ==="
python -u scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
  --base_checkpoint "${BASE_CKPT}" \
  --cache_dir "${OUT_ROOT}/cache" --out_dir "${OUT_ROOT}/stage2"

echo "=== [${CELL}] TRACK-EXPERT-V5-FULL01 pipeline complete ==="
