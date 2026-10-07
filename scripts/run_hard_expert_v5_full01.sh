#!/usr/bin/env bash
# TRACK-HARD-EXPERT-V5-FULL01, one cell at a time:
#   bash scripts/run_hard_expert_v5_full01.sh <cell> <ref_ckpt> <pred_len> <seq_len>
# Stage1 ONLY (spec section 14 -- Stage2 is not run automatically; it is
# a separate, explicitly-decided follow-up once Stage1 specialization
# results are reviewed). Full-Candidate, fresh-every-step,
# candidate-gradient-ON, exactly 10 epochs (no early stopping).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; PRED_LEN="$3"; SEQ_LEN="$4"

echo "=== [${CELL}] Hard-Expert-V5 Stage-1 (10 fixed epochs, writes its own per-head report) ==="
python -u scripts/train_hard_expert_v5_full01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --skip_init_hash_check \
  --out_dir "results/TRACK-HARD-EXPERT-V5-FULL01" --checkpoints "checkpoints/track_hard_expert_v5_full01"

echo "=== [${CELL}] TRACK-HARD-EXPERT-V5-FULL01 Stage1 complete ==="
