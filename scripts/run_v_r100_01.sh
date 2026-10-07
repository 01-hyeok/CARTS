#!/usr/bin/env bash
# TRACK-V-R100-EFFICIENCY01, one cell at a time:
#   bash scripts/run_v_r100_01.sh <cell> <ref_ckpt> <pred_len> <seq_len> <base_ckpt> <stage1_checkpoint_dir_unused>
# Pure implementation-efficiency re-run of the EXISTING Full-Candidate
# V0/V1/V2/V5 arms: channel-first preprocessing + full-candidate-bank
# refresh every 100 optimizer steps. Candidate support is ALWAYS the
# full memory bank N -- this is NOT a Top-M/P100 arm, and shares NO
# code/results with TRACK-V-MULTIQUERY-GENERALIZATION01's pool_top100
# subtree. Cache-building and Stage-2 reuse the EXISTING, UNMODIFIED
# `build_t2_true_original_kl_cache01.py` / `build_t_multislot_cache01.py`
# / `train_r_stage2_lambda01.py` -- valid because an R100 checkpoint's
# state_dict schema is byte-identical to the canonical Full-online
# checkpoint's, and Stage-2 caching always re-encodes a FRESH full bank
# regardless of which script produced the checkpoint (spec section 6).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; PRED_LEN="$3"; SEQ_LEN="$4"; BASE_CKPT="$5"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-100}"

OUT_ROOT="results/TRACK-V-R100-EFFICIENCY01/${CELL}"
CKPT_ROOT="checkpoints/track_v_r100_efficiency01/${CELL}"

echo "=== [${CELL}] V0 R100 Stage-1 ==="
python -u scripts/train_v0_r100_01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --refresh_interval "${REFRESH_INTERVAL}" \
  --out_dir "results/TRACK-V-R100-EFFICIENCY01" --checkpoints "checkpoints/track_v_r100_efficiency01"

echo "=== [${CELL}] V0 R100 cache (reused unmodified build_t2_true_original_kl_cache01.py) ==="
python -u scripts/build_t2_true_original_kl_cache01.py \
  --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
  --retriever_checkpoint "${CKPT_ROOT}/V0/checkpoint.pth" \
  --out_dir "${OUT_ROOT}/V0/cache"

echo "=== [${CELL}] V0 R100 Stage-2 (reused unmodified train_r_stage2_lambda01.py) ==="
python -u scripts/train_r_stage2_lambda01.py \
  --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
  --base_checkpoint "${BASE_CKPT}" \
  --cache_dir "${OUT_ROOT}/V0/cache" --out_dir "${OUT_ROOT}/V0/stage2"

for S in 1 2 5; do
  ARM="V${S}"
  echo "=== [${CELL}] ${ARM} R100 Stage-1 ==="
  python -u scripts/train_v_r100_01.py \
    --reference_ckpt "${REF}" --num_slots "${S}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
    --refresh_interval "${REFRESH_INTERVAL}" --skip_init_hash_check \
    --out_dir "results/TRACK-V-R100-EFFICIENCY01" --checkpoints "checkpoints/track_v_r100_efficiency01"

  echo "=== [${CELL}] ${ARM} R100 cache (reused unmodified build_t_multislot_cache01.py) ==="
  python -u scripts/build_t_multislot_cache01.py \
    --reference_ckpt "${REF}" --num_slots "${S}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
    --retriever_checkpoint "${CKPT_ROOT}/${ARM}/checkpoint.pth" \
    --out_dir "${OUT_ROOT}/${ARM}/cache"

  echo "=== [${CELL}] ${ARM} R100 Stage-2 (reused unmodified train_r_stage2_lambda01.py) ==="
  python -u scripts/train_r_stage2_lambda01.py \
    --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
    --base_checkpoint "${BASE_CKPT}" \
    --cache_dir "${OUT_ROOT}/${ARM}/cache" --out_dir "${OUT_ROOT}/${ARM}/stage2"
done

echo "=== [${CELL}] staleness diagnostic (V0 and V5, representative arms) ==="
python -u scripts/compute_v_r100_staleness01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --arm V0 --stage1_checkpoint "${CKPT_ROOT}/V0/checkpoint.pth" \
  --out_dir "${OUT_ROOT}/staleness"
python -u scripts/compute_v_r100_staleness01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --arm V5 --stage1_checkpoint "${CKPT_ROOT}/V5/checkpoint.pth" \
  --out_dir "${OUT_ROOT}/staleness"

echo "=== [${CELL}] same-environment efficiency benchmark (V0 and V5) ==="
python -u scripts/bench_v_r100_efficiency01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --arm V0 --n_steps 150 --refresh_interval "${REFRESH_INTERVAL}" --out_dir "${OUT_ROOT}/benchmark"
python -u scripts/bench_v_r100_efficiency01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --arm V5 --n_steps 150 --refresh_interval "${REFRESH_INTERVAL}" --out_dir "${OUT_ROOT}/benchmark"

echo "=== [${CELL}] TRACK-V-R100-EFFICIENCY01 pipeline complete ==="
