#!/usr/bin/env bash
# Corrective rerun: H96's Stage-2 (5 configs) was first run with
# unauthorized lr=0.01/epochs=50/patience=10 -- reran here with the
# project's own established Stage-2 defaults (lr=0.001/epochs=10/
# patience=5, matching every other Stage-2 run in this repo). Does not
# touch H720's still-running Stage-1.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/exp_oracle_scratch01"
CKPT_ROOT="checkpoints/exp_oracle_scratch01"
RESULT_ROOT="results/EXP-ORACLE-SCRATCH01"
SHARED_BASE="$LOG_ROOT/shared_base_init_H96.pth"
SHARED_GATE="$LOG_ROOT/shared_gate_init_H96.pth"
ARMS=("individual cosine" "individual asymmetric" "set cosine" "set asymmetric")

FIRST=1
for ARM in "${ARMS[@]}"; do
  read -r TARGET SCORER <<< "$ARM"
  CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${SCORER}_H96"
  MODEL_ID="carts_oracle_scratch01_stage2_${TARGET}_${SCORER}_H96_fixed"
  OUT_DIR="$RESULT_ROOT/H96/${TARGET}_${SCORER}/stage2_fixed"
  LOG="$LOG_ROOT/stage2_${TARGET}_${SCORER}_H96_fixed.log"
  echo "[rerun_stage2] H96 target=${TARGET} scorer=${SCORER}"
  if [ "$FIRST" -eq 1 ]; then
    python -u scripts/train_oracle_scratch01_stage2.py \
      --retrieval_cache_dir "$CACHE_DIR" --seq_len 96 --pred_len 96 --enc_in 7 \
      --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
      --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
      --shared_base_init_out "$SHARED_BASE" --shared_gate_init_out "$SHARED_GATE" \
      > "$LOG" 2>&1
    FIRST=0
  else
    python -u scripts/train_oracle_scratch01_stage2.py \
      --retrieval_cache_dir "$CACHE_DIR" --seq_len 96 --pred_len 96 --enc_in 7 \
      --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
      --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
      --shared_base_init_in "$SHARED_BASE" --shared_gate_init_in "$SHARED_GATE" \
      > "$LOG" 2>&1
  fi
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 ${TARGET}/${SCORER} H96"; exit 1; fi
  echo "[ok] Stage-2 ${TARGET}/${SCORER} H96 (fixed hyperparameters)"
done

OUT_DIR="$RESULT_ROOT/H96/base_only_fixed"
LOG="$LOG_ROOT/stage2_base_only_H96_fixed.log"
BASEONLY_CACHE="$CKPT_ROOT/retrieval_cache/individual_cosine_H96"
echo "[rerun_stage2] H96 Base Forecaster only"
python -u scripts/train_oracle_scratch01_stage2.py \
  --retrieval_cache_dir "$BASEONLY_CACHE" --no_retrieval --seq_len 96 --pred_len 96 --enc_in 7 \
  --checkpoints "$CKPT_ROOT" --model_id "carts_oracle_scratch01_stage2_base_only_H96_fixed" \
  --des "base_only_H96_fixed" --out_dir "$OUT_DIR" \
  --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
  --shared_base_init_in "$SHARED_BASE" \
  > "$LOG" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 base_only H96"; exit 1; fi
echo "[ok] Stage-2 base_only H96 (fixed hyperparameters)"
echo "[rerun_stage2] DONE $(date -Is)"
