#!/usr/bin/env bash
# Track B4: EXP-CORRECTION-ENCODER01. Scope per spec section 11: ETTh1,
# pred_len=96, seed 0, all 7 channels ONLY for this first round -- other
# horizons/datasets are explicitly deferred until this result is meaningful.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/exp_correction_encoder01"
CKPT_ROOT="checkpoints/exp_correction_encoder01"
RESULT_ROOT="results/EXP-CORRECTION-ENCODER01"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT/H96"

ETTH1_S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
ETTH1_S2_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"

echo "[correction_encoder01] === Stage-1: C-Encoder training ==="
python -u scripts/train_correction_encoder01.py \
  --reference_ckpt "$ETTH1_S1_96" --stage2_checkpoint "$ETTH1_S2_96" --pred_len 96 \
  --checkpoints "$CKPT_ROOT" --model_id carts_correction_encoder01_H96 --des correction_encoder01_H96 \
  --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
  > "$LOG_ROOT/stage1_H96.log" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] Stage-1 C-Encoder"; exit 1; fi
echo "[ok] Stage-1 C-Encoder"

CORR_CKPT="$CKPT_ROOT/ETTh1/seq96_pred96/carts_correction_encoder01_H96/checkpoint.pth"

echo "[correction_encoder01] === Eval: retrieval quality (future) ==="
python -u scripts/eval_correction_encoder01.py \
  --reference_ckpt "$ETTH1_S1_96" --stage2_checkpoint "$ETTH1_S2_96" \
  --encoder future --pred_len 96 --chunk_size 2048 \
  --out "$RESULT_ROOT/H96/eval_future.json" \
  > "$LOG_ROOT/eval_future_H96.log" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval future"; exit 1; fi
echo "[ok] eval future"

echo "[correction_encoder01] === Eval: retrieval quality (correction) ==="
python -u scripts/eval_correction_encoder01.py \
  --reference_ckpt "$ETTH1_S1_96" --stage2_checkpoint "$ETTH1_S2_96" \
  --encoder correction --correction_encoder_checkpoint "$CORR_CKPT" --pred_len 96 --chunk_size 2048 \
  --out "$RESULT_ROOT/H96/eval_correction.json" \
  > "$LOG_ROOT/eval_correction_H96.log" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval correction"; exit 1; fi
echo "[ok] eval correction"

echo "[correction_encoder01] === Stage-2: future encoder + correction fusion ==="
python -u scripts/train_correction_encoder01_stage2.py \
  --reference_ckpt "$ETTH1_S1_96" --stage2_checkpoint "$ETTH1_S2_96" \
  --encoder future --pred_len 96 --chunk_size 2048 --gate_epochs 200 --gate_patience 20 \
  --out_dir "$RESULT_ROOT/H96/stage2_future" \
  > "$LOG_ROOT/stage2_future_H96.log" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 future"; exit 1; fi
echo "[ok] stage2 future"

echo "[correction_encoder01] === Stage-2: correction encoder + correction fusion ==="
python -u scripts/train_correction_encoder01_stage2.py \
  --reference_ckpt "$ETTH1_S1_96" --stage2_checkpoint "$ETTH1_S2_96" \
  --encoder correction --correction_encoder_checkpoint "$CORR_CKPT" --pred_len 96 --chunk_size 2048 \
  --gate_epochs 200 --gate_patience 20 \
  --out_dir "$RESULT_ROOT/H96/stage2_correction" \
  > "$LOG_ROOT/stage2_correction_H96.log" 2>&1
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 correction"; exit 1; fi
echo "[ok] stage2 correction"

echo "[correction_encoder01] ALL DONE $(date -Is)"
