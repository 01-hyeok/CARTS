#!/usr/bin/env bash
# EXP-ORACLE-SCRATCH-TF01: teacher-forced Individual vs Set Oracle,
# scratch Stage-1 encoder. Completely separate from EXP-ORACLE-SCRATCH01
# (own checkpoints/results/logs). Runs H96 then H720 in sequence (per the
# user's explicit go-ahead to queue H720 too -- originally the spec asked
# to stop after H96, but the user later authorized queuing H720
# immediately after).
#
# Stage-2 hyperparameters are the project's OWN established defaults
# (lr=0.001, epochs=10, patience=5) -- NOT invented, matching the
# corrected EXP-ORACLE-SCRATCH01 run exactly, per the user's explicit
# instruction not to deviate again.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/exp_oracle_scratch_tf01"
CKPT_ROOT="checkpoints/exp_oracle_scratch_tf01"
RESULT_ROOT="results/EXP-ORACLE-SCRATCH-TF01"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

ETTH1_S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
ETTH1_S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"

run_horizon() {
  local H="$1" REF="$2"
  echo "=================================================================="
  echo "[oracle_scratch_tf01] ===== HORIZON H${H} ====="
  echo "=================================================================="
  local SHARED_ENC="$LOG_ROOT/shared_encoder_init_H${H}.pth"
  local SHARED_BASE="$LOG_ROOT/shared_base_init_H${H}.pth"
  local SHARED_GATE="$LOG_ROOT/shared_gate_init_H${H}.pth"
  local ARMS=("individual cosine" "individual asymmetric" "set cosine" "set asymmetric")

  # ---- STAGE 1 (teacher forced): 4 arms, first saves the shared encoder init ----
  local FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local MODEL_ID="carts_oracle_scratch_tf01_${TARGET}_${SCORER}_H${H}"
    local LOG="$LOG_ROOT/stage1_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch_tf01] --- Stage-1 (TF) H${H} target=${TARGET} scorer=${SCORER} ---"
    if [ "$FIRST" -eq 1 ]; then
      python -u scripts/train_oracle_scratch_tf01.py \
        --reference_ckpt "$REF" --target "$TARGET" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_out "$SHARED_ENC" \
        > "$LOG" 2>&1
      FIRST=0
    else
      python -u scripts/train_oracle_scratch_tf01.py \
        --reference_ckpt "$REF" --target "$TARGET" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_in "$SHARED_ENC" \
        > "$LOG" 2>&1
    fi
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] Stage-1 ${TARGET}/${SCORER} H${H}"; return 1; fi
    echo "[ok] Stage-1 ${TARGET}/${SCORER} H${H}"
  done

  # ---- EVAL: teacher-forced diagnostics + free-running cache, 4 arms ----
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local MODEL_ID="carts_oracle_scratch_tf01_${TARGET}_${SCORER}_H${H}"
    local S1_CKPT="$CKPT_ROOT/ETTh1/seq${H}_pred${H}/${MODEL_ID}/checkpoint.pth"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${SCORER}_H${H}"
    local EVAL_OUT="$RESULT_ROOT/H${H}/${TARGET}_${SCORER}"
    local LOG="$LOG_ROOT/eval_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch_tf01] --- Eval H${H} target=${TARGET} scorer=${SCORER} ---"
    python -u scripts/eval_oracle_scratch_tf01.py \
      --reference_ckpt "$REF" --stage1_checkpoint "$S1_CKPT" \
      --top_k 10 --chunk_size 4096 --n_diag_queries 500 \
      --out_dir "$EVAL_OUT" --cache_dir "$CACHE_DIR" \
      > "$LOG" 2>&1
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval ${TARGET}/${SCORER} H${H}"; return 1; fi
    echo "[ok] eval ${TARGET}/${SCORER} H${H}"
  done

  # ---- STAGE 2 (project defaults ONLY: lr=0.001/epochs=10/patience=5): ----
  # 4 retrieval arms + 1 no-retrieval Base-only control, all sharing one
  # random Base/Gate init. Reuses scripts/train_oracle_scratch01_stage2.py
  # UNCHANGED (per spec section 9's own instruction: "기존 CARTS protocol을 유지").
  FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${SCORER}_H${H}"
    local MODEL_ID="carts_oracle_scratch_tf01_stage2_${TARGET}_${SCORER}_H${H}"
    local OUT_DIR="$RESULT_ROOT/H${H}/${TARGET}_${SCORER}/stage2"
    local LOG="$LOG_ROOT/stage2_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch_tf01] --- Stage-2 H${H} target=${TARGET} scorer=${SCORER} ---"
    if [ "$FIRST" -eq 1 ]; then
      python -u scripts/train_oracle_scratch01_stage2.py \
        --retrieval_cache_dir "$CACHE_DIR" --seq_len "$H" --pred_len "$H" --enc_in 7 \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
        --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
        --shared_base_init_out "$SHARED_BASE" --shared_gate_init_out "$SHARED_GATE" \
        > "$LOG" 2>&1
      FIRST=0
    else
      python -u scripts/train_oracle_scratch01_stage2.py \
        --retrieval_cache_dir "$CACHE_DIR" --seq_len "$H" --pred_len "$H" --enc_in 7 \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
        --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
        --shared_base_init_in "$SHARED_BASE" --shared_gate_init_in "$SHARED_GATE" \
        > "$LOG" 2>&1
    fi
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 ${TARGET}/${SCORER} H${H}"; return 1; fi
    echo "[ok] Stage-2 ${TARGET}/${SCORER} H${H}"
  done

  local BASEONLY_CACHE="$CKPT_ROOT/retrieval_cache/individual_cosine_H${H}"
  local OUT_DIR="$RESULT_ROOT/H${H}/base_only"
  local LOG="$LOG_ROOT/stage2_base_only_H${H}.log"
  echo "[oracle_scratch_tf01] --- Stage-2 H${H} Base Forecaster only (no retrieval) ---"
  python -u scripts/train_oracle_scratch01_stage2.py \
    --retrieval_cache_dir "$BASEONLY_CACHE" --no_retrieval --seq_len "$H" --pred_len "$H" --enc_in 7 \
    --checkpoints "$CKPT_ROOT" --model_id "carts_oracle_scratch_tf01_stage2_base_only_H${H}" \
    --des "base_only_H${H}" --out_dir "$OUT_DIR" \
    --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
    --shared_base_init_in "$SHARED_BASE" \
    > "$LOG" 2>&1
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 base_only H${H}"; return 1; fi
  echo "[ok] Stage-2 base_only H${H}"

  echo "[oracle_scratch_tf01] ===== HORIZON H${H} COMPLETE ====="
}

run_horizon 96 "$ETTH1_S1_96" || { echo "[oracle_scratch_tf01] H96 FAILED, stopping"; exit 1; }
run_horizon 720 "$ETTH1_S1_720" || { echo "[oracle_scratch_tf01] H720 FAILED, stopping"; exit 1; }

echo "[oracle_scratch_tf01] ALL DONE $(date -Is)"
