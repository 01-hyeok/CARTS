#!/usr/bin/env bash
# EXP-ORACLE-SCRATCH01: Individual Oracle vs Set Oracle, Cosine vs
# Asymmetric scorer, scratch Stage-1 encoder, then a jointly-(re)trained
# Stage-2 (BaseForecastHead + RetrievalGate, from a SHARED random init
# across the 4 arms in a horizon -- no separate Base pretrain/freeze round,
# per the user's explicit revision of the original spec).
#
# Runs the 8 Stage-1 arms SEQUENTIALLY (not in parallel) to avoid the
# GPU-1 self-contention this project has repeatedly hit when running
# multiple of its own jobs on the same GPU at once. Stage-2 (given a
# precomputed retrieval cache) is cheap (~1s/epoch, all in-memory tensors)
# and stays sequential too for simplicity.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/exp_oracle_scratch01"
CKPT_ROOT="checkpoints/exp_oracle_scratch01"
RESULT_ROOT="results/EXP-ORACLE-SCRATCH01"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

ETTH1_S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
ETTH1_S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"

run_horizon() {
  local H="$1" REF="$2"
  echo "=================================================================="
  echo "[oracle_scratch01] ===== HORIZON H${H} ====="
  echo "=================================================================="
  local SHARED_ENC="$LOG_ROOT/shared_encoder_init_H${H}.pth"
  local SHARED_BASE="$LOG_ROOT/shared_base_init_H${H}.pth"
  local SHARED_GATE="$LOG_ROOT/shared_gate_init_H${H}.pth"

  # ---- STAGE 1: 4 arms, first one saves the shared encoder init ----
  local ARMS=("individual cosine" "individual asymmetric" "set cosine" "set asymmetric")
  local FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local MODEL_ID="carts_oracle_scratch01_${TARGET}_${SCORER}_H${H}"
    local LOG="$LOG_ROOT/stage1_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch01] --- Stage-1 H${H} target=${TARGET} scorer=${SCORER} ---"
    if [ "$FIRST" -eq 1 ]; then
      python -u scripts/train_oracle_scratch01.py \
        --reference_ckpt "$REF" --target "$TARGET" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_out "$SHARED_ENC" \
        > "$LOG" 2>&1
      FIRST=0
    else
      python -u scripts/train_oracle_scratch01.py \
        --reference_ckpt "$REF" --target "$TARGET" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_in "$SHARED_ENC" \
        > "$LOG" 2>&1
    fi
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] Stage-1 ${TARGET}/${SCORER} H${H}"; return 1; fi
    echo "[ok] Stage-1 ${TARGET}/${SCORER} H${H}"
  done

  # ---- EVAL / RETRIEVAL CACHE: 4 arms ----
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local MODEL_ID="carts_oracle_scratch01_${TARGET}_${SCORER}_H${H}"
    local S1_CKPT="$CKPT_ROOT/ETTh1/seq${H}_pred${H}/${MODEL_ID}/checkpoint.pth"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${SCORER}_H${H}"
    local EVAL_OUT="$RESULT_ROOT/H${H}/${TARGET}_${SCORER}"
    local LOG="$LOG_ROOT/eval_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch01] --- Eval/cache H${H} target=${TARGET} scorer=${SCORER} ---"
    python -u scripts/eval_oracle_scratch01.py \
      --reference_ckpt "$REF" --stage1_checkpoint "$S1_CKPT" \
      --top_k 10 --chunk_size 4096 --n_diag_queries 500 \
      --out_dir "$EVAL_OUT" --cache_dir "$CACHE_DIR" \
      > "$LOG" 2>&1
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval ${TARGET}/${SCORER} H${H}"; return 1; fi
    echo "[ok] eval/cache ${TARGET}/${SCORER} H${H}"
  done

  # ---- STAGE 2: 4 retrieval arms + 1 no-retrieval (Base Forecaster) control ----
  FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET SCORER <<< "$ARM"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${SCORER}_H${H}"
    local MODEL_ID="carts_oracle_scratch01_stage2_${TARGET}_${SCORER}_H${H}"
    local OUT_DIR="$RESULT_ROOT/H${H}/${TARGET}_${SCORER}/stage2"
    local LOG="$LOG_ROOT/stage2_${TARGET}_${SCORER}_H${H}.log"
    echo "[oracle_scratch01] --- Stage-2 H${H} target=${TARGET} scorer=${SCORER} ---"
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

  # Base Forecaster-only control (no retrieval), sharing the SAME base init
  local BASEONLY_CACHE="$CKPT_ROOT/retrieval_cache/individual_cosine_H${H}"  # any arm's cache; Y_ret zeroed out
  local OUT_DIR="$RESULT_ROOT/H${H}/base_only"
  local LOG="$LOG_ROOT/stage2_base_only_H${H}.log"
  echo "[oracle_scratch01] --- Stage-2 H${H} Base Forecaster only (no retrieval) ---"
  python -u scripts/train_oracle_scratch01_stage2.py \
    --retrieval_cache_dir "$BASEONLY_CACHE" --no_retrieval --seq_len "$H" --pred_len "$H" --enc_in 7 \
    --checkpoints "$CKPT_ROOT" --model_id "carts_oracle_scratch01_stage2_base_only_H${H}" \
    --des "base_only_H${H}" --out_dir "$OUT_DIR" \
    --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
    --shared_base_init_in "$SHARED_BASE" \
    > "$LOG" 2>&1
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 base_only H${H}"; return 1; fi
  echo "[ok] Stage-2 base_only H${H}"

  echo "[oracle_scratch01] ===== HORIZON H${H} COMPLETE ====="
}

run_horizon 96 "$ETTH1_S1_96" || { echo "[oracle_scratch01] H96 FAILED, stopping"; exit 1; }
run_horizon 720 "$ETTH1_S1_720" || { echo "[oracle_scratch01] H720 FAILED, stopping"; exit 1; }

echo "[oracle_scratch01] ALL DONE $(date -Is)"
