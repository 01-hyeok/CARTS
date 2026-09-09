#!/usr/bin/env bash
# EXP-ORACLE-RANK-GAIN01: 8-arm (Individual/Set x TF/On-policy x Cosine/
# Asymmetric) scratch Stage-1, Epoch0-vs-Best ranking gain, across ETTh1
# and Weather, H96 and H720. Order: ETTh1 H96 -> ETTh1 H720 -> Weather H96
# -> Weather H720 (spec section 2). Each cell's 8 arms share ONE freshly
# generated encoder init (never reused from EXP-ORACLE-SCRATCH01/-TF01,
# per the user's explicit choice to retrain everything for a strict,
# consistent-init comparison). Stage-2 hyperparameters are the project's
# own established defaults (lr=0.001/epochs=10/patience=5) -- not invented.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/exp_oracle_rank_gain01"
CKPT_ROOT="checkpoints/exp_oracle_rank_gain01"
RESULT_ROOT="results/EXP-ORACLE-RANK-GAIN01"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

ETTH1_S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
ETTH1_S1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
WEATHER_S1_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
WEATHER_S1_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"

run_cell() {
  local DS="$1" H="$2" REF="$3"
  echo "=================================================================="
  echo "[oracle_rank_gain01] ===== ${DS} H${H} ====="
  echo "=================================================================="
  local SHARED_ENC="$LOG_ROOT/shared_encoder_init_${DS}_H${H}.pth"
  local SHARED_BASE="$LOG_ROOT/shared_base_init_${DS}_H${H}.pth"
  local SHARED_GATE="$LOG_ROOT/shared_gate_init_${DS}_H${H}.pth"
  local RCELL="$RESULT_ROOT/${DS}/H${H}"
  mkdir -p "$RCELL"
  local ARMS=("individual tf cosine" "individual tf asymmetric" "individual onpolicy cosine" "individual onpolicy asymmetric"
              "set tf cosine" "set tf asymmetric" "set onpolicy cosine" "set onpolicy asymmetric")

  # ---- STAGE 1: 8 arms, first saves the shared encoder init ----
  local FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET PREFIX SCORER <<< "$ARM"
    local MODEL_ID="carts_rank_gain01_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}"
    local LOG="$LOG_ROOT/stage1_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}.log"
    echo "[oracle_rank_gain01] --- Stage-1 ${DS} H${H} target=${TARGET} prefix=${PREFIX} scorer=${SCORER} ---"
    if [ "$FIRST" -eq 1 ]; then
      python -u scripts/train_oracle_rank_gain01.py \
        --reference_ckpt "$REF" --target "$TARGET" --prefix_policy "$PREFIX" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_out "$SHARED_ENC" \
        > "$LOG" 2>&1
      FIRST=0
    else
      python -u scripts/train_oracle_rank_gain01.py \
        --reference_ckpt "$REF" --target "$TARGET" --prefix_policy "$PREFIX" --scorer "$SCORER" --pred_len "$H" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" \
        --train_epochs 10 --patience 5 --chunk_size 4096 --seed 0 \
        --shared_init_in "$SHARED_ENC" \
        > "$LOG" 2>&1
    fi
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] Stage-1 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"; return 1; fi
    echo "[ok] Stage-1 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"
  done

  # ---- EVAL: Epoch0 AND Best, per-step diagnostics + free-running cache, 8 arms ----
  for ARM in "${ARMS[@]}"; do
    read -r TARGET PREFIX SCORER <<< "$ARM"
    local MODEL_ID="carts_rank_gain01_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}"
    local S1_DIR="$CKPT_ROOT/${DS}/seq${H}_pred${H}/${MODEL_ID}"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}"
    local ARM_OUT="$RCELL/${TARGET}_${PREFIX}_${SCORER}"
    mkdir -p "$ARM_OUT"

    echo "[oracle_rank_gain01] --- Eval EPOCH0 ${DS} H${H} ${TARGET}/${PREFIX}/${SCORER} ---"
    python -u scripts/eval_oracle_rank_gain01.py \
      --reference_ckpt "$REF" --stage1_checkpoint "${S1_DIR}/checkpoint_epoch0.pth" \
      --top_k 10 --chunk_size 4096 --n_diag_queries 500 \
      --out "$ARM_OUT/diagnostics_epoch0.json" \
      > "$LOG_ROOT/eval_epoch0_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}.log" 2>&1
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval epoch0 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"; return 1; fi
    echo "[ok] eval epoch0 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"

    echo "[oracle_rank_gain01] --- Eval BEST ${DS} H${H} ${TARGET}/${PREFIX}/${SCORER} (+ retrieval cache) ---"
    python -u scripts/eval_oracle_rank_gain01.py \
      --reference_ckpt "$REF" --stage1_checkpoint "${S1_DIR}/checkpoint.pth" \
      --top_k 10 --chunk_size 4096 --n_diag_queries 500 \
      --out "$ARM_OUT/diagnostics_best.json" --cache_dir "$CACHE_DIR" \
      > "$LOG_ROOT/eval_best_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}.log" 2>&1
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] eval best ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"; return 1; fi
    echo "[ok] eval best ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"
  done

  # ---- STAGE 2: 8 retrieval arms + 1 no-retrieval Base-only control ----
  # (reuses scripts/train_oracle_scratch01_stage2.py UNCHANGED, per spec section 21)
  local ENC_IN=7
  if [ "$DS" = "Weather" ]; then ENC_IN=21; fi
  FIRST=1
  for ARM in "${ARMS[@]}"; do
    read -r TARGET PREFIX SCORER <<< "$ARM"
    local CACHE_DIR="$CKPT_ROOT/retrieval_cache/${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}"
    local MODEL_ID="carts_rank_gain01_stage2_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}"
    local OUT_DIR="$RCELL/${TARGET}_${PREFIX}_${SCORER}/stage2"
    local LOG="$LOG_ROOT/stage2_${TARGET}_${PREFIX}_${SCORER}_${DS}_H${H}.log"
    echo "[oracle_rank_gain01] --- Stage-2 ${DS} H${H} ${TARGET}/${PREFIX}/${SCORER} ---"
    if [ "$FIRST" -eq 1 ]; then
      python -u scripts/train_oracle_scratch01_stage2.py \
        --retrieval_cache_dir "$CACHE_DIR" --seq_len "$H" --pred_len "$H" --enc_in "$ENC_IN" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
        --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
        --shared_base_init_out "$SHARED_BASE" --shared_gate_init_out "$SHARED_GATE" \
        > "$LOG" 2>&1
      FIRST=0
    else
      python -u scripts/train_oracle_scratch01_stage2.py \
        --retrieval_cache_dir "$CACHE_DIR" --seq_len "$H" --pred_len "$H" --enc_in "$ENC_IN" \
        --checkpoints "$CKPT_ROOT" --model_id "$MODEL_ID" --des "$MODEL_ID" --out_dir "$OUT_DIR" \
        --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
        --shared_base_init_in "$SHARED_BASE" --shared_gate_init_in "$SHARED_GATE" \
        > "$LOG" 2>&1
    fi
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"; return 1; fi
    echo "[ok] Stage-2 ${TARGET}/${PREFIX}/${SCORER} ${DS} H${H}"
  done

  local BASEONLY_CACHE="$CKPT_ROOT/retrieval_cache/individual_tf_cosine_${DS}_H${H}"
  local OUT_DIR="$RCELL/base_only"
  local LOG="$LOG_ROOT/stage2_base_only_${DS}_H${H}.log"
  echo "[oracle_rank_gain01] --- Stage-2 ${DS} H${H} Base Forecaster only ---"
  python -u scripts/train_oracle_scratch01_stage2.py \
    --retrieval_cache_dir "$BASEONLY_CACHE" --no_retrieval --seq_len "$H" --pred_len "$H" --enc_in "$ENC_IN" \
    --checkpoints "$CKPT_ROOT" --model_id "carts_rank_gain01_stage2_base_only_${DS}_H${H}" \
    --des "base_only_${DS}_H${H}" --out_dir "$OUT_DIR" \
    --train_epochs 10 --patience 5 --lr 0.001 --batch_size 32 --seed 0 \
    --shared_base_init_in "$SHARED_BASE" \
    > "$LOG" 2>&1
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FAIL] stage2 base_only ${DS} H${H}"; return 1; fi
  echo "[ok] Stage-2 base_only ${DS} H${H}"

  echo "[oracle_rank_gain01] ===== ${DS} H${H} COMPLETE ====="
}

run_cell ETTh1 96 "$ETTH1_S1_96" || { echo "[oracle_rank_gain01] ETTh1 H96 FAILED, stopping"; exit 1; }
run_cell ETTh1 720 "$ETTH1_S1_720" || { echo "[oracle_rank_gain01] ETTh1 H720 FAILED, stopping"; exit 1; }
run_cell Weather 96 "$WEATHER_S1_96" || { echo "[oracle_rank_gain01] Weather H96 FAILED, stopping"; exit 1; }
run_cell Weather 720 "$WEATHER_S1_720" || { echo "[oracle_rank_gain01] Weather H720 FAILED, stopping"; exit 1; }

echo "[oracle_rank_gain01] ALL DONE $(date -Is)"
