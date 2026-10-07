#!/usr/bin/env bash
# TRACK-V-MULTIQUERY-GENERALIZATION01 Phase B (H192/H336), Full-memory,
# ETTh1 + Weather. Runs as the THIRD concurrent GPU1 job (alongside
# TRACK-W-TIMESTAMP-FUSION01 and TRACK-V-R100-EFFICIENCY01), by
# explicit user authorization. Internal steps are strictly sequential.
#
# Step 1: train the missing soft_set_mse S0_wce Stage-1 REFERENCE
#         checkpoint for each of the 4 new cells (never existed before).
# Step 2: train the base forecaster for each new cell.
# Step 3: run the EXISTING, UNMODIFIED `run_v_one_setting01.sh`
#         (V0 TRUE-Original-KL + V1/V2/V5 SlotHeads, Full-candidate
#         support) for each new cell -- identical recipe to the
#         already-completed H96/H720 Phase A cells.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1

echo "=== Phase B Step 1: soft_set_mse S0_wce reference checkpoints (H192/H336) ==="
bash scripts/run_soft_set_mse_stage1_h192h336_s0only01.sh

find_ref_ckpt () {
  local data_dir="$1" pred="$2"
  find "checkpoints/soft_set_mse/stage1/${data_dir}/seq${pred}_pred${pred}" \
    -maxdepth 2 -name checkpoint.pth 2>/dev/null | head -1
}

declare -A DATA_DIR=( [ETTh1]=ETTh1 [Weather]=custom )
declare -A ROOT_PATH=( [ETTh1]="../Dataset/Time-Series-Library_dataset/ETT-small/" \
                      [Weather]="../Dataset/Time-Series-Library_dataset/weather/" )

for SPEC in "ETTh1 192" "ETTh1 336" "Weather 192" "Weather 336"; do
  read -r DS PRED <<< "${SPEC}"
  CELL="${DS}_${PRED}"
  REF="$(find_ref_ckpt "${DATA_DIR[$DS]}" "${PRED}")"
  if [ -z "${REF}" ]; then
    echo "[ISSUE][ABORT] no Stage-1 reference checkpoint found for ${CELL} under " \
        "checkpoints/soft_set_mse/stage1/${DATA_DIR[$DS]}/seq${PRED}_pred${PRED}/"
    exit 1
  fi
  echo "=== Phase B Step 2: ${CELL} base forecaster (ref=${REF}) ==="
  BASE_ROOT="results/TRACK-V-MULTIQUERY-GENERALIZATION01/${DS}/H${PRED}/seed0"
  BASE_CKROOT="checkpoints/track_v_multiquery_generalization01/${DS}/H${PRED}/seed0"
  BASE_CKPT="${BASE_CKROOT}/base/checkpoint.pth"
  if [ ! -f "${BASE_CKPT}" ]; then
    python -u scripts/train_r_base_forecaster01.py \
      --reference_ckpt "${REF}" --pred_len "${PRED}" --seq_len "${PRED}" --seed 0 \
      --out_dir "${BASE_ROOT}/base" --checkpoint_path "${BASE_CKPT}"
  else
    echo "---- [${CELL}] base forecaster already exists, skipping ----"
  fi

  echo "=== Phase B Step 3: ${CELL} V0/V1/V2/V5 Full-memory (reused unmodified run_v_one_setting01.sh) ==="
  bash scripts/run_v_one_setting01.sh "${DS}" "${REF}" "${PRED}" 0 "${BASE_CKPT}"
done

echo "=== TRACK-V-MULTIQUERY-GENERALIZATION01 Phase B (H192/H336, all 4 cells) complete ==="
