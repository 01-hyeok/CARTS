#!/usr/bin/env bash
# Solar canonical Stage-2 S0_wce host, built via the EXACT SAME pipeline as
# scripts/run_soft_set_mse_stage2.sh, pointed at the Solar S0_wce Stage-1
# checkpoint produced by scripts/run_solar_softset_host_stage1.sh. This is
# the frozen Stage-2 host used purely as the fixed exogenous aggregation-
# weighting score function for Greedy Set Oracle / FR-Agg in
# TRACK-A-TF-ORACLE-LEARNABILITY01 -- not a subject of study itself.
#
# Separate script (not a modification of run_soft_set_mse_stage2.sh) so
# ETTh1/Weather's existing canonical pipeline is untouched. Flags copied
# verbatim from run_soft_set_mse_stage2.sh; only the dataset spec and the
# Solar relation graph are Solar-specific.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

STAGE1_ROOT="${STAGE1_ROOT:-checkpoints/soft_set_mse/stage1}"
LOG_ROOT="${LOG_ROOT:-logs/soft_set_mse}"
mkdir -p "$LOG_ROOT"

ARM=S0_wce
SPECS='Solar 96  137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt
Solar 720 137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt'

expected=0 executed=0 completed=0 skipped_done=0 skipped_missing=0 failed=0

while read -r DS PRED ENC DKEY ROOT DPATH; do
  [ -n "${DS:-}" ] || continue
  GRAPH_ARGS=(--relation_graph_path metrics/relation_graphs/solar/pearson_self_top1.json)
  DIR="$LOG_ROOT/${DS}/pred${PRED}"; mkdir -p "$DIR"

  expected=$((expected + 1))
  S1CK=$(ls -d "${STAGE1_ROOT}/${DKEY}/seq${PRED}_pred${PRED}"/*"carts_softset_${DS}_${PRED}_${ARM}"*/ 2>/dev/null | head -1)checkpoint.pth
  if [ ! -f "$S1CK" ]; then
    echo "[skip:MISSING_CHECKPOINT] ${DS}/pred${PRED}/${ARM}: no Stage-1 checkpoint at ${STAGE1_ROOT}/${DKEY}/seq${PRED}_pred${PRED}"
    skipped_missing=$((skipped_missing + 1))
    continue
  fi
  LOG="$DIR/${ARM}_stage2.log"; MARKER="$DIR/${ARM}_stage2.done"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$MARKER" ]; then
    echo "[skip:already_done] ${DS}/pred${PRED}/${ARM}_stage2"
    skipped_done=$((skipped_done + 1))
    continue
  fi
  executed=$((executed + 1))
  echo "=============================================================="
  echo "[${DS}/pred${PRED}/${ARM}] stage2 downstream, stage1_ckpt=${S1CK}"
  echo "=============================================================="
  {
    echo "### RUN ${DS}/pred${PRED}/${ARM}_stage2 $(date -Is)"
    echo "### git $(git rev-parse HEAD)"
    echo "### stage1_ckpt ${S1CK}"
    python -u run.py --task_name stage2_relation --is_training 1 \
      --model_id "carts_softset_s2_${DS}_${PRED}_${ARM}" --model RelationStage2 \
      --data "$DKEY" --root_path "$ROOT" --data_path "$DPATH" --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" --enc_in "$ENC" \
      --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      "${GRAPH_ARGS[@]}" \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 5 \
      --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
      --stage1_ckpt_path "$S1CK" --stage1_encoder_init checkpoint \
      --freeze_stage1_encoder 1 --stage2_e2e 0 \
      --oracle_candidate_eval 1 \
      --des "softset_s2_${ARM}_${DS}_sl${PRED}_pl${PRED}" || exit 21
    echo "### RUN COMPLETE ${DS}/pred${PRED}/${ARM}_stage2 $(date -Is)"
  } 2>&1 | tee "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "$LOG"; then
    touch "$MARKER"; echo "[ok] ${DS}/pred${PRED}/${ARM}_stage2"
    completed=$((completed + 1))
  else
    echo "[FAIL] ${DS}/pred${PRED}/${ARM}_stage2"
    failed=$((failed + 1))
  fi
done < <(printf '%s\n' "${SPECS_OVERRIDE:-$SPECS}")

echo "=============================================================="
echo "[summary] expected=${expected} executed=${executed} completed=${completed} "\
"skipped_already_done=${skipped_done} skipped_missing_checkpoint=${skipped_missing} failed=${failed}"
echo "Solar softset Stage-2 (S0_wce only) finished $(date -Is)"

if [ "$skipped_missing" -gt 0 ] || { [ "$expected" -gt 0 ] && [ "$executed" -eq 0 ] && [ "$skipped_done" -eq 0 ]; } || [ "$failed" -gt 0 ]; then
  echo "[summary] FAIL: missing_checkpoint=${skipped_missing} failed=${failed} executed=${executed}"
  exit 1
fi
echo "[summary] OK"
