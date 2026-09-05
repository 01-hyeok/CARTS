#!/usr/bin/env bash
# Stage-2 downstream evaluation for the soft_set_mse Stage-1 arms.
#
# Only the Stage-1 checkpoint changes across arms. Stage-2 architecture, base
# forecaster protocol, fusion (residual/scalar gate), top_k, tau_topk=0.1,
# query order, seed, split and candidate masking are identical for every arm --
# and identical to what run_soft_set_mse_stage1.sh trained relation_top_n=1
# self-retrieval under, so this stays inside the corrected self-retrieval
# setting and is not mixed with the cross-channel question.
#
# Standard frozen-Stage-1 Stage-2 training (stage2_e2e=0); E2E is out of scope
# for this experiment.
#
# Safety guard (added after a checkpoint-path mismatch silently skipped all 20
# runs while the script still exited 0 and logged "sweep finished"): every run
# is counted into one of completed / skipped(already done) / skipped(missing
# checkpoint) / failed, the totals are printed at the end, and a missing-
# checkpoint skip or zero executed runs is a non-zero-exit FAIL -- it is not
# treated as a quiet no-op the way an already-done skip is.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

STAGE1_ROOT="${STAGE1_ROOT:-checkpoints/soft_set_mse/stage1}"
LOG_ROOT="${LOG_ROOT:-logs/soft_set_mse}"
mkdir -p "$LOG_ROOT"

# ARM list is overridable so a representative subset (e.g. "S0_wce S3_lam30")
# can be targeted per cell without touching the arm definitions themselves.
ARMS="${ARMS:-S0_wce S1_set_only S2_lam10 S3_lam30 S4_lam50}"

SPECS='ETTh1 96  7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv
ETTh1 720 7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv
Weather 96  21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv
Weather 720 21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv'

expected=0 executed=0 completed=0 skipped_done=0 skipped_missing=0 failed=0

# Process substitution, not a pipe: `... | while read; do ...; done` runs the
# loop body in a subshell, so counters incremented inside it are invisible
# once the pipeline exits. `< <(...)` keeps the loop in this shell.
while read -r DS PRED ENC DKEY ROOT DPATH; do
  [ -n "${DS:-}" ] || continue
  GRAPH_ARGS=()
  if [ "$DS" = Weather ]; then
    GRAPH_ARGS=(--relation_graph_path metrics/relation_graphs/weather/pearson_self_top1.json)
  fi
  DIR="$LOG_ROOT/${DS}/pred${PRED}"; mkdir -p "$DIR"

  for ARM in $ARMS; do
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
  done
done < <(printf '%s\n' "${SPECS_OVERRIDE:-$SPECS}")

echo "=============================================================="
echo "[summary] expected=${expected} executed=${executed} completed=${completed} "\
"skipped_already_done=${skipped_done} skipped_missing_checkpoint=${skipped_missing} failed=${failed}"
echo "soft_set_mse Stage-2 sweep finished $(date -Is)"

# A missing-checkpoint skip or a completely empty run is exactly the failure
# mode that produced a silent, all-skipped "successful" sweep before -- treat
# both as FAIL, distinct from the benign already-done skip.
if [ "$skipped_missing" -gt 0 ] || { [ "$expected" -gt 0 ] && [ "$executed" -eq 0 ] && [ "$skipped_done" -eq 0 ]; } || [ "$failed" -gt 0 ]; then
  echo "[summary] FAIL: missing_checkpoint=${skipped_missing} failed=${failed} executed=${executed}"
  exit 1
fi
echo "[summary] OK"
