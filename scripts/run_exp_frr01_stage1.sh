#!/usr/bin/env bash
# EXP-FRR01 Stage-1 pilot: R0/R1/R2/R12/R3, ETTh1 H96 only.
#
# B0 is NOT run here -- it is the existing EXP-3 S0_wce Stage-1/Stage-2
# checkpoint pair, reused as-is (see research/EXPERIMENT_LOG.md EXP-3-CLOSURE).
#
# Every arm below is the identical B0 Stage-1 recipe (encoder, optimizer,
# candidate_mask=raft, full_online gradient mode, wce_weight=1/set_mse_weight=0
# -- i.e. plain WCE, no soft_set_mse term) plus only the new EXP-FRR01 flags:
#
#   R0   residual teacher only            --stage1_residual_teacher 1
#   R1   R0 + query base conditioning     + --stage1_query_base_conditioning 1
#   R2   R0 + candidate residual repr     + --stage1_candidate_residual_conditioning 1
#   R12  R1 + R2 together                 both conditioning flags
#   R3   R12 + asymmetric dual encoder    + --stage1_retrieval_metric asymmetric
#
# The residual-teacher cache is rebuilt (not reused) from the current
# campaign's own B0 Stage-2 checkpoint -- the pre-existing
# cache/residual_teacher/ETTh1_pred96.pt was built from an older KL/raft_concat
# reference checkpoint, a provenance mismatch this run does not want to carry.
# See cache/residual_teacher_frr01/ETTh1_pred96.pt (already rebuilt this run).
#
# Single GPU (GPU 1 by default), strictly sequential, flock-guarded against a
# second concurrent launch. Same expected/executed/completed/skipped/failed
# safety-guard convention as scripts/run_soft_set_mse_stage2.sh.
set -uo pipefail
LOCK=/tmp/carts_exp_frr01_stage1.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] another run_exp_frr01_stage1.sh instance holds the lock"; exit 3; }

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

CKPT_ROOT="${CKPT_ROOT:-checkpoints/exp_frr01}"
LOG_ROOT="${LOG_ROOT:-logs/exp_frr01}"
RESIDUAL_CACHE="${RESIDUAL_CACHE:-cache/residual_teacher_frr01}"
mkdir -p "$LOG_ROOT" "$CKPT_ROOT"

if [ ! -f "${RESIDUAL_CACHE}/ETTh1_pred96.pt" ]; then
  echo "[FAIL] residual cache ${RESIDUAL_CACHE}/ETTh1_pred96.pt missing -- rebuild with " \
       "scripts/precompute_residual_teacher.py before running this driver"
  exit 4
fi

ARMS="${ARMS:-R0 R1 R2 R12 R3}"

common_args=(
  --task_name stage1_relation --is_training 1 --model RelationStage1
  --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
  --data_path ETTh1.csv --features M
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7
  --batch_size 32 --num_workers 0
  --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2
  --patch_len 16 --stride 16 --seed 0 --candidate_mask raft
  --relation_input_space delta_last --relation_teacher_space delta_last
  --source_mode auto --relation_top_n 1 --target_mode all
  --relation_encoder_type mlp --relation_self_fill linear
  --learning_rate 1e-3 --train_epochs 10 --patience 5
  --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1
  --teacher_mse_space normalized --stage1_teacher_mode mse
  --stage1_loss_mode wce_soft_set_mse --stage1_coverage_top_k 10
  --stage1_wce_weight 1.0 --stage1_set_mse_weight 0.0
  --stage1_set_tau 0.015 --stage1_set_mse_normalization mean
  --stage1_full_memory_gradient_mode full_online
  --stage1_checkpoint_metric hard_aggregate_mse10
  --stage1_probe_vis 0
  --stage1_residual_teacher 1
  --stage1_residual_teacher_cache "$RESIDUAL_CACHE"
  --checkpoints "$CKPT_ROOT"
)

expected=0 executed=0 completed=0 skipped_done=0 failed=0

for ARM in $ARMS; do
  expected=$((expected + 1))
  arm_args=()
  case "$ARM" in
    R0)  ;;
    R1)  arm_args=(--stage1_query_base_conditioning 1) ;;
    R2)  arm_args=(--stage1_candidate_residual_conditioning 1) ;;
    R12) arm_args=(--stage1_query_base_conditioning 1 --stage1_candidate_residual_conditioning 1) ;;
    R3)  arm_args=(--stage1_query_base_conditioning 1 --stage1_candidate_residual_conditioning 1
                    --stage1_retrieval_metric asymmetric) ;;
    *) echo "[FAIL] unknown arm $ARM"; failed=$((failed + 1)); continue ;;
  esac

  MODEL_ID="carts_frr01_ETTh1_96_${ARM}"
  LOG="$LOG_ROOT/${ARM}_stage1.log"; MARKER="$LOG_ROOT/${ARM}_stage1.done"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$MARKER" ]; then
    echo "[skip:already_done] ${ARM}_stage1"
    skipped_done=$((skipped_done + 1))
    continue
  fi
  executed=$((executed + 1))
  echo "=============================================================="
  echo "[${ARM}] stage1 EXP-FRR01 pilot, ETTh1 H96"
  echo "=============================================================="
  {
    echo "### RUN ${ARM}_stage1 $(date -Is)"
    echo "### git $(git rev-parse HEAD)"
    python -u run.py "${common_args[@]}" "${arm_args[@]}" \
      --model_id "$MODEL_ID" \
      --des "frr01_${ARM}_ETTh1_sl96_pl96" || exit 21
    echo "### RUN COMPLETE ${ARM}_stage1 $(date -Is)"
  } 2>&1 | tee "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "$LOG"; then
    touch "$MARKER"; echo "[ok] ${ARM}_stage1"
    completed=$((completed + 1))
  else
    echo "[FAIL] ${ARM}_stage1"
    failed=$((failed + 1))
  fi
done

echo "=============================================================="
echo "[summary] expected=${expected} executed=${executed} completed=${completed} " \
     "skipped_already_done=${skipped_done} failed=${failed}"
echo "EXP-FRR01 Stage-1 pilot finished $(date -Is)"

if [ "$failed" -gt 0 ] || { [ "$expected" -gt 0 ] && [ "$executed" -eq 0 ] && [ "$skipped_done" -eq 0 ]; }; then
  echo "[summary] FAIL: failed=${failed} executed=${executed}"
  exit 1
fi
echo "[summary] OK"
