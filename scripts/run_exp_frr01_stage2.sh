#!/usr/bin/env bash
# EXP-FRR01 Stage-2 downstream evaluation: R0/R1/R2/R12/R3, ETTh1 H96 only.
#
# Identical Stage-2 protocol to B0 (EXP-3's S0_wce Stage-2: architecture, base
# forecaster, fusion=residual/scalar-gate, top_k=10, tau_topk=0.1,
# candidate_mask=raft, freeze_stage1_encoder=1, stage2_e2e=0, seed=0) -- only
# the Stage-1 checkpoint and the per-arm conditioning flags differ:
#
#   R0   plain load, no extra flags (residual teacher only shaped training)
#   R1   --stage2_query_base_conditioning 1 --stage2_residual_cache ...
#   R2   --stage2_candidate_residual_conditioning 1 --stage2_residual_cache ...
#   R12  both conditioning flags
#   R3   R12 flags + --stage1_retrieval_metric asymmetric (must match Stage-1)
#
# Same expected/executed/completed/skipped/failed safety-guard convention as
# scripts/run_soft_set_mse_stage2.sh, and the same checkpoint-path bug that
# script's history warns about (this one hardcodes the exact known dir instead
# of a glob, since EXP-FRR01 only ever has one Stage-1 run per arm).
set -uo pipefail
LOCK=/tmp/carts_exp_frr01_stage2.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] another run_exp_frr01_stage2.sh instance holds the lock"; exit 3; }

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

STAGE1_ROOT="checkpoints/exp_frr01/stage1/ETTh1/seq96_pred96"
LOG_ROOT="${LOG_ROOT:-logs/exp_frr01}"
RESIDUAL_CACHE="${RESIDUAL_CACHE:-cache/residual_teacher_frr01}"
mkdir -p "$LOG_ROOT"

ARMS="${ARMS:-R0 R1 R2 R12 R3}"

common_args=(
  --task_name stage2_relation --is_training 1 --model RelationStage2
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
  --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar
  --stage1_encoder_init checkpoint --freeze_stage1_encoder 1 --stage2_e2e 0
  --checkpoints checkpoints/exp_frr01
)

expected=0 executed=0 completed=0 skipped_done=0 skipped_missing=0 failed=0

for ARM in $ARMS; do
  expected=$((expected + 1))
  S1CK=$(ls -d "${STAGE1_ROOT}"/*"frr01_${ARM}_ETTh1"*/ 2>/dev/null | head -1)checkpoint.pth
  if [ ! -f "$S1CK" ]; then
    echo "[skip:MISSING_CHECKPOINT] ${ARM}: no Stage-1 checkpoint at ${STAGE1_ROOT}/*frr01_${ARM}_ETTh1*"
    skipped_missing=$((skipped_missing + 1))
    continue
  fi

  arm_args=()
  case "$ARM" in
    R0)  ;;
    R1)  arm_args=(--stage2_query_base_conditioning 1 --stage2_residual_cache "$RESIDUAL_CACHE") ;;
    R2)  arm_args=(--stage2_candidate_residual_conditioning 1 --stage2_residual_cache "$RESIDUAL_CACHE") ;;
    R12) arm_args=(--stage2_query_base_conditioning 1 --stage2_candidate_residual_conditioning 1
                    --stage2_residual_cache "$RESIDUAL_CACHE") ;;
    R3)  arm_args=(--stage2_query_base_conditioning 1 --stage2_candidate_residual_conditioning 1
                    --stage2_residual_cache "$RESIDUAL_CACHE" --stage1_retrieval_metric asymmetric) ;;
    *) echo "[FAIL] unknown arm $ARM"; failed=$((failed + 1)); continue ;;
  esac

  MODEL_ID="carts_frr01_s2_ETTh1_96_${ARM}"
  LOG="$LOG_ROOT/${ARM}_stage2.log"; MARKER="$LOG_ROOT/${ARM}_stage2.done"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$MARKER" ]; then
    echo "[skip:already_done] ${ARM}_stage2"
    skipped_done=$((skipped_done + 1))
    continue
  fi
  executed=$((executed + 1))
  echo "=============================================================="
  echo "[${ARM}] stage2 EXP-FRR01, stage1_ckpt=${S1CK}"
  echo "=============================================================="
  {
    echo "### RUN ${ARM}_stage2 $(date -Is)"
    echo "### git $(git rev-parse HEAD)"
    echo "### stage1_ckpt ${S1CK}"
    python -u run.py "${common_args[@]}" "${arm_args[@]}" \
      --stage1_ckpt_path "$S1CK" \
      --model_id "$MODEL_ID" \
      --des "frr01_s2_${ARM}_ETTh1_sl96_pl96" || exit 21
    echo "### RUN COMPLETE ${ARM}_stage2 $(date -Is)"
  } 2>&1 | tee "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "$LOG"; then
    touch "$MARKER"; echo "[ok] ${ARM}_stage2"
    completed=$((completed + 1))
  else
    echo "[FAIL] ${ARM}_stage2"
    failed=$((failed + 1))
  fi
done

echo "=============================================================="
echo "[summary] expected=${expected} executed=${executed} completed=${completed} " \
     "skipped_already_done=${skipped_done} skipped_missing_checkpoint=${skipped_missing} failed=${failed}"
echo "EXP-FRR01 Stage-2 pilot finished $(date -Is)"

if [ "$skipped_missing" -gt 0 ] || { [ "$expected" -gt 0 ] && [ "$executed" -eq 0 ] && [ "$skipped_done" -eq 0 ]; } || [ "$failed" -gt 0 ]; then
  echo "[summary] FAIL: missing_checkpoint=${skipped_missing} failed=${failed} executed=${executed}"
  exit 1
fi
echo "[summary] OK"
