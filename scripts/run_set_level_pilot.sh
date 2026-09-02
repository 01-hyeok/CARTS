#!/usr/bin/env bash
# Step 1: does supervising the aggregate, not the individuals, help?
#
# S0 is the existing weighted Top-K cross-entropy. S1 adds the full-memory
# set term. Both are checkpointed on validation hard_aggregate_mse10 -- the
# error of the one aggregate Stage-2 builds -- so neither arm is selected on
# its own objective. S0's numbers therefore will not match the earlier WCE
# runs, which were selected on retrieved_mse10.
#
# tau_set comes from the Step-0 support sweep on this horizon: 0.015 gives
# effective support ~32 with ~69% of the mass on the Top-10, so the relaxation
# still concentrates where deployment looks while every candidate keeps a
# gradient. Support regularisation stays off until this shows it is needed.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PRED="${PRED:-336}"; SEED="${SEED:-0}"; TAU_SET="${TAU_SET:-0.015}"
LAMBDAS=(${LAMBDAS:-1 10 30 50})
DIR="logs/set_level/ETTh1/pred${PRED}"; mkdir -p "${DIR}"

stage1(){
  local short="$1"; shift
  local log="${DIR}/${short}.log" marker="${DIR}/${short}.done"
  [ -f "${marker}" ] && { echo "[skip] ${short}"; return; }
  { echo "### RUN ${short} $(date -Is)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "carts_set_${short}_ETTh1_${PRED}" --model RelationStage1 \
      --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
      --enc_in 7 --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 5 \
      --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_coverage_top_k 10 \
      --stage1_full_memory_gradient_mode full_online \
      --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
      "$@" --des "set_${short}_ETTh1_sl${PRED}_pl${PRED}" || exit 21
    echo "### RUN COMPLETE ${short} $(date -Is)"
  } 2>&1 | tee "${log}"
  grep -q '### RUN COMPLETE' "${log}" && touch "${marker}"
}

echo "=== S0: WCE only (baseline, re-selected on hard_aggregate_mse10) ==="
stage1 s0_wce --stage1_loss_mode weighted_topk_ce

for LAM in "${LAMBDAS[@]}"; do
  echo "=== S1: WCE + SetMSE  lambda=${LAM}  tau_set=${TAU_SET} ==="
  stage1 "s1_set_lam${LAM}" --stage1_loss_mode wce_soft_set_mse \
    --stage1_set_mse_weight "${LAM}" --stage1_set_tau "${TAU_SET}" \
    --stage1_set_mse_normalization mean
done
echo "set-level Stage-1 pilot finished $(date -Is)"
