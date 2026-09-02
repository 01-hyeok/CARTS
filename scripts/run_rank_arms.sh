#!/usr/bin/env bash
# ETTh1 WCE + boundary hard-pair rank. Baseline is the existing WCE checkpoint,
# so only the two proposed arms are trained.
#
# lambda comes from the measured gradient ratio, not the loss values: the hinge
# and the cross-entropy are not on one scale, and what decides the balance is
# what reaches the encoder. Solved for a 10% gradient share per horizon.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
DIR=logs/rank_arms; mkdir -p "$DIR"
run(){
  local pred="$1" lam="$2"
  local log="${DIR}/ETTh1_pred${pred}_wce_rank.log"
  [ -f "${DIR}/ETTh1_pred${pred}.done" ] && { echo "[skip] $pred"; return; }
  { echo "### RUN ETTh1/pred${pred} lambda=${lam} $(date -Is)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "carts_rank_ETTh1_${pred}" --model RelationStage1 \
      --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len "$pred" --label_len 0 --pred_len "$pred" \
      --enc_in 7 --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 5 \
      --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_loss_mode weighted_topk_ce --stage1_coverage_top_k 10 \
      --stage1_full_memory_gradient_mode full_online \
      --rank_loss_weight "$lam" --rank_margin 0.01 \
      --rank_pool_end 100 --rank_pairs_per_query 32 --rank_gap_weighted 1 \
      --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
      --des "rank_ETTh1_sl${pred}_pl${pred}" || exit 21
    echo "### RUN COMPLETE ETTh1/pred${pred} $(date -Is)"; } 2>&1 | tee "$log"
  grep -q '### RUN COMPLETE' "$log" && touch "${DIR}/ETTh1_pred${pred}.done"
}
run 96 "${LAM96:-1.17}"
run 720 "${LAM720:-2.26}"
echo "rank arms finished $(date -Is)"
