#!/usr/bin/env bash
# Step D: does pushing individual candidate quality harder cost candidate spread?
#
# Expected future MSE is linear in the student distribution, so its minimiser
# puts all the mass on the single lowest-error candidate. That makes it a
# stronger version of exactly the pressure the Top-K cross-entropy already
# applies, and the registered prediction is that it improves individual quality
# and Recall while lowering spread and, at the long horizon, worsening the
# aggregate that Stage-2 consumes. A result the other way would refute the
# redundancy reading, which is why it is worth running.
#
# lambda=12 comes from the measured raw scales (WCE 7.56, normalised expected
# MSE 0.62), not from a guess. H96 runs as the contrast: the redundancy signal
# is strongest at H720, so a mechanism claim needs the short horizon too.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
DIR=logs/expected_mse; mkdir -p "$DIR"
LAM="${LAM:-12}"

stage1(){
  local pred="$1" short="$2"; shift 2
  local log="${DIR}/ETTh1_pred${pred}_${short}.log"
  [ -f "${DIR}/ETTh1_pred${pred}_${short}.done" ] && { echo "[skip] $pred/$short"; return; }
  { echo "### RUN ${pred}/${short} $(date -Is)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "carts_exp_${short}_ETTh1_${pred}" --model RelationStage1 \
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
      --stage1_coverage_top_k 10 --stage1_full_memory_gradient_mode full_online \
      --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
      "$@" --des "exp_${short}_ETTh1_sl${pred}_pl${pred}" || exit 21
    echo "### RUN COMPLETE ${pred}/${short} $(date -Is)"; } 2>&1 | tee "$log"
  grep -q '### RUN COMPLETE' "$log" && touch "${DIR}/ETTh1_pred${pred}_${short}.done"
}

stage2(){
  local pred="$1" short="$2"
  local ck=$(ls -d ./checkpoints/stage1/ETTh1/seq${pred}_pred${pred}/*exp_${short}_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$ck" ] || { echo "[skip s2] no ckpt $pred/$short"; return; }
  local log="${DIR}/ETTh1_pred${pred}_${short}_stage2.log"
  [ -f "${DIR}/ETTh1_pred${pred}_${short}_stage2.done" ] && return
  { echo "### RUN stage2 ${pred}/${short} $(date -Is)"
    python -u run.py --task_name stage2_relation --is_training 1 \
      --model_id "carts_exp_s2_${short}_ETTh1_${pred}" --model RelationStage2 \
      --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len "$pred" --label_len 0 --pred_len "$pred" \
      --enc_in 7 --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 3 \
      --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
      --stage1_ckpt_path "$ck" --stage1_encoder_init checkpoint \
      --freeze_stage1_encoder 1 --stage2_e2e 0 \
      --des "exp_s2_${short}_ETTh1_sl${pred}_pl${pred}" || exit 21
    echo "### RUN COMPLETE stage2 ${pred}/${short} $(date -Is)"; } 2>&1 | tee "$log"
  grep -q '### RUN COMPLETE' "$log" && touch "${DIR}/ETTh1_pred${pred}_${short}_stage2.done"
}

for PRED in 720 96; do
  stage1 "$PRED" L0_wce      --stage1_loss_mode weighted_topk_ce
  stage1 "$PRED" L1_expmse   --stage1_loss_mode expected_mse
  stage1 "$PRED" L2_wce_exp  --stage1_loss_mode wce_expected_mse --stage1_expected_mse_lambda "$LAM"
  for S in L0_wce L1_expmse L2_wce_exp; do stage2 "$PRED" "$S"; done
done
echo "expected-mse arms finished $(date -Is)"
