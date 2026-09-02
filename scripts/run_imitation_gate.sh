#!/usr/bin/env bash
# Phase C gate: can a per-candidate score reproduce a set chosen for how its
# members combine? Both arms share the pool, the frozen encoder, the scorer and
# every hyperparameter; only which oracle supplies the K targets differs.
# The greedy target is not a sort, so its restart stability is measured too --
# a noisier target would make the set arm look harder for the wrong reason.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_TRAIN_DIAG=1 CARTS_TRAIN_DIAG_QUERIES=256
DIR=logs/imitation; mkdir -p "$DIR"
declare -A TAU=( [96]=0.015 [192]=0.015 [336]=0.01 [720]=0.02 )
for PRED in ${PREDS:-96 720}; do
  CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_weighted_topk_ce_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] pred${PRED}"; continue; }
  for TGT in individual set; do
    LOG="${DIR}/pred${PRED}_${TGT}.log"
    [ -s "$LOG" ] && grep -q 'Epoch 3 Vali' "$LOG" && { echo "[skip] $PRED/$TGT"; continue; }
    echo "=== ETTh1 pred${PRED}  target=${TGT}  tau=${TAU[$PRED]} ==="
    python -u run.py --task_name stage1_relation --is_training 1 --train_epochs 3 \
      --model_id "carts_imit_${TGT}_ETTh1_${PRED}" --model RelationStage1 --data ETTh1 \
      --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
      --enc_in 7 --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --patience 5 \
      --top_k 10 --tau_student "${TAU[$PRED]}" --tau_teacher 0.1 --tau_topk 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_loss_mode oracle_imitation --stage1_coverage_top_k 10 \
      --stage1_imitation_target "$TGT" --stage1_imitation_pool 100 \
      --stage1_full_memory_gradient_mode full_online \
      --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
      --stage1_freeze_encoder 1 \
      --stage1_retrieval_metric asymmetric --stage1_metric_output cosine \
      --stage1_metric_layer_norm 0 \
      --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
      --des "imit_${TGT}_ETTh1_sl${PRED}_pl${PRED}" > "$LOG" 2>&1
    grep -q 'Epoch 3 Vali' "$LOG" && echo "[ok] $PRED/$TGT" || echo "[FAIL] $PRED/$TGT"
  done
done
echo "imitation gate finished $(date -Is)"
