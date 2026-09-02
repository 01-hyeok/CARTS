#!/usr/bin/env bash
# Measure the encoder gradient each term contributes before any training run.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_GRAD_PROBE=1 CARTS_GRAD_PROBE_BATCHES="${BATCHES:-4}" CARTS_GRAD_PROBE_SHARE="${SHARE:-0.1}"
OUT=logs/rank_probe; mkdir -p "$OUT"
for PRED in ${PREDS:-96 720}; do
  CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_weighted_topk_ce_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  echo "=== ETTh1 pred${PRED}  ckpt=${CK} ==="
  python -u run.py --task_name stage1_relation --is_training 1 \
    --model_id "probe_rank_ETTh1_${PRED}" --model RelationStage1 \
    --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv --features M \
    --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
    --enc_in 7 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
    --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
    --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_encoder_type mlp --relation_self_fill linear \
    --learning_rate 1e-3 --train_epochs 1 --patience 5 \
    --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
    --teacher_mse_space normalized --stage1_teacher_mode mse \
    --stage1_loss_mode weighted_topk_ce --stage1_coverage_top_k 10 \
    --stage1_full_memory_gradient_mode full_online \
    --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
    --rank_loss_weight 1.0 --rank_margin 0.01 \
    --rank_pool_end 100 --rank_pairs_per_query 32 --rank_gap_weighted 1 \
    --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
    --des "probe_rank_ETTh1_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "\[probe\]|rank_positive_outside|Traceback|Error" | head -12
done
