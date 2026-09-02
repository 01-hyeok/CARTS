#!/usr/bin/env bash
# Phase 0: choose the softmax temperature per horizon before any soft-set loss.
# Reads the score distribution only -- no future labels -- on train queries, with
# the encoder frozen and the scorer at its identity initialisation, so every arm
# starts from the same geometry.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_TAUCAL_DIAG=1 CARTS_TAUCAL_BATCHES="${BATCHES:-4}"
DIR=logs/tau_calibration; mkdir -p "$DIR"
for PRED in ${PREDS:-96 192 336 720}; do
  CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_weighted_topk_ce_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] pred${PRED}"; continue; }
  python -u run.py --task_name stage1_relation --is_training 0 \
    --model RelationStage1 --data ETTh1 \
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv --features M \
    --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
    --enc_in 7 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
    --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
    --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_encoder_type mlp --relation_self_fill linear \
    --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
    --teacher_mse_space normalized --stage1_teacher_mode mse \
    --stage1_loss_mode weighted_topk_ce --stage1_coverage_top_k 10 \
    --stage1_full_memory_gradient_mode full_online --stage1_probe_vis 0 \
    --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
    --stage1_retrieval_metric asymmetric --stage1_metric_output cosine \
    --stage1_metric_layer_norm 0 --stage1_freeze_encoder 1 \
    --model_id "carts_e2_cos_weighted_topk_ce_ETTh1_${PRED}" \
    --des "e2_cos_weighted_topk_ce_ETTh1_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "\[taucal\]|cosine_init_deviation|Traceback|Error" | tee "$DIR/pred${PRED}.log"
done
