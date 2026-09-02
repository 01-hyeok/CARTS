#!/usr/bin/env bash
# Oracle diagnostic only: no training, no loss changes. Four selections over the
# frozen cosine Top-100, so pool coverage is held identical and only the choice
# of ten differs. The encoder is the WCE checkpoint and stays frozen; the
# retrieval metric is left at cosine so the baseline needs no metric weights.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_SETORACLE_DIAG=1 CARTS_SETORACLE_BATCHES="${BATCHES:-4}"
export CARTS_SETORACLE_POOL="${POOLS:-50,100,200,500,full}"
export CARTS_SETORACLE_K="${KS:-1,3,5,10,20}"
export CARTS_SETORACLE_GOOD=30
export CARTS_SETORACLE_OUT=logs/set_oracle
mkdir -p "$CARTS_SETORACLE_OUT"
for PRED in ${PREDS:-96 720}; do
  CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_weighted_topk_ce_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] pred${PRED}"; continue; }
  echo "=== ETTh1 pred${PRED}  ckpt=$(basename $(dirname $CK)) ==="
  CARTS_SETORACLE_TAG="pred${PRED}" python -u run.py \
    --task_name stage1_relation --is_training 0 --model RelationStage1 --data ETTh1 \
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
    --model_id "carts_e2_cos_weighted_topk_ce_ETTh1_${PRED}" \
    --des "e2_cos_weighted_topk_ce_ETTh1_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "\[setoracle\]|Traceback|Error"
done
