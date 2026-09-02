#!/usr/bin/env bash
# Step 1: measure the score band before choosing a ranking margin.
#
# Cosine spans [-1, 1] in theory but the learned scores occupy a much narrower
# range, so a conventional margin can exceed the whole rank-10 to rank-100 gap
# and leave the hinge open on every pair. Evaluation only, on the existing WCE
# checkpoints, so nothing is trained and no margin is assumed.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
OUT="${OUT:-logs/score_geometry}"; mkdir -p "$OUT"
for PRED in ${PREDS:-96 720}; do
  CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_weighted_topk_ce_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] ETTh1 pred${PRED}"; continue; }
  echo "=== ETTh1 pred${PRED} ==="
  python -u run.py --task_name stage1_relation --is_training 0 \
    --model_id "carts_e2_cos_weighted_topk_ce_ETTh1_${PRED}" --model RelationStage1 \
    --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
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
    --stage1_full_memory_gradient_mode full_online \
    --stage1_checkpoint_metric retrieved_mse10 --stage1_probe_vis 0 \
    --rank_pool_end 100 \
    --des "e2_cos_weighted_topk_ce_ETTh1_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "Stage1 Test \||Traceback|Error" > "${OUT}/ETTh1_pred${PRED}.log"
  grep -q "Stage1 Test" "${OUT}/ETTh1_pred${PRED}.log" && echo "[ok] pred${PRED}" || echo "[FAIL] pred${PRED}"
done
