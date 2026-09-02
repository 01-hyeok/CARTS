#!/usr/bin/env bash
# Does sparse mining, not learnability, limit the ranking signal?
#
# rank_only isolates the ranking objective; pool_end stays at 100 so only the
# supervision density inside the same region changes. Everything else -- init,
# seed, optimiser, encoder, scorer -- is held, so the two arms differ in how
# many distinct better-than-selected candidates get supervised at all.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_TRAIN_DIAG=1 CARTS_SCORE_GRAD_DIAG=1 CARTS_TRAIN_DIAG_QUERIES=256
DIR=logs/coverage_probe; mkdir -p "$DIR"
CK=$(ls -d ./checkpoints/stage1/ETTh1/seq96_pred96/*e2_cos_weighted_topk_ce_ETTh1*/ | head -1)checkpoint.pth
run(){
  local tag="$1" mode="$2" pairs="$3"
  local log="${DIR}/${tag}.log"
  [ -s "$log" ] && grep -q 'Epoch 3 Vali' "$log" && { echo "[skip] $tag"; return; }
  echo "=== ${tag}: mode=${mode} pairs=${pairs} ==="
  python -u run.py --task_name stage1_relation --is_training 1 \
    --model_id "carts_cov_${tag}_ETTh1_96" --model RelationStage1 --data ETTh1 \
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv --features M \
    --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 --patch_len 16 --stride 16 --seed 0 \
    --candidate_mask raft --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_encoder_type mlp --relation_self_fill linear \
    --learning_rate 1e-3 --train_epochs 3 --patience 5 \
    --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
    --teacher_mse_space normalized --stage1_teacher_mode mse \
    --stage1_loss_mode rank_only --stage1_coverage_top_k 10 \
    --stage1_full_memory_gradient_mode full_online \
    --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
    --rank_loss_weight 1.0 --rank_margin 0.01 --rank_pool_end 100 \
    --rank_pairs_per_query "$pairs" --rank_mining_mode "$mode" --rank_gap_weighted 1 \
    --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
    --des "cov_${tag}_ETTh1_sl96_pl96" > "$log" 2>&1
  grep -q 'Epoch 3 Vali' "$log" && echo "[ok] $tag" || echo "[FAIL] $tag"
}
run R32 pair 32
run Coverage candidate 64
echo "coverage probe finished $(date -Is)"
