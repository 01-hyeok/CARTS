#!/usr/bin/env bash
# When does the representation collapse, and does the objective or the mining
# cause it? One epoch from the same WCE checkpoint, three arms, geometry read on
# one fixed query/candidate subset so the numbers are comparable across arms and
# steps. Nothing is tuned here.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_COLLAPSE_PROBE=1 CARTS_COLLAPSE_QUERIES=64 CARTS_COLLAPSE_CANDIDATES=1024
DIR=logs/collapse_diag; mkdir -p "$DIR"
CK=$(ls -d ./checkpoints/stage1/ETTh1/seq96_pred96/*e2_cos_weighted_topk_ce_ETTh1*/ | head -1)checkpoint.pth
run(){
  local tag="$1" loss="$2" mode="$3"
  local log="${DIR}/${tag}.log"
  [ -s "$log" ] && grep -q 'collapse. epoch1' "$log" && { echo "[skip] $tag"; return; }
  echo "=== ${tag}: loss=${loss} mining=${mode} ==="
  python -u run.py --task_name stage1_relation --is_training 1 \
    --model_id "carts_col_${tag}_ETTh1_96" --model RelationStage1 --data ETTh1 \
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv --features M \
    --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 --patch_len 16 --stride 16 --seed 0 \
    --candidate_mask raft --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_encoder_type mlp --relation_self_fill linear \
    --learning_rate 1e-3 --train_epochs 1 --patience 5 \
    --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
    --teacher_mse_space normalized --stage1_teacher_mode mse \
    --stage1_loss_mode "$loss" --stage1_coverage_top_k 10 \
    --stage1_full_memory_gradient_mode full_online \
    --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
    --rank_loss_weight 1.0 --rank_margin 0.01 --rank_pool_end 100 \
    --rank_pairs_per_query 32 --rank_mining_mode "$mode" --rank_gap_weighted 1 \
    --stage1_checkpoint_metric hard_aggregate_mse10 --stage1_probe_vis 0 \
    --des "col_${tag}_ETTh1_sl96_pl96" > "$log" 2>&1
  grep -q 'collapse. epoch1' "$log" && echo "[ok] $tag" || echo "[FAIL] $tag"
}
run WCE       weighted_topk_ce pair
run Dynamic   rank_only        candidate
run Persistent rank_only       persistent
echo "collapse diag finished $(date -Is)"
