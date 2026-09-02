#!/usr/bin/env bash
# Diagnosis only: existing checkpoints, no training. Retention is recomputed for
# every arm through the same retrieval path, including the arm trained without an
# anchor whose value was previously a formatting default rather than a measurement.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_SWAP_DIAG=1 CARTS_SWAP_BATCHES="${BATCHES:-2}"
export CARTS_SWAP_OUT=logs/swap_conflict
mkdir -p "$CARTS_SWAP_OUT"; rm -f "$CARTS_SWAP_OUT/swap_rows.csv" "$CARTS_SWAP_OUT/fingerprints.txt"
CK=$(ls -d ./checkpoints/stage1/ETTh1/seq96_pred96/*e2_cos_weighted_topk_ce_ETTh1*/ | head -1)checkpoint.pth
common=(--task_name stage1_relation --is_training 0 --model RelationStage1 --data ETTh1
  --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
  --data_path ETTh1.csv --features M
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --batch_size 32 --num_workers 0
  --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 --patch_len 16 --stride 16 --seed 0
  --candidate_mask raft --relation_input_space delta_last --relation_teacher_space delta_last
  --source_mode auto --relation_top_n 1 --target_mode all
  --relation_encoder_type mlp --relation_self_fill linear
  --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1
  --teacher_mse_space normalized --stage1_teacher_mode mse
  --stage1_loss_mode rank_only --stage1_coverage_top_k 10
  --stage1_full_memory_gradient_mode full_online
  --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint
  --stage1_freeze_encoder 1 --stage1_probe_vis 0
  --stage1_retrieval_metric asymmetric --stage1_metric_output cosine
  --stage1_metric_layer_norm 0
  --rank_loss_weight 1.0 --rank_margin 0.01 --rank_pool_end 100
  --rank_pairs_per_query 32 --rank_mining_mode candidate)

run(){  # arm beta model_id des
  echo "=== $1 (beta=$2) ==="
  CARTS_SWAP_ARM="$1" CARTS_SWAP_BETA="$2" \
  python -u run.py "${common[@]}" --model_id "$3" --des "$4" \
    --stage1_global_anchor_weight "$2" 2>&1 | grep -E "\[swap\]|Traceback|Error" | head -8
}
# The untrained scorer: identity initialisation, so it scores exactly as
# cosine and is the reference the other arms departed from.
CARTS_SWAP_ARM=cosine CARTS_SWAP_BETA=0 \
  python -u run.py "${common[@]}" --model_id carts_e2_cos_weighted_topk_ce_ETTh1_96 \
  --des e2_cos_weighted_topk_ce_ETTh1_sl96_pl96 --stage1_global_anchor_weight 0 2>&1 \
  | grep -E "\[swap\]|Traceback|Error" | head -10
run rank_b0 0    carts_frz_asym_ETTh1_96   frz_asym_ETTh1_sl96_pl96
run ga_b0.1 0.1  carts_ga_beta0p1_ETTh1_96 ga_beta0p1_ETTh1_sl96_pl96
run ga_b1.0 1.0  carts_ga_beta1p0_ETTh1_96 ga_beta1p0_ETTh1_sl96_pl96
echo "swap conflict diag finished $(date -Is)"
