#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -u run.py \
  --task_name stage1_relation \
  --is_training 1 \
  --model_id CARTS_stage1_tiny_overfit_topk_coverage_mlp_linear_ETTh1_96 \
  --model RelationStage1 \
  --data ETTh1 \
  --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.0 \
  --patch_len 16 \
  --stride 16 \
  --learning_rate 1e-3 \
  --train_epochs 1 \
  --patience 1 \
  --top_k 10 \
  --candidate_mask raft \
  --relation_input_space delta_last \
  --relation_teacher_space delta_last \
  --source_mode all \
  --target_mode single \
  --target_channel 6 \
  --relation_encoder_type mlp \
  --relation_self_fill linear \
  --tau_student "${TAU_STUDENT:-0.1}" \
  --tau_teacher 0.1 \
  --teacher_mse_space normalized \
  --stage1_teacher_mode mse \
  --stage1_loss_mode topk_coverage \
  --stage1_coverage_top_k "${COVERAGE_TOP_K:-10}" \
  --stage1_overfit_queries "${OVERFIT_QUERIES:-32}" \
  --stage1_overfit_candidates "${OVERFIT_CANDIDATES:-512}" \
  --stage1_overfit_steps "${OVERFIT_STEPS:-1000}" \
  --stage1_overfit_oracle_per_query "${ORACLE_PER_QUERY:-20}" \
  --stage1_overfit_key_refresh "${KEY_REFRESH:-step}" \
  --stage1_overfit_self_only "${SELF_ONLY:-1}" \
  --stage1_probe_vis 0 \
  --des stage1_tiny_overfit_topk_coverage_mlp_linear_ETTh1_seq96_pred96
