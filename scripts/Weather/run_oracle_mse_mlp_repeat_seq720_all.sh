#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SEQ_LEN="${SEQ_LEN:-720}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})

for PRED_LEN in "${PRED_LENS[@]}"; do
  STAGE1_CKPT_PATH="./checkpoints/stage1/custom/seq${SEQ_LEN}_pred${PRED_LEN}/stage1_CARTS_stage1_mse_mlp_linear_Weather_${PRED_LEN}_RelationStage1_custom_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_stage1_mse_mlp_linear_Weather_seq${SEQ_LEN}_pred${PRED_LEN}_0/checkpoint.pth"
  STAGE2_SETTING="stage2_CARTS_stage2_mse_mlp_linear_Weather_${PRED_LEN}_RelationStage2_custom_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_stage2_mse_mlp_linear_Weather_seq${SEQ_LEN}_pred${PRED_LEN}_topk10_0"
  STAGE2_CKPT_PATH="./checkpoints/stage2/custom/seq${SEQ_LEN}_pred${PRED_LEN}/${STAGE2_SETTING}/checkpoint.pth"

  for CKPT_PATH in "${STAGE1_CKPT_PATH}" "${STAGE2_CKPT_PATH}"; do
    if [ ! -f "${CKPT_PATH}" ]; then
      echo "[Weather][seq${SEQ_LEN}_pred${PRED_LEN}][oracle] Missing checkpoint: ${CKPT_PATH}"
      exit 1
    fi
  done

  echo "[Weather][seq${SEQ_LEN}_pred${PRED_LEN}][mse_mlp_linear] Candidate Oracle Test"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 0 \
    --model_id "CARTS_stage2_mse_mlp_linear_Weather_${PRED_LEN}" \
    --model RelationStage2 \
    --data custom \
    --root_path ../Dataset/Time-Series-Library_dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --target OT \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len "${PRED_LEN}" \
    --enc_in 21 \
    --batch_size 32 \
    --num_workers 0 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --patch_len 16 \
    --stride 16 \
    --candidate_mask raft \
    --relation_input_space delta_last \
    --relation_teacher_space delta_last \
    --relation_value_space delta_last \
    --source_mode auto \
    --target_mode all \
    --relation_encoder_type mlp \
    --relation_self_fill linear \
    --focus_channel OT \
    --base_head_mode shared_target_linear \
    --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --memory_chunk_size 1024 \
    --top_k 10 \
    --tau_topk 0.10 \
    --relation_mixer_input retrieved \
    --fusion_mode residual \
    --gate_mode scalar \
    --oracle_candidate_eval 1 \
    --use_tensorboard 0 \
    --des "stage2_mse_mlp_linear_Weather_seq${SEQ_LEN}_pred${PRED_LEN}_topk10"
done
