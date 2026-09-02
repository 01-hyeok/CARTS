#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1}" >&2
  exit 2
fi

DATASET="$1"
case "${DATASET}" in
  ETTh1)
    DATA_PATH="ETTh1.csv"
    ;;
  ETTm1)
    DATA_PATH="ETTm1.csv"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
# Single seed 0 matches every other condition in the retrieval ablation suite.
SEEDS=(${SEEDS:-0})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
RESUME_COMPLETED="${RESUME_COMPLETED:-0}"
REPORT_SUFFIX="${REPORT_SUFFIX:-}"

LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/random_retrieval_backbone"
RAW_CSV="${LOG_DIR}/seed_metrics${REPORT_SUFFIX}.csv"
SUMMARY_CSV="${LOG_DIR}/seed_summary${REPORT_SUFFIX}.csv"
mkdir -p "${LOG_DIR}"
printf 'pred_len,seed,final_mse,final_mae,log_path\n' > "${RAW_CSV}"
printf 'dataset,pred_len,num_seeds,final_mse_mean,final_mse_std,final_mae_mean,final_mae_std\n' > "${SUMMARY_CSV}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"

  for SEED in "${SEEDS[@]}"; do
    EXPERIMENT="random_retrieval_backbone_mlp_linear_top${RELATION_TOP_N}_gate_raft_tauk0p10_seed${SEED}"
    LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}_seed${SEED}.log"

    EXISTING_FINAL_MSE="$(awk '/^final_mse:/ {value=$2} END {print value}' "${LOG_PATH}" 2>/dev/null || true)"
    EXISTING_FINAL_MAE="$(awk '/^final_mae:/ {value=$2} END {print value}' "${LOG_PATH}" 2>/dev/null || true)"
    if [ "${RESUME_COMPLETED}" = "1" ] && [ -n "${EXISTING_FINAL_MSE}" ] && [ -n "${EXISTING_FINAL_MAE}" ]; then
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][seed${SEED}] Reusing completed random-backbone result"
    else
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][seed${SEED}] Random retrieval backbone"
      python -u run.py \
        --task_name stage2_relation \
        --learning_rate 1e-2 \
        --is_training 1 \
        --seed "${SEED}" \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
        --model RelationStage2 \
        --data "${DATASET}" \
        --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
        --data_path "${DATA_PATH}" \
        --features M \
        --seq_len "${SEQ_LEN}" \
        --label_len 0 \
        --pred_len "${PRED_LEN}" \
        --enc_in 7 \
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
        --relation_top_n "${RELATION_TOP_N}" \
        --target_mode all \
        --relation_encoder_type mlp \
        --relation_self_fill linear \
        --base_head_mode shared_target_linear \
        --train_epochs 10 \
        --patience 5 \
        --stage1_encoder_init random \
        --stage2_retrieval_encoder online \
        --freeze_stage1_encoder 1 \
        --memory_cache_mode precompute \
        --refresh_memory_every_epoch 0 \
        --memory_chunk_size 1024 \
        --top_k 10 \
        --tau_topk 0.10 \
        --stage2_relation_fusion gate \
        --relation_mixer_input retrieved \
        --fusion_mode raft_concat \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee "${LOG_PATH}"
    fi

    FINAL_MSE="$(awk '/^final_mse:/ {value=$2} END {print value}' "${LOG_PATH}")"
    FINAL_MAE="$(awk '/^final_mae:/ {value=$2} END {print value}' "${LOG_PATH}")"
    if [ -z "${FINAL_MSE}" ] || [ -z "${FINAL_MAE}" ]; then
      echo "Could not parse final metrics from ${LOG_PATH}" >&2
      exit 1
    fi
    printf '%s,%s,%s,%s,%s\n' \
      "${PRED_LEN}" "${SEED}" "${FINAL_MSE}" "${FINAL_MAE}" "${LOG_PATH}" \
      >> "${RAW_CSV}"
  done

  awk -F, -v dataset="${DATASET}" -v pred_len="${PRED_LEN}" '
    NR > 1 && $1 == pred_len {
      n += 1
      mse_sum += $3
      mse_square_sum += $3 * $3
      mae_sum += $4
      mae_square_sum += $4 * $4
    }
    END {
      if (n == 0) {
        exit 1
      }
      mse_mean = mse_sum / n
      mae_mean = mae_sum / n
      mse_variance = mse_square_sum / n - mse_mean * mse_mean
      mae_variance = mae_square_sum / n - mae_mean * mae_mean
      if (mse_variance < 0) mse_variance = 0
      if (mae_variance < 0) mae_variance = 0
      printf "%s,%s,%d,%.10f,%.10f,%.10f,%.10f\n", \
        dataset, pred_len, n, mse_mean, sqrt(mse_variance), mae_mean, sqrt(mae_variance)
    }
  ' "${RAW_CSV}" >> "${SUMMARY_CSV}"
done

echo "Random-backbone seed metrics: ${RAW_CSV}"
echo "Random-backbone mean/std summary: ${SUMMARY_CSV}"
