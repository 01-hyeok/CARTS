#!/bin/bash
set -uo pipefail

# Drop Top-K and let one end-to-end loss train retrieval.
#
# Top-K selects indices, which is not differentiable. The forecasting loss can
# therefore only reshape alpha over candidates that were already picked, and
# alpha sits at an effective 9.8 out of 10 across every arm measured so far - a
# plain average. That is why retrieval quality moves a lot (recall@10 0.8% ->
# 5.1% across arms) while final MSE barely moves (0.4168 -> 0.4006).
#
# soft_all weights the whole bank with softmax(scores/tau_topk), so every
# candidate score is on the gradient path and the MSE alone can train retrieval.
#
# tau_topk is the parameter that matters. Measured on ETTh1 (8449 candidates,
# delta_last cosine), the effective candidate count of a full-bank softmax is:
#     tau=0.10  -> 1403   averaging this many futures is close to a global mean
#     tau=0.05  ->  453
#     tau=0.01  ->   17   comparable to Top-K=10, and alpha_top1 reaches 0.47
# so 0.01 is the setting to look at first and 0.10 is expected to be bad.
#
# lambda=0 by default: the point is to test whether the forecasting loss alone
# is enough once the path is differentiable. LAMBDA=1.0 runs it with the KL still
# on, which separates "soft attention helps" from "the KL was carrying it".
#
# Usage
#   bash scripts/run_soft_retrieval.sh                        # ETTh1+ETTm1, tau sweep
#   TAUS=0.01 bash scripts/run_soft_retrieval.sh
#   LAMBDA=1.0 bash scripts/run_soft_retrieval.sh             # keep the KL on
#   DATASETS=ETTh1 PRED_LENS=96 bash scripts/run_soft_retrieval.sh
#   nohup bash scripts/run_soft_retrieval.sh > logs/soft_retrieval_driver.log 2>&1 &

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
source "${VENV_ACTIVATE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336})
TAUS=(${TAUS:-0.01 0.05 0.10})
LAMBDA="${LAMBDA:-0.0}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
TOP_K="${TOP_K:-10}"          # only feeds recall@k / oracle diagnostics in soft mode
FORCE="${FORCE:-0}"

echo "=== soft retrieval (no Top-K selection) ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  pred_lens  : ${PRED_LENS[*]}"
echo "  taus       : ${TAUS[*]}"
echo "  lambda     : ${LAMBDA}   (0 = forecasting loss only)"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo

for DS in "${DATASETS[@]}"; do
  case "${DS}" in
    ETTh1) DATA_PATH="ETTh1.csv" ;;
    ETTm1) DATA_PATH="ETTm1.csv" ;;
    *) echo "Unsupported dataset: ${DS}" >&2; exit 2 ;;
  esac
  LOG_DIR="${PROJECT_ROOT}/logs/${DS}/soft_retrieval"
  mkdir -p "${LOG_DIR}"

  for TAU in "${TAUS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      SEQ_LEN="${PRED_LEN}"
      TAG="tau${TAU//./p}_lam${LAMBDA//./p}"
      EXPERIMENT="soft_${TAG}"
      LOG_PATH="${LOG_DIR}/${TAG}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
      if [ "${FORCE}" != "1" ] && grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
        echo "[${DS}][seq${SEQ_LEN}_pred${PRED_LEN}][${TAG}] already finished, skipping"
        continue
      fi
      [ "${FORCE}" = "1" ] && : > "${LOG_PATH}"

      echo "[${DS}][seq${SEQ_LEN}_pred${PRED_LEN}][${TAG}] soft_all, tau_topk=${TAU}"
      if ! python -u run.py \
        --task_name stage2_relation \
        --is_training 1 \
        --model RelationStage2 \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DS}_${PRED_LEN}" \
        --data "${DS}" \
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
        --learning_rate 1e-2 \
        --train_epochs "${TRAIN_EPOCHS}" \
        --patience "${PATIENCE}" \
        --tau_student "${STUDENT_TEMP}" \
        --tau_teacher "${TEACHER_TEMP}" \
        --candidate_mask raft \
        --relation_input_space delta_last \
        --relation_teacher_space delta_last \
        --relation_value_space delta_last \
        --source_mode auto \
        --relation_top_n 1 \
        --target_mode all \
        --relation_encoder_type mlp \
        --relation_self_fill linear \
        --base_head_mode shared_target_linear \
        --stage2_retrieval_backbone stage1 \
        --stage1_encoder_init random \
        --freeze_stage1_encoder 0 \
        --retrieval_soft_all 1 \
        --retrieval_kl_weight "${LAMBDA}" \
        --retrieval_kl_teacher future_mse \
        --memory_cache_mode precompute \
        --refresh_memory_every_epoch 1 \
        --memory_chunk_size 1024 \
        --top_k "${TOP_K}" \
        --tau_topk "${TAU}" \
        --stage2_relation_fusion gate \
        --relation_mixer_input retrieved \
        --fusion_mode raft_concat \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DS}_seq${SEQ_LEN}_pred${PRED_LEN}_topk${TOP_K}" \
        2>&1 | tee -a "${LOG_PATH}"
      then
        echo "[FAILED] ${DS} seq${SEQ_LEN}_pred${PRED_LEN} ${TAG}, see ${LOG_PATH}" \
          | tee -a "${LOG_DIR}/_failures.txt" >&2
        continue
      fi
    done
  done
done

echo
echo "  finished   : $(date '+%Y-%m-%d %H:%M:%S')"
bash "${SCRIPT_DIR}/summarize_soft_retrieval.sh" || true
