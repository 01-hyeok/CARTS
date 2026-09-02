#!/bin/bash
# STEP 3 -- does raw past retrieval beat the learned encoder in actual forecasting?
#
# Stage-2 is untouched: only the retrieval source changes. All four conditions
# share every Stage-2 hyperparameter so the comparison isolates the retriever.
#
#   no_retrieval   --disable_retrieval 1                (base head only)
#   learned        --stage2_retrieval_backbone stage1   (trained Stage-1 encoder)
#   raw_l2         --stage2_retrieval_backbone identity + --retrieval_similarity l2
#   oracle         --stage2_oracle_train_mode candidate (upper bound)
#
# The learned arm is already trained by run_candidate_reencode_kl_full.sh; this
# script reruns it only when its checkpoint is missing.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
CONDITIONS=(${CONDITIONS:-no_retrieval raw_l2 oracle})
EPOCHS="${STAGE2_EPOCHS:-10}"
PATIENCE="${STAGE2_PATIENCE:-5}"
SEED="${SEED:-0}"
RELATION_TOP_N="${RELATION_TOP_N:-1}"
LOG_ROOT="${LOG_ROOT:-./logs/raw_retrieval_stage2}"
mkdir -p "${LOG_ROOT}"

condition_args() {
  case "$1" in
    no_retrieval) echo "--disable_retrieval 1" ;;
    learned)      echo "--stage2_retrieval_backbone stage1" ;;
    raw_l2)       echo "--stage2_retrieval_backbone identity --retrieval_similarity l2 --stage1_encoder_init random" ;;
    raw_cos)      echo "--stage2_retrieval_backbone identity --retrieval_similarity cosine --stage1_encoder_init random" ;;
    oracle)       echo "--stage2_oracle_train_mode candidate --freeze_stage1_encoder 1 --stage2_retrieval_backbone identity --stage1_encoder_init random" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

FAILED=0
for DATASET in "${DATASETS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED_LEN}"
    mkdir -p "${LOG_DIR}"
    for COND in "${CONDITIONS[@]}"; do
      MARKER="${LOG_DIR}/${COND}.done"
      if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then
        echo "[skip] ${DATASET}/pred${PRED_LEN}/${COND}"
        continue
      fi
      EXTRA="$(condition_args "${COND}")" || exit 1
      ID="carts_s2_raw_${COND}_${DATASET}_${PRED_LEN}"
      DES="s2_raw_${COND}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      echo "=============================================================="
      echo "[${DATASET}/pred${PRED_LEN}/${COND}] ${EXTRA}"
      {
        echo "### RUN ${DATASET}/pred${PRED_LEN}/${COND} $(date -Is)"
        python -u run.py \
          --task_name stage2_relation \
          --is_training 1 \
          --model_id "${ID}" \
          --model RelationStage2 \
          --base_head_mode shared_target_linear \
          --data "${DATASET}" \
          --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
          --data_path "${DATASET}.csv" \
          --features M \
          --seq_len "${SEQ_LEN}" --label_len 0 --pred_len "${PRED_LEN}" \
          --enc_in 7 --batch_size 32 --num_workers 0 \
          --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
          --patch_len 16 --stride 16 --seed "${SEED}" \
          --candidate_mask raft \
          --relation_input_space delta_last \
          --relation_teacher_space delta_last \
          --relation_value_space delta_last \
          --source_mode auto --relation_top_n "${RELATION_TOP_N}" --target_mode all \
          --relation_encoder_type mlp --relation_self_fill linear \
          --learning_rate 1e-2 \
          --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
          --freeze_stage1_encoder 1 \
          --memory_cache_mode precompute \
          --refresh_memory_every_epoch 1 --memory_chunk_size 1024 \
          --top_k 10 --tau_topk 0.10 \
          --relation_mixer_input retrieved \
          --fusion_mode residual --gate_mode scalar \
          ${EXTRA} \
          --des "${DES}" || exit 21
        echo "### RUN COMPLETE $(date -Is)"
      } 2>&1 | tee "${LOG_DIR}/${COND}.log"
      if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG_DIR}/${COND}.log"; then
        touch "${MARKER}"; echo "[ok] ${DATASET}/pred${PRED_LEN}/${COND}"
      else
        FAILED=$((FAILED+1)); echo "[FAILED] ${DATASET}/pred${PRED_LEN}/${COND}"
      fi
    done
  done
done
echo "failed: ${FAILED}   logs: ${LOG_ROOT}"
