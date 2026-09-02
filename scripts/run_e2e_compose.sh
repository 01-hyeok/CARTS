#!/bin/bash
set -euo pipefail

# End-to-end compose study: how the source channel enters the retrieval query.
#
#   self    relation_top_n=1, self_fill=linear
#           No cross relation at all. The encoder sees the target window only.
#   concat  relation_top_n=3, self_fill=repeat
#           Cross branches feed the encoder two raw rows [target; source] and the
#           encoder's first Linear(2L -> d_ff) mixes them. Self branch is [t; t].
#   linear  relation_top_n=3, self_fill=linear   (already run, kept for reference)
#           Cross branches go through the shared 2L -> L projection first.
#
# Every arm is true end-to-end: no Stage-1, random encoder init, encoder trained
# with the Stage-2 loss. lambda > 0 is mandatory - Top-K is not differentiable,
# so with lambda=0 the retrieval encoder cannot reorder candidates and barely
# trains at all. Both retrieval KL teachers are swept.

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

DATASET="${1:-ETTh1}"
case "${DATASET}" in
  ETTh1) DATA_PATH="ETTh1.csv" ;;
  ETTm1) DATA_PATH="ETTm1.csv" ;;
  *) echo "Usage: $0 {ETTh1|ETTm1}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/e2e_compose"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
COMPOSES=(${COMPOSES:-self concat})
TEACHERS=(${TEACHERS:-future_mse ema})
LAMBDA="${LAMBDA:-1.0}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
TAU_TOPK="${TAU_TOPK:-0.10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"

if [ "$(echo "${LAMBDA} == 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  echo "[warn] LAMBDA=0 leaves the retrieval encoder without a usable gradient." >&2
fi

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  for COMPOSE in "${COMPOSES[@]}"; do
    case "${COMPOSE}" in
      self)   TOP_N=1; SELF_FILL=linear ;;
      concat) TOP_N=3; SELF_FILL=repeat ;;
      linear) TOP_N=3; SELF_FILL=linear ;;
      *) echo "Unknown compose: ${COMPOSE}" >&2; exit 2 ;;
    esac

    for TEACHER in "${TEACHERS[@]}"; do
      EXPERIMENT="e2e_${COMPOSE}_${TEACHER}_lambda${LAMBDA/./p}"
      LOG_PATH="${LOG_DIR}/${COMPOSE}_${TEACHER}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
      if grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
        echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${COMPOSE}/${TEACHER}] already finished, skipping"
        continue
      fi

      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${COMPOSE}/${TEACHER}] top_n=${TOP_N} self_fill=${SELF_FILL} lambda=${LAMBDA}"
      python -u run.py \
        --task_name stage2_relation \
        --is_training 1 \
        --model RelationStage2 \
        --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
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
        --relation_top_n "${TOP_N}" \
        --target_mode all \
        --relation_encoder_type mlp \
        --relation_self_fill "${SELF_FILL}" \
        --base_head_mode shared_target_linear \
        --stage2_retrieval_backbone stage1 \
        --stage1_encoder_init random \
        --freeze_stage1_encoder 0 \
        --retrieval_kl_weight "${LAMBDA}" \
        --retrieval_kl_teacher "${TEACHER}" \
        --memory_cache_mode precompute \
        --refresh_memory_every_epoch 1 \
        --memory_chunk_size 1024 \
        --top_k 10 \
        --tau_topk "${TAU_TOPK}" \
        --stage2_relation_fusion gate \
        --relation_mixer_input retrieved \
        --fusion_mode raft_concat \
        --oracle_candidate_eval 1 \
        --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee "${LOG_PATH}"
    done
  done
done

echo "e2e compose logs: ${LOG_DIR}"
