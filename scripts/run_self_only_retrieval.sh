#!/bin/bash
set -euo pipefail

# Self-only retrieval: no cross-channel relation.
#
# relation_top_n=1 keeps only the self source, because the relation graph ranks
# self first and top_n counts the total source list including self. This is the
# TS-RAG retrieval setting (per-variable, retrieve from its own history) placed
# inside the CARTS Stage-2 pipeline, so the encoder is the only thing that
# differs between arms.
#
# Arms are all encoder-free or frozen, so none of them needs a Stage-1
# checkpoint and they can be compared directly.

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
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/self_only"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
ARMS=(${ARMS:-identity pearson chronos})
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
RELATION_TOP_N=1

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  for ARM in "${ARMS[@]}"; do
    case "${ARM}" in
      identity)
        BACKBONE_ARGS=(--stage2_retrieval_backbone identity --stage1_encoder_init none
                       --refresh_memory_every_epoch 0)
        ;;
      pearson)
        BACKBONE_ARGS=(--stage2_retrieval_backbone pearson --stage1_encoder_init none
                       --refresh_memory_every_epoch 0)
        ;;
      chronos)
        BACKBONE_ARGS=(--stage2_retrieval_backbone chronos --stage1_encoder_init none
                       --chronos_model_id "${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
                       --chronos_dtype bfloat16 --chronos_finetune 0 --chronos_random_init 0
                       --refresh_memory_every_epoch 0)
        ;;
      *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
    esac

    EXPERIMENT="self_only_${ARM}"
    LOG_PATH="${LOG_DIR}/${ARM}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
    if grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] already finished, skipping"
      continue
    fi

    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] self-only Stage-2"
    python -u run.py \
      --task_name stage2_relation \
      --is_training 1 \
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
      --learning_rate 1e-2 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --candidate_mask raft \
      --relation_input_space delta_last \
      --relation_value_space delta_last \
      --source_mode auto \
      --relation_top_n "${RELATION_TOP_N}" \
      --target_mode all \
      --base_head_mode shared_target_linear \
      --freeze_stage1_encoder 1 \
      --memory_cache_mode precompute \
      --memory_chunk_size 1024 \
      --top_k 10 \
      --tau_topk 0.10 \
      --stage2_relation_fusion gate \
      --relation_mixer_input retrieved \
      --fusion_mode raft_concat \
      --oracle_candidate_eval 1 \
      "${BACKBONE_ARGS[@]}" \
      --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
      2>&1 | tee "${LOG_PATH}"
  done
done

echo "Self-only retrieval logs: ${LOG_DIR}"
