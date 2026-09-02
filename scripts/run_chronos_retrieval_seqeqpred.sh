#!/bin/bash
set -euo pipefail

# Chronos retrieval backbone: Top-K selected in a pretrained-encoder embedding
# space, in the spirit of RAF (arXiv:2411.08249).
#
#   frozen      RAF's setup: Chronos is a black box, retrieval is not trained
#   finetune    the encoder receives gradients from the Stage-2 loss
#   random      Chronos architecture with re-initialised weights (control)
#
# Only the embedding function and therefore the Top-K selection differ; the
# candidate graph, top_k, tau_topk, gate fusion, base head and loss are the
# same as every other condition in the retrieval ablation suite.
#
# RAF ranks candidates by L2 distance over embeddings while this pipeline uses
# cosine. On L2-normalised vectors ||a-b||^2 = 2(1-cos), so the ranking is the
# same and the two are interchangeable here.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1|Weather} [frozen|projection|frozen_projection|finetune|random]" >&2
  exit 2
fi

DATASET="$1"
MODE="${2:-frozen}"

case "${DATASET}" in
  ETTh1)
    DATA_NAME="ETTh1"; DATA_PATH="ETTh1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  ETTm1)
    DATA_NAME="ETTm1"; DATA_PATH="ETTm1.csv"; ENC_IN=7
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/ETT-small/"
    ;;
  Weather)
    DATA_NAME="custom"; DATA_PATH="weather.csv"; ENC_IN=21
    ROOT_PATH="../Dataset/Time-Series-Library_dataset/weather/"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/chronos_${MODE}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
CHRONOS_EMBEDDING_DIM="${CHRONOS_EMBEDDING_DIM:-768}"
CHRONOS_CONTEXT_LENGTH="${CHRONOS_CONTEXT_LENGTH:-512}"
# Empty keeps the Stage-2 learning rate for the encoder as well; set it to give
# the pretrained weights a smaller step.
CHRONOS_LR="${CHRONOS_LR:-}"
# Chronos encodes every channel of every memory window, so keep the batch of
# memory rows per encode call small enough to fit the T5 activations.
MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE:-256}"
# Chronos embeddings sit in a narrow cosine band, so the Top-K softmax
# temperature has to be swept: 0.10 leaves the weights nearly uniform.
TAU_TOPK="${TAU_TOPK:-0.10}"
# cross_only mirrors shared_cross_projection (2D -> D), so the dim must be
# the Chronos embedding dim; uniform allows any width.
CHRONOS_PROJECTION_MODE="${CHRONOS_PROJECTION_MODE:-cross_only}"
if [ "${CHRONOS_PROJECTION_MODE}" = "cross_only" ]; then
  CHRONOS_PROJECTION_DIM="${CHRONOS_PROJECTION_DIM:-${CHRONOS_EMBEDDING_DIM}}"
else
  CHRONOS_PROJECTION_DIM="${CHRONOS_PROJECTION_DIM:-128}"
fi

case "${MODE}" in
  frozen)
    MODE_ARGS=(--chronos_dtype "${CHRONOS_DTYPE:-bfloat16}" --chronos_finetune 0 --chronos_random_init 0
               --refresh_memory_every_epoch 0)
    EXPERIMENT="chronos_frozen"
    ;;
  random)
    MODE_ARGS=(--chronos_dtype "${CHRONOS_DTYPE:-bfloat16}" --chronos_finetune 0 --chronos_random_init 1
               --refresh_memory_every_epoch 0)
    EXPERIMENT="chronos_random_init"
    ;;
  projection)
    # Frozen T5 with a learned Linear(2*768 -> d_model) on the concatenated
    # [target || source] embeddings. The pooled T5 features are cached once, so
    # per-epoch re-indexing only re-applies the cheap linear map.
    MODE_ARGS=(--chronos_dtype "${CHRONOS_DTYPE:-bfloat16}" --chronos_finetune 0 --chronos_random_init 0
               --chronos_projection_dim "${CHRONOS_PROJECTION_DIM}"
               --chronos_projection_mode "${CHRONOS_PROJECTION_MODE}"
               --chronos_projection_trainable 1
               --refresh_memory_every_epoch 1)
    EXPERIMENT="chronos_projection_${CHRONOS_PROJECTION_MODE}${CHRONOS_PROJECTION_DIM}"
    ;;
  frozen_projection)
    # Control for the above: same geometry, projection randomly initialised and frozen.
    MODE_ARGS=(--chronos_dtype "${CHRONOS_DTYPE:-bfloat16}" --chronos_finetune 0 --chronos_random_init 0
               --chronos_projection_dim "${CHRONOS_PROJECTION_DIM}"
               --chronos_projection_mode "${CHRONOS_PROJECTION_MODE}"
               --chronos_projection_trainable 0
               --refresh_memory_every_epoch 0)
    EXPERIMENT="chronos_frozen_projection_${CHRONOS_PROJECTION_MODE}${CHRONOS_PROJECTION_DIM}"
    ;;
  finetune)
    # float32 weights and per-epoch re-indexing are both required; the encoder
    # is enforced to float32 inside ChronosRelationEncoder anyway.
    MODE_ARGS=(--chronos_dtype float32 --chronos_finetune 1 --chronos_random_init 0
               --refresh_memory_every_epoch 1)
    EXPERIMENT="chronos_finetune"
    if [ -n "${CHRONOS_LR}" ]; then
      MODE_ARGS+=(--chronos_lr "${CHRONOS_LR}")
      EXPERIMENT="chronos_finetune_lr${CHRONOS_LR}"
    fi
    ;;
  *)
    echo "Unsupported mode: ${MODE} (expected frozen|projection|frozen_projection|finetune|random)" >&2
    exit 2
    ;;
esac

TAU_TAG="tauk${TAU_TOPK/./p}"
EXPERIMENT="${EXPERIMENT}_${TAU_TAG}"
LOG_DIR="${LOG_DIR}_${TAU_TAG}"
mkdir -p "${LOG_DIR}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"
  LOG_PATH="${LOG_DIR}/seq${SEQ_LEN}_pred${PRED_LEN}.log"

  echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][chronos ${MODE}] Stage-2"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model RelationStage2 \
    --model_id "CARTS_stage2_${EXPERIMENT}_top${RELATION_TOP_N}_${DATASET}_${PRED_LEN}" \
    --data "${DATA_NAME}" \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len "${PRED_LEN}" \
    --enc_in "${ENC_IN}" \
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
    --stage1_encoder_init none \
    --stage2_retrieval_backbone chronos \
    --chronos_model_id "${CHRONOS_MODEL_ID}" \
    --chronos_embedding_dim "${CHRONOS_EMBEDDING_DIM}" \
    --chronos_context_length "${CHRONOS_CONTEXT_LENGTH}" \
    "${MODE_ARGS[@]}" \
    --freeze_stage1_encoder 1 \
    --memory_cache_mode precompute \
    --memory_chunk_size "${MEMORY_CHUNK_SIZE}" \
    --top_k 10 \
    --tau_topk "${TAU_TOPK}" \
    --stage2_relation_fusion gate \
    --relation_mixer_input retrieved \
    --fusion_mode raft_concat \
    --oracle_candidate_eval 1 \
    --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
    2>&1 | tee "${LOG_PATH}"
done

echo "[${DATASET}][chronos ${MODE}] logs: ${LOG_DIR}"
