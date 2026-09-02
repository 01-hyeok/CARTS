#!/bin/bash
set -euo pipefail

# RAF (arXiv:2411.08249) on the CARTS harness, pred_len 96 only.
#
#   frozen    Naive RAF: zero-shot Chronos + retrieval augmentation at inference
#   finetune  Advanced RAF: Chronos fine-tuned on retrieval-augmented data, then
#             evaluated with the same augmentation
#
# A zero-shot Chronos run with no retrieval is included as RAF's own baseline so
# the augmentation gain can be read off directly.
#
# pred_len is fixed at 96: RAF feeds
# [retrieved context | retrieved future | query context] to Chronos, which is
# 3*L under seq_len == pred_len. Only 96 (288 tokens) fits the 512-token context.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAF_REPO="${RAF_REPO:-/data/pjh_workspace/RAF}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
LOG_DIR="${PROJECT_ROOT}/logs/raf_port"
WORK_DIR="${PROJECT_ROOT}/outputs/raf_finetune"
SUMMARY="${LOG_DIR}/summary.csv"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LEN=96
SEQ_LEN=96
TOP_N="${TOP_N:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
# README fine-tunes chronos-t5-base for 1000 steps at 1e-5.
FT_MAX_STEPS="${FT_MAX_STEPS:-1000}"
FT_LEARNING_RATE="${FT_LEARNING_RATE:-1e-5}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
MODES=(${MODES:-frozen finetune})

dataset_args() {
  case "$1" in
    ETTh1) echo "--data ETTh1 --data_path ETTh1.csv --freq h --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/" ;;
    ETTm1) echo "--data ETTm1 --data_path ETTm1.csv --freq t --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/" ;;
    *) echo "Unsupported dataset: $1" >&2; exit 2 ;;
  esac
}

check_finetune_deps() {
  if ! python -c 'import gluonts, typer, typer_config' 2>/dev/null; then
    cat >&2 <<'EOF'
Advanced RAF needs the RAF trainer's dependencies, which are not in this venv:

    pip install gluonts==0.15.1 typer typer-config

They are only required for the fine-tuning step; the frozen (Naive RAF) runs and
all evaluation work without them.
EOF
    return 1
  fi
}

for DATASET in "${DATASETS[@]}"; do
  DS_ARGS="$(dataset_args "${DATASET}")"

  for MODE in "${MODES[@]}"; do
    case "${MODE}" in

      frozen)
        echo "[${DATASET}] Naive RAF (frozen Chronos, retrieval augmented)"
        python -u eval/run_raf_baseline.py ${DS_ARGS} \
          --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" \
          --top_n "${TOP_N}" --num_samples "${NUM_SAMPLES}" \
          --chronos_model_id "${CHRONOS_MODEL_ID}" \
          --augment --output_csv "${SUMMARY}" \
          2>&1 | tee "${LOG_DIR}/${DATASET,,}_raf_frozen.log"

        echo "[${DATASET}] RAF baseline (zero-shot Chronos, no retrieval)"
        python -u eval/run_raf_baseline.py ${DS_ARGS} \
          --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" \
          --num_samples "${NUM_SAMPLES}" \
          --chronos_model_id "${CHRONOS_MODEL_ID}" \
          --no-augment --output_csv "${SUMMARY}" \
          2>&1 | tee "${LOG_DIR}/${DATASET,,}_chronos_zeroshot.log"
        ;;

      finetune)
        check_finetune_deps || exit 1

        ARROW="${WORK_DIR}/${DATASET,,}_raf_train.arrow"
        FT_CONFIG="${WORK_DIR}/${DATASET,,}_raf_finetune.yaml"
        FT_OUT="${WORK_DIR}/${DATASET,,}_raf_model"
        # The arrow series are [augmented context | future]; the trainer needs the
        # context width, which is 2L+H for the augmented variant.
        CONTEXT_LENGTH=$((2 * SEQ_LEN + PRED_LEN))

        echo "[${DATASET}] Building Advanced-RAF fine-tuning data"
        python -u eval/make_raf_finetune_arrow.py ${DS_ARGS} \
          --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" \
          --top_n "${TOP_N}" \
          --chronos_model_id "${CHRONOS_MODEL_ID}" \
          --augment --out "${ARROW}" \
          2>&1 | tee "${LOG_DIR}/${DATASET,,}_raf_arrow.log"

        cat > "${FT_CONFIG}" <<EOF
training_data_paths:
- ${ARROW}
probability:
- 1.0
context_length: ${CONTEXT_LENGTH}
prediction_length: ${PRED_LEN}
min_past: 64
max_steps: ${FT_MAX_STEPS}
save_steps: ${FT_MAX_STEPS}
log_steps: 100
per_device_train_batch_size: ${FT_BATCH_SIZE}
learning_rate: ${FT_LEARNING_RATE}
optim: adamw_torch_fused
num_samples: ${NUM_SAMPLES}
shuffle_buffer_length: 100_000
gradient_accumulation_steps: 1
model_id: ${CHRONOS_MODEL_ID}
model_type: seq2seq
random_init: false
tie_embeddings: true
output_dir: ${FT_OUT}
tf32: true
torch_compile: false
tokenizer_class: "MeanScaleUniformBins"
tokenizer_kwargs:
  low_limit: -15.0
  high_limit: 15.0
n_tokens: 4096
lr_scheduler_type: linear
warmup_ratio: 0.0
dataloader_num_workers: 1
max_missing_prop: 0.9
use_eos_token: true
EOF

        echo "[${DATASET}] Fine-tuning Chronos with the RAF trainer"
        (
          cd "${RAF_REPO}"
          python -u chronos_training/train.py --config "${FT_CONFIG}" \
            --model-id "${CHRONOS_MODEL_ID}" \
            --no-random-init \
            --max-steps "${FT_MAX_STEPS}" \
            --learning-rate "${FT_LEARNING_RATE}"
        ) 2>&1 | tee "${LOG_DIR}/${DATASET,,}_raf_finetune_train.log"

        CKPT="$(find "${FT_OUT}" -maxdepth 3 -name 'checkpoint-*' -type d | sort -V | tail -1)"
        if [ -z "${CKPT}" ]; then
          echo "No fine-tuned checkpoint found under ${FT_OUT}" >&2
          exit 1
        fi
        echo "[${DATASET}] Advanced RAF evaluation with ${CKPT}"
        python -u eval/run_raf_baseline.py ${DS_ARGS} \
          --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" \
          --top_n "${TOP_N}" --num_samples "${NUM_SAMPLES}" \
          --chronos_model_id "${CKPT}" \
          --augment --output_csv "${SUMMARY}" \
          2>&1 | tee "${LOG_DIR}/${DATASET,,}_raf_finetune_eval.log"
        ;;

      *)
        echo "Unsupported mode: ${MODE} (expected frozen|finetune)" >&2
        exit 2
        ;;
    esac
  done
done

echo "RAF port results: ${SUMMARY}"
