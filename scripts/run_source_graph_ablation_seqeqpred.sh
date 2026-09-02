#!/bin/bash
set -euo pipefail

# Source-selection controls for the concat relation design, run on top of the
# current learned EMA encoder (condition 5) so the only thing that changes is
# which source channel each target is paired with.
#
#   A (self)   sources = [target]                 -> does a source channel help at all?
#   B (random) sources = [target, random peers]   -> is the correlated source better
#                                                    than an arbitrary one?
#
# Everything else matches the retrieval ablation suite: seq==pred, top_k 10,
# tau_topk 0.10, gate fusion, raft_concat, 10 epochs, seed 0.

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {ETTh1|ETTm1|Weather}" >&2
  exit 2
fi

DATASET="$1"
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
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/source_graph_ablation"
SETTING_COMPONENT_MAX_BYTES=200

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

shorten_path_component() {
  local value="$1"
  local byte_length
  byte_length="$(printf '%s' "${value}" | wc -c)"
  if (( byte_length <= SETTING_COMPONENT_MAX_BYTES )); then
    printf '%s' "${value}"
    return
  fi
  local digest prefix_max_bytes prefix
  digest="$(printf '%s' "${value}" | sha256sum)"
  digest="${digest%% *}"
  digest="${digest:0:12}"
  prefix_max_bytes=$((SETTING_COMPONENT_MAX_BYTES - ${#digest} - 1))
  prefix="${value:0:${prefix_max_bytes}}"
  while [[ "${prefix}" == *'.' || "${prefix}" == *' ' || "${prefix}" == *'_' || "${prefix}" == *'-' ]]; do
    prefix="${prefix%?}"
  done
  printf '%s_%s' "${prefix}" "${digest}"
}

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRED_LENS=(${PRED_LENS:-96 192 336 720})
RELATION_TOP_N="${RELATION_TOP_N:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
# VARIANTS: self, random, or both. RANDOM_SEEDS controls how many independent
# random source draws are averaged over; the draw itself is the treatment here.
VARIANTS=(${VARIANTS:-self random})
RANDOM_SEEDS=(${RANDOM_SEEDS:-0 1 2})

STUDENT_TEMP_TAG="${STUDENT_TEMP/./p}"
TEACHER_TEMP_TAG="${TEACHER_TEMP/./p}"
TEMP_TAG="tau_s${STUDENT_TEMP_TAG}_t${TEACHER_TEMP_TAG}"

GRAPH_STEM="${DATA_PATH%.csv}"
BASE_GRAPH="${PROJECT_ROOT}/metrics/relation_graphs/${GRAPH_STEM}/pearson_self_top${RELATION_TOP_N}.json"
VARIANT_DIR="${PROJECT_ROOT}/metrics/relation_graphs/${GRAPH_STEM}/source_controls"

if [ ! -f "${BASE_GRAPH}" ]; then
  echo "Base relation graph not found: ${BASE_GRAPH}" >&2
  echo "Run any source_mode=auto experiment on ${DATASET} first to generate it." >&2
  exit 1
fi

echo "[${DATASET}] Building source-control graphs from ${BASE_GRAPH}"
python3 "${SCRIPT_DIR}/make_relation_graph_variants.py" \
  --base_graph "${BASE_GRAPH}" \
  --out_dir "${VARIANT_DIR}" \
  --random_seeds "${RANDOM_SEEDS[@]}"

run_one() {
  local variant_tag="$1" graph_path="$2" top_n="$3" pred_len="$4"
  local seq_len="${pred_len}"
  local experiment="source_${variant_tag}_mlp_linear_top${top_n}_${TEMP_TAG}"
  local stage1_model_id="CARTS_stage1_${experiment}_${DATASET}_${pred_len}"
  local stage1_des="stage1_${experiment}_${DATASET}_seq${seq_len}_pred${pred_len}"
  local stage1_setting_full stage1_setting stage1_ckpt log_path

  stage1_setting_full="stage1_${stage1_model_id}_RelationStage1_${DATA_NAME}_ftM_sl${seq_len}_ll0_pl${pred_len}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${stage1_des}_0"
  stage1_setting="$(shorten_path_component "${stage1_setting_full}")"
  stage1_ckpt="./checkpoints/stage1/${DATA_NAME}/seq${seq_len}_pred${pred_len}/${stage1_setting}/checkpoint.pth"
  log_path="${LOG_DIR}/${variant_tag}_seq${seq_len}_pred${pred_len}.log"

  local common_args=(
    --data "${DATA_NAME}"
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_PATH}"
    --features M
    --seq_len "${seq_len}"
    --label_len 0
    --pred_len "${pred_len}"
    --enc_in "${ENC_IN}"
    --batch_size 32
    --num_workers 0
    --d_model 128
    --n_heads 4
    --e_layers 2
    --d_ff 256
    --patch_len 16
    --stride 16
    --candidate_mask raft
    --relation_input_space delta_last
    --relation_teacher_space delta_last
    --relation_value_space delta_last
    --source_mode auto
    --relation_top_n "${top_n}"
    --relation_graph_path "${graph_path}"
    --target_mode all
    --relation_encoder_type mlp
    --relation_self_fill linear
  )

  echo "[${DATASET}][${variant_tag}][seq${seq_len}_pred${pred_len}] Stage-1"
  python -u run.py \
    --task_name stage1_relation \
    --is_training 1 \
    --model_id "${stage1_model_id}" \
    --model RelationStage1 \
    --learning_rate 1e-3 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --tau_student "${STUDENT_TEMP}" \
    --tau_teacher "${TEACHER_TEMP}" \
    --teacher_mse_space normalized \
    --stage1_teacher_mode ema_target \
    --stage1_loss_mode kl \
    --stage1_probe_vis 0 \
    --stage1_ema_momentum_base 0.99 \
    --stage1_ema_momentum_final 0.9995 \
    --des "${stage1_des}" \
    "${common_args[@]}" \
    2>&1 | tee "${log_path}"

  if [ ! -f "${stage1_ckpt}" ]; then
    echo "[${DATASET}][${variant_tag}][seq${seq_len}_pred${pred_len}] Missing Stage-1 checkpoint: ${stage1_ckpt}" >&2
    exit 1
  fi

  echo "[${DATASET}][${variant_tag}][seq${seq_len}_pred${pred_len}] Stage-2"
  python -u run.py \
    --task_name stage2_relation \
    --is_training 1 \
    --model_id "CARTS_stage2_${experiment}_${DATASET}_${pred_len}" \
    --model RelationStage2 \
    --learning_rate 1e-2 \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --base_head_mode shared_target_linear \
    --stage1_ckpt_path "${stage1_ckpt}" \
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
    --des "stage2_${experiment}_${DATASET}_seq${seq_len}_pred${pred_len}_topk10" \
    "${common_args[@]}" \
    2>&1 | tee -a "${log_path}"
}

for VARIANT in "${VARIANTS[@]}"; do
  case "${VARIANT}" in
    self)
      # top_n = 1: the target retrieves with its own history only.
      for PRED_LEN in "${PRED_LENS[@]}"; do
        run_one "selfonly" "${VARIANT_DIR}/self_only_top1.json" 1 "${PRED_LEN}"
      done
      ;;
    random)
      for SEED in "${RANDOM_SEEDS[@]}"; do
        for PRED_LEN in "${PRED_LENS[@]}"; do
          run_one "randomsrc_seed${SEED}" \
            "${VARIANT_DIR}/random_source_top${RELATION_TOP_N}_seed${SEED}.json" \
            "${RELATION_TOP_N}" "${PRED_LEN}"
        done
      done
      ;;
    *)
      echo "Unsupported variant: ${VARIANT}" >&2
      exit 2
      ;;
  esac
done

echo "[${DATASET}] Source-graph ablation logs: ${LOG_DIR}"
