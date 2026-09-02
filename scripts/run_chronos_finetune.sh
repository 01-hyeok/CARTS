#!/bin/bash
set -uo pipefail

# Give the frozen Chronos retrieval path something to train.
#
# The chronos arms in the main study have zero trainable parameters in the
# retrieval path, so the encoder never adapts to "which past predicts a similar
# future". These arms attach the Stage-2 retrieval KL to it.
#
#   proj      Chronos stays frozen; a Linear(2D -> D) on top of the pooled
#             embedding is trained. Cheap: _chronos_pooled_cache keeps the frozen
#             T5 output, so a refresh only redoes the linear map.
#             The mode must be 'uniform', not 'cross_only': cross_only skips the
#             projection on self branches, and with relation_top_n=1 every branch
#             is a self branch, so the layer would never be applied and never
#             receive a gradient.
#   finetune  The T5 encoder itself is trained. Every refresh re-encodes the
#             whole candidate bank, so this is the expensive arm.
#   both      proj on top of a fine-tuned encoder.
#
# teacher is future_mse, not ema. The EMA teacher deep-copies stage1_encoder,
# which is None for a Chronos backbone, so build_teacher_key_bank returns None
# and the forward raises. future_mse scores candidates by the raw L2 between
# futures - no encoder involved - so it works here, and it beat ema in every
# arm of the main study.
#
# lambda must be > 0. Top-K is not differentiable, so with lambda=0 the only
# gradient reaching the retrieval path is through the alpha weights over
# candidates already chosen, and alpha is close to uniform. Every earlier Chronos
# fine-tune run in this repo used lambda=0, which is why none of them moved.
#
# Usage
#   bash scripts/run_chronos_finetune.sh                       # ETTh1, proj+finetune
#   ARMS=proj bash scripts/run_chronos_finetune.sh
#   DATASETS="ETTh1 ETTm1" PRED_LENS=96 bash scripts/run_chronos_finetune.sh
#   nohup bash scripts/run_chronos_finetune.sh > logs/chronos_ft_driver.log 2>&1 &

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
source "${VENV_ACTIVATE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-2}"
DATASETS=(${DATASETS:-ETTh1})
PRED_LENS=(${PRED_LENS:-96 192 336})
ARMS=(${ARMS:-proj finetune})
LAMBDA="${LAMBDA:-1.0}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
TAU_TOPK="${TAU_TOPK:-0.10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
CHRONOS_LR="${CHRONOS_LR:-1e-5}"     # the encoder needs a far smaller step than the heads
POOLING="${POOLING:-eos}"            # EOS pooling was worth ~3x recall over mean
FORCE="${FORCE:-0}"

echo "=== Chronos retrieval fine-tuning (self-only) ==="
echo "  GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "  datasets   : ${DATASETS[*]}"
echo "  arms       : ${ARMS[*]}"
echo "  pred_lens  : ${PRED_LENS[*]}"
echo "  lambda     : ${LAMBDA}   teacher: future_mse   pooling: ${POOLING}"
echo "  chronos_lr : ${CHRONOS_LR}"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo

for DS in "${DATASETS[@]}"; do
  case "${DS}" in
    ETTh1) DATA_PATH="ETTh1.csv" ;;
    ETTm1) DATA_PATH="ETTm1.csv" ;;
    *) echo "Unsupported dataset: ${DS}" >&2; exit 2 ;;
  esac
  LOG_DIR="${PROJECT_ROOT}/logs/${DS}/chronos_ft"
  mkdir -p "${LOG_DIR}"

  for ARM in "${ARMS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      SEQ_LEN="${PRED_LEN}"
      EXPERIMENT="chronos_ft_${ARM}"
      LOG_PATH="${LOG_DIR}/${ARM}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
      if [ "${FORCE}" != "1" ] && grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
        echo "[${DS}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] already finished, skipping"
        continue
      fi
      [ "${FORCE}" = "1" ] && : > "${LOG_PATH}"

      # T5 attention costs chunk * C * heads * L^2; a 1024-window chunk asks for
      # 36 GiB at L=336. Fine-tuning also holds activations, so shrink further.
      if   [ "${SEQ_LEN}" -le 128 ]; then CHUNK=512
      elif [ "${SEQ_LEN}" -le 256 ]; then CHUNK=128
      else                                CHUNK=48
      fi

      case "${ARM}" in
        proj)
          MODE_ARGS=(--chronos_finetune 0
                     --chronos_projection_dim 768
                     --chronos_projection_mode uniform
                     --chronos_projection_trainable 1
                     --chronos_dtype bfloat16)
          ;;
        finetune)
          MODE_ARGS=(--chronos_finetune 1
                     --chronos_lr "${CHRONOS_LR}"
                     --chronos_lr_decay 0
                     --chronos_grad_checkpointing 1
                     --chronos_projection_dim 0
                     --chronos_dtype float32)
          ;;
        both)
          MODE_ARGS=(--chronos_finetune 1
                     --chronos_lr "${CHRONOS_LR}"
                     --chronos_lr_decay 0
                     --chronos_grad_checkpointing 1
                     --chronos_projection_dim 768
                     --chronos_projection_mode uniform
                     --chronos_projection_trainable 1
                     --chronos_dtype float32)
          ;;
        *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
      esac

      echo "[${DS}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] chunk=${CHUNK} lambda=${LAMBDA}"
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
        --stage2_retrieval_backbone chronos \
        --stage1_encoder_init none \
        --chronos_model_id "${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}" \
        --chronos_pooling "${POOLING}" \
        --chronos_random_init 0 \
        --retrieval_kl_weight "${LAMBDA}" \
        --retrieval_kl_teacher future_mse \
        --freeze_stage1_encoder 1 \
        --memory_cache_mode precompute \
        --refresh_memory_every_epoch 1 \
        --memory_chunk_size "${CHUNK}" \
        --top_k 10 \
        --tau_topk "${TAU_TOPK}" \
        --stage2_relation_fusion gate \
        --relation_mixer_input retrieved \
        --fusion_mode raft_concat \
        --oracle_candidate_eval 1 \
        "${MODE_ARGS[@]}" \
        --des "stage2_${EXPERIMENT}_${DS}_seq${SEQ_LEN}_pred${PRED_LEN}_topk10" \
        2>&1 | tee -a "${LOG_PATH}"
      then
        echo "[FAILED] ${DS} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM}, see ${LOG_PATH}" \
          | tee -a "${LOG_DIR}/_failures.txt" >&2
        continue
      fi
    done
  done
done

echo
echo "  finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "logs: logs/<dataset>/chronos_ft/"
