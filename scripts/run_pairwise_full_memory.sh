#!/bin/bash
# The pair scorer, trained on the same candidate set it is evaluated on.
#
# The earlier pairwise sweep mined 228 candidates per query for training and
# ranked all 8449 at evaluation. It lost at pred 96/192 and won at 336, and none
# of that is readable: a loss cannot be told apart from "the scorer never saw
# 8349 of the candidates it was tested on". Direct evidence of the size of that
# effect -- common mining without random negatives gave Spearman -0.457, adding
# 128 random negatives gave +0.504 on the same setup.
#
# Mining was there because materialising every pair looked prohibitive: a pair
# feature is 2-4x the embedding width and cannot be folded into a matmul. That
# was an estimate. Measured at B=32, N=8449, D=128 it is 1.62 GiB (pair2) /
# 2.39 GiB (pair4) per target channel, 15.3 GiB with all seven accumulated, and
# 0.12 s for forward+backward. The card has 80 GiB. The subset was never needed.
#
#   P0  cosine                    baseline, no learned score
#   P1  pairwise pair2            learned comparison, concatenation only
#   P2  pairwise pair4            plus the signed difference and its magnitude
#
# All three train and evaluate over the full memory, so a difference here is the
# score function and not the support. GRAD_MODE=full_online re-encodes every
# candidate each step, so the candidate side carries gradient everywhere rather
# than only inside a mined subset.
#
# Checkpoint is pre-registered on validation retrieved_future_mse@10 -- what
# Stage-2 consumes -- matching the full-memory metric sweep so the two are
# directly comparable. Side checkpoints on recall10 / ndcg10 are still saved.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SMOKE="${SMOKE:-0}"; FORCE="${FORCE:-0}"; SEED="${SEED:-0}"
DATASET="${DATASET:-ETTh1}"
GRAD_MODE="${GRAD_MODE:-full_online}"
LOSS="${LOSS:-weighted_topk_ce}"
CKPT_METRIC="${CKPT_METRIC:-retrieved_mse10}"
if [ "${SMOKE}" = "1" ]; then
  PRED_LENS=(${PRED_LENS:-96}); EPOCHS=1; PATIENCE=1; TAG="smoke"; ARMS=(${ARMS:-p2_pair4})
else
  PRED_LENS=(${PRED_LENS:-96 192 336 720}); EPOCHS="${EPOCHS:-10}"; PATIENCE="${PATIENCE:-5}"
  TAG="full"; ARMS=(${ARMS:-p0_cosine p1_pair2 p2_pair4})
fi
LOG_ROOT="./logs/pairwise_full_memory_${TAG}"; mkdir -p "${LOG_ROOT}"

FAILED=(); OK=(); SKIP=()
for PRED in "${PRED_LENS[@]}"; do
  SEQ="${PRED}"; LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED}"; mkdir -p "${LOG_DIR}"
  for ARM in "${ARMS[@]}"; do
    RUN_ID="${DATASET}/pred${PRED}/${ARM}"; LOG="${LOG_DIR}/${ARM}.log"; MARKER="${LOG_DIR}/${ARM}.done"
    if [ "${FORCE}" != "1" ] && [ -f "${MARKER}" ]; then
      echo "[skip] ${RUN_ID}"; SKIP+=("${RUN_ID}"); continue
    fi
    case "${ARM}" in
      p0_*) SCORE=cosine;      FEAT=pair4 ;;
      p1_*) SCORE=pairwise_mlp; FEAT=pair2 ;;
      *)    SCORE=pairwise_mlp; FEAT=pair4 ;;
    esac
    ID="carts_pfm_${ARM}_${DATASET}_${PRED}"; DES="pfm_${ARM}_${DATASET}_sl${SEQ}_pl${PRED}"
    echo "=============================================================="
    echo "[${RUN_ID}] score=${SCORE} feature=${FEAT} grad=${GRAD_MODE} gpu=${CUDA_VISIBLE_DEVICES}"
    echo "=============================================================="
    {
      echo "### RUN ${RUN_ID} score=${SCORE} feature=${FEAT} grad=${GRAD_MODE} $(date -Is)"
      python -u run.py --task_name stage1_relation --is_training 1 \
        --model_id "${ID}" --model RelationStage1 \
        --data "${DATASET}" \
        --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
        --data_path "${DATASET}.csv" --features M \
        --seq_len "${SEQ}" --label_len 0 --pred_len "${PRED}" \
        --enc_in 7 --batch_size 32 --num_workers 0 \
        --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
        --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft \
        --relation_input_space delta_last --relation_teacher_space delta_last \
        --source_mode auto --relation_top_n 1 --target_mode all \
        --relation_encoder_type mlp --relation_self_fill linear \
        --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
        --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
        --teacher_mse_space normalized --stage1_teacher_mode mse \
        --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k 10 \
        --stage1_retrieval_score "${SCORE}" --stage1_pairwise_feature "${FEAT}" \
        --stage1_full_memory_gradient_mode "${GRAD_MODE}" \
        --stage1_checkpoint_metric "${CKPT_METRIC}" --stage1_probe_vis 0 \
        --des "${DES}" || exit 21
      echo "### RUN COMPLETE ${RUN_ID} $(date -Is)"
    } 2>&1 | tee "${LOG}"
    STATUS="${PIPESTATUS[0]}"
    if [ "${STATUS}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
      touch "${MARKER}"; OK+=("${RUN_ID}"); echo "[ok] ${RUN_ID}"
    else
      FAILED+=("${RUN_ID} (exit ${STATUS})"); echo "[FAILED] ${RUN_ID} exit=${STATUS}"
    fi
  done
done
echo "completed=${#OK[@]} skipped=${#SKIP[@]} failed=${#FAILED[@]}"
for r in "${FAILED[@]:-}"; do [ -n "${r}" ] && echo "  FAILED ${r}"; done
