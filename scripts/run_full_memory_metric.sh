#!/bin/bash
# Can a learnable-but-indexable metric find better Top-10 in the whole memory?
#
# The pair-MLP attempt trained on 228 mined candidates and was evaluated over
# 8449; it lost at short horizons, and that support mismatch is the likeliest
# reason. Every kind here stays bilinear in the two embeddings, so a bank scores
# as one matmul and training uses the same candidate support as evaluation.
#
#   F0  cosine       W = I                       (baseline, nothing learned)
#   F1  mahalanobis  (L z_q)^T (L z_i)           one shared space, W = L^T L PSD
#   F2  asymmetric   (W_q z_q)^T (W_k z_i)       separate spaces, W free
#
# These nest by expressiveness -- cosine < mahalanobis < asymmetric -- so a win
# is attributable to the constraint that was lifted. The earlier plan paired
# `bilinear` with `asymmetric`, which spans the same functions (W_q^T W_k ranges
# over every D x D matrix): that comparison measures optimisation under
# over-parameterisation, not expressive power. `bilinear` stays available in the
# CLI for the older runs but is not an arm here.
#
# METRIC_OUTPUT=cosine with no LayerNorm renormalises after projecting. It is
# what makes identity initialisation reproduce the baseline exactly (verified at
# construction: every run prints cosine_init_deviation, which must read 0.0e+00),
# and it holds all three arms in [-1, 1] so one tau_student means the same
# sharpness everywhere and no arm can lift a candidate by growing its norm
# instead of turning its direction.
#
# GRAD_MODE=full_online re-encodes the whole memory each step, so candidate-side
# gradient reaches every candidate and no negative-sampling hyperparameter enters
# the comparison. Measured at 21.5 s/epoch against 23.0 for selected_reencode --
# it is both the cleaner and the faster option here.
#
# Checkpoint is pre-registered on validation retrieved_future_mse@10: the quantity
# Stage-2 consumes. recall10 / ndcg10 / retMSE10 side checkpoints are still saved.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SMOKE="${SMOKE:-0}"; FORCE="${FORCE:-0}"; SEED="${SEED:-0}"
DATASET="${DATASET:-ETTh1}"
GRAD_MODE="${GRAD_MODE:-full_online}"
METRIC_OUTPUT="${METRIC_OUTPUT:-cosine}"
METRIC_LAYER_NORM="${METRIC_LAYER_NORM:-0}"
CKPT_METRIC="${CKPT_METRIC:-retrieved_mse10}"
LOSS="${LOSS:-weighted_topk_ce}"
if [ "${SMOKE}" = "1" ]; then
  PRED_LENS=(${PRED_LENS:-96}); EPOCHS=1; PATIENCE=1; TAG="smoke"; ARMS=(${ARMS:-f1_mahalanobis})
else
  PRED_LENS=(${PRED_LENS:-96 336}); EPOCHS="${EPOCHS:-10}"; PATIENCE="${PATIENCE:-5}"
  TAG="${TAG:-full}"; ARMS=(${ARMS:-f0_cosine f1_mahalanobis f2_asymmetric})
fi
# A run is identified by its gradient mode as well as its arm. Without this the
# bank-mode control reuses the full_online run's model_id and overwrites the
# checkpoint it exists to be compared against.
SUFFIX=""
if [ "${GRAD_MODE}" != "full_online" ]; then SUFFIX="_${GRAD_MODE}"; fi
LOG_ROOT="./logs/full_memory_metric_${TAG}"; mkdir -p "${LOG_ROOT}"

arm_metric() { case "$1" in f0_*) echo cosine ;; f1_*) echo mahalanobis ;; *) echo asymmetric ;; esac; }

FAILED=(); OK=(); SKIP=()
for PRED in "${PRED_LENS[@]}"; do
  SEQ="${PRED}"; LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED}"; mkdir -p "${LOG_DIR}"
  for ARM in "${ARMS[@]}"; do
    RUN_ID="${DATASET}/pred${PRED}/${ARM}"; LOG="${LOG_DIR}/${ARM}.log"; MARKER="${LOG_DIR}/${ARM}.done"
    if [ "${FORCE}" != "1" ] && [ -f "${MARKER}" ]; then
      echo "[skip] ${RUN_ID}"; SKIP+=("${RUN_ID}"); continue
    fi
    METRIC="$(arm_metric "${ARM}")"
    ID="carts_fm_${ARM}${SUFFIX}_${DATASET}_${PRED}"; DES="fm_${ARM}${SUFFIX}_${DATASET}_sl${SEQ}_pl${PRED}"
    echo "=============================================================="
    echo "[${RUN_ID}] metric=${METRIC} loss=${LOSS} grad=${GRAD_MODE} gpu=${CUDA_VISIBLE_DEVICES}"
    echo "=============================================================="
    {
      echo "### RUN ${RUN_ID} metric=${METRIC} loss=${LOSS} grad=${GRAD_MODE} $(date -Is)"
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
        --stage1_retrieval_metric "${METRIC}" \
        --stage1_metric_output "${METRIC_OUTPUT}" \
        --stage1_metric_layer_norm "${METRIC_LAYER_NORM}" \
        --stage1_full_memory_gradient_mode "${GRAD_MODE}" \
        --stage1_full_memory_hard_negatives 100 \
        --stage1_full_memory_random_negatives 128 \
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
