#!/bin/bash
# The score x loss cross on Weather.
#
# ETTh1 said score computation is the largest of the four axes, with the gain
# concentrated in asymmetry. Weather tests whether that holds where the shape of
# the problem changes: 21 channels instead of 7, and 36,696 candidates instead of
# 8,449 -- about 13x the per-step work.
#
# The MLP arm is absent and cannot be run here. Its memory scales with
# batch x candidates x feature-width x channels, which comes to ~124 GiB on a
# 79 GiB card (it reaches 12 of 21 channels before failing). The metric arms are
# 1.3 GiB. That asymmetry is the same one that makes the MLP unusable as an index
# at serving time, so the arms that survive here are the deployable ones.
#
# ETTh1's best arm was WCE+MLP, so its absence is a real gap in the comparison and
# is stated rather than worked around by shrinking the batch, which would break
# comparability with the ETTh1 numbers.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

DATASET=weather
ROOT=../Dataset/Time-Series-Library_dataset/weather/
CHANNELS=21
GRAPH=metrics/relation_graphs/weather/pearson_self_top1.json
SEED="${SEED:-0}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})
ARMS=(${ARMS:-cosine:kl cosine:wce asymmetric:kl asymmetric:wce})
EPOCHS="${EPOCHS:-10}"; PATIENCE="${PATIENCE:-5}"
LOG_ROOT="${LOG_ROOT:-./logs/weather_stage1}"; mkdir -p "${LOG_ROOT}"

for PRED in "${PRED_LENS[@]}"; do
  DIR="${LOG_ROOT}/pred${PRED}"; mkdir -p "${DIR}"
  for ARM in "${ARMS[@]}"; do
    SCORE="${ARM%%:*}"; LOSS_SHORT="${ARM##*:}"
    [ "${LOSS_SHORT}" = wce ] && LOSS=weighted_topk_ce || LOSS=kl
    case "${SCORE}" in
      cosine)     METRIC_ARGS=() ;;
      asymmetric) METRIC_ARGS=(--stage1_retrieval_metric asymmetric
                               --stage1_metric_output cosine --stage1_metric_layer_norm 0) ;;
    esac
    SHORT="${SCORE}_${LOSS_SHORT}"
    LOG="${DIR}/${SHORT}.log"; MARKER="${DIR}/${SHORT}.done"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then echo "[skip] ${PRED}/${SHORT}"; continue; fi
    echo "=============================================================="
    echo "[${DATASET}/pred${PRED}/${SHORT}] score=${SCORE} loss=${LOSS}"
    echo "=============================================================="
    {
      echo "### RUN ${DATASET}/pred${PRED}/${SHORT} $(date -Is)"
      python -u run.py --task_name stage1_relation --is_training 1 \
        --model_id "carts_w1_${SHORT}_${DATASET}_${PRED}" --model RelationStage1 \
        --data custom --root_path "${ROOT}" --data_path "${DATASET}.csv" --features M \
        --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
        --enc_in "${CHANNELS}" --batch_size 32 --num_workers 0 \
        --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
        --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft \
        --relation_input_space delta_last --relation_teacher_space delta_last \
        --source_mode auto --relation_top_n 1 --target_mode all \
        --relation_graph_path "${GRAPH}" \
        --relation_encoder_type mlp --relation_self_fill linear \
        --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
        --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
        --teacher_mse_space normalized --stage1_teacher_mode mse \
        --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k 10 \
        "${METRIC_ARGS[@]}" \
        --stage1_full_memory_gradient_mode full_online \
        --stage1_checkpoint_metric retrieved_mse10 --stage1_probe_vis 0 \
        --des "w1_${SHORT}_${DATASET}_sl${PRED}_pl${PRED}" || exit 21
      echo "### RUN COMPLETE ${DATASET}/pred${PRED}/${SHORT} $(date -Is)"
    } 2>&1 | tee "${LOG}"
    if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
      touch "${MARKER}"; echo "[ok] ${PRED}/${SHORT}"
    else
      echo "[FAILED] ${PRED}/${SHORT}"
    fi
  done
done
echo "weather stage1 sweep finished $(date -Is)"
