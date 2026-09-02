#!/bin/bash
# Does a better Stage-1 retriever produce a better forecast?
#
# Stage-1 answered "score computation is the largest of the four axes" -- test
# Recall@10 rises up to +371% from cosine to asymmetric. None of that has been
# carried into Stage-2, and six earlier attempts to transfer a Stage-1 gain into
# forecasting produced nothing.
#
# It could not have transferred before this: Stage-2 loaded only the encoder from
# the Stage-1 checkpoint and scored candidates with a plain dot product, so an arm
# that learned an asymmetric metric had that half of its retriever dropped at the
# bridge. Stage-2 now carries the same comparison.
#
# Two training modes, because they answer different questions:
#   stage2   Stage-1 frozen, Stage-2 trained on top. Does the retriever help?
#   e2e      joint. Does the forecast loss reshape the retriever usefully?
#
# e2e rebuilds the key bank every epoch. Selection reads the bank while only the
# selected Top-K are re-encoded live, so without the rebuild an encoder that
# moves for ten epochs would still be choosing candidates with its starting
# embeddings -- the same staleness whose removal raised Stage-1 Recall@10 by
# 5.6-41.9%, and it would show up here as e2e losing to stage2 for a reason that
# has nothing to do with joint training. Frozen runs skip it: the bank cannot go
# stale when the encoder does not move.
#
# Baselines are cosine x kl in both modes, as specified.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

DATASET="${DATASET:-ETTh1}"
# Dataset-specific: root, csv, channel count, and the --data loader key. Weather
# and the other custom sets need an explicit relation graph; ETT ships one.
case "${DATASET}" in
  ETTh1|ETTh2|ETTm1|ETTm2)
    ROOT="${ROOT:-../Dataset/Time-Series-Library_dataset/ETT-small/}"
    LOADER="${LOADER:-${DATASET}}"; CHANNELS="${CHANNELS:-7}"; GRAPH="${GRAPH:-}" ;;
  weather)
    ROOT="${ROOT:-../Dataset/Time-Series-Library_dataset/weather/}"
    LOADER="${LOADER:-custom}"; CHANNELS="${CHANNELS:-21}"
    GRAPH="${GRAPH:-metrics/relation_graphs/weather/pearson_self_top1.json}" ;;
  *) echo "unknown DATASET=${DATASET}; set ROOT/LOADER/CHANNELS/GRAPH" >&2; exit 2 ;;
esac
GRAPH_ARGS=(); [ -n "${GRAPH}" ] && GRAPH_ARGS=(--relation_graph_path "${GRAPH}")
SEED="${SEED:-0}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})
MODES=(${MODES:-stage2 e2e})
# arm = score:loss   (score in cosine|asymmetric|pair2, loss in kl|wce)
ARMS=(${ARMS:-cosine:kl cosine:wce asymmetric:kl asymmetric:wce pair2:kl pair2:wce})
EPOCHS="${EPOCHS:-10}"; PATIENCE="${PATIENCE:-3}"
LOG_ROOT="${LOG_ROOT:-./logs/stage2_learned_score}"
# Distinguishes this sweep's artifacts from earlier ones. Empty reproduces the
# original names, so existing .done markers and checkpoints keep their meaning.
RUN_TAG="${RUN_TAG:-}"
mkdir -p "${LOG_ROOT}"

# Stage-1 checkpoint directory for a (score, loss) pair.
stage1_dir(){
  local score="$1" loss="$2" pred="$3" tag
  case "${score}:${loss}" in
    cosine:kl)      tag="${COS_KL_TAG:-e2_cos_kl}" ;;
    cosine:wce)     tag="${COS_WCE_TAG:-e2_cos_weighted_topk_ce}" ;;
    pair2:kl)       tag="e2_pair2_kl" ;;
    pair2:wce)      tag="e2_pair2_weighted_topk_ce" ;;
    # The E2 loss sweep trained both asymmetric arms under the same protocol, and
    # its kl checkpoints reproduce the on-demand ones to six decimals. Using them
    # keeps one source and is the only place pred192 exists.
    asymmetric:wce) tag="${ASYM_WCE_TAG:-e2_asym_weighted_topk_ce}" ;;
    asymmetric:kl)  tag="${ASYM_KL_TAG:-e2_asym_kl}" ;;
    *) return 1 ;;
  esac
  ls -d "./checkpoints/stage1/${LOADER}/seq${pred}_pred${pred}"/*"${tag}_${DATASET}"*/ 2>/dev/null | head -1
}


for PRED in "${PRED_LENS[@]}"; do
  DIR="${LOG_ROOT}/${DATASET}/pred${PRED}"; mkdir -p "${DIR}"
  for ARM in "${ARMS[@]}"; do
    SCORE="${ARM%%:*}"; LOSS="${ARM##*:}"
    # asymmetric checkpoints now come from the E2 sweep; nothing to train here.
    CKDIR="$(stage1_dir "${SCORE}" "${LOSS}" "${PRED}")"
    if [ -z "${CKDIR}" ]; then echo "[skip] no Stage-1 checkpoint for ${SCORE}/${LOSS}/pred${PRED}"; continue; fi
    CK="${CKDIR}checkpoint.pth"
    [ -f "${CK}" ] || { echo "[skip] missing ${CK}"; continue; }

    case "${SCORE}" in
      cosine)      SCORE_ARGS=() ;;
      asymmetric)  SCORE_ARGS=(--stage1_retrieval_metric asymmetric
                               --stage1_metric_output cosine --stage1_metric_layer_norm 0) ;;
      pair2)       SCORE_ARGS=(--stage1_retrieval_score pairwise_mlp
                               --stage1_pairwise_feature pair2) ;;
    esac

    for MODE in "${MODES[@]}"; do
      [ "${MODE}" = e2e ] && E2E=1 || E2E=0
      SHORT="${SCORE}_${LOSS}_${MODE}"
      LOG="${DIR}/${SHORT}.log"; MARKER="${DIR}/${SHORT}.done"
      if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then echo "[skip] ${PRED}/${SHORT}"; continue; fi
      echo "=============================================================="
      echo "[${DATASET}/pred${PRED}/${SHORT}] score=${SCORE} loss=${LOSS} mode=${MODE}"
      echo "=============================================================="
      {
        echo "### RUN ${DATASET}/pred${PRED}/${SHORT} $(date -Is)"
        echo "### stage1_ckpt ${CK}"
        python -u run.py --task_name stage2_relation --is_training 1 \
          --model_id "carts_s2ls${RUN_TAG}_${SHORT}_${DATASET}_${PRED}" --model RelationStage2 \
          --data "${LOADER}" --root_path "${ROOT}" \
          --data_path "${DATASET}.csv" --features M \
          --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
          --enc_in "${CHANNELS}" --batch_size 32 --num_workers 0 \
          --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
          --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft \
          --relation_input_space delta_last --relation_teacher_space delta_last \
          --source_mode auto --relation_top_n 1 --target_mode all \
          "${GRAPH_ARGS[@]}" \
          --relation_encoder_type mlp --relation_self_fill linear \
          --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
          --top_k 10 --tau_topk 0.1 \
          --fusion_mode residual --gate_mode scalar \
          --stage1_ckpt_path "${CK}" --stage1_encoder_init checkpoint \
          --freeze_stage1_encoder "$([ "${MODE}" = stage2 ] && echo 1 || echo 0)" \
          --stage2_e2e "${E2E}" \
          --refresh_memory_every_epoch "${E2E}" \
          --stage2_e2e_full_online "${E2E}" \
          "${SCORE_ARGS[@]}" \
          --des "s2ls${RUN_TAG}_${SHORT}_${DATASET}_sl${PRED}_pl${PRED}" || exit 21
        echo "### RUN COMPLETE ${DATASET}/pred${PRED}/${SHORT} $(date -Is)"
      } 2>&1 | tee "${LOG}"
      if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
        touch "${MARKER}"; echo "[ok] ${PRED}/${SHORT}"
      else
        echo "[FAILED] ${PRED}/${SHORT}"
      fi
    done
  done
done
echo "stage2 learned-score sweep finished $(date -Is)"
