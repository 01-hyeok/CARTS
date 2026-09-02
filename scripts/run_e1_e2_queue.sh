#!/bin/bash
# E1 (capacity) then E2 (loss), both under the corrected Stage-1 conditions.
#
# Everything earlier in this study trained Stage-1 with the detached key bank, so
# the encoder only ever learned through the query branch. The control run just
# measured what that cost: Recall@10 rises 5.6-41.9% under full_online and the
# embedding's effective rank rises with it (13.1 -> 21.8 at pred 336). A capacity
# verdict reached with the candidate side frozen was reached under conditions
# where extra capacity had little to act on, which is why E1 repeats it.
#
# E1  d_model 32/128/256/512, judged on TRAIN Recall@10.
#     32 is a positive control, not a data point: if it is not clearly worse the
#     metric cannot see capacity and the rest of the sweep says nothing.
#
# E2  four listwise losses on cosine, two of them repeated on pair2 so the 2x2
#     answers whether the best loss depends on the score function -- the arms
#     already disagree on Spearman under a shared loss, so the two axes are not
#     independent. Plus two true pairwise-ranking arms.
#
#     The four listwise losses all take a softmax over the whole memory and differ
#     only in how they define positives, so the mismatch the shuffle test found
#     -- 97% of the loss mass sitting where Stage-2 never looks -- survives in all
#     of them. `kl_rank` is the one arm that leaves that structure: it compares
#     candidate against candidate directly, and only for pairs whose future-MSE
#     actually differs by min_mse_gap, so near-ties are not forced into an order.
#     That matters here because the Oracle's 10th and 11th differ by 1.4%.
#
#     Narrowing the softmax to a shortlist would target the mismatch more
#     directly, and is deliberately not included: it changes the task from "rank
#     the whole memory" to "rank a shortlist", which is the support narrowing that
#     made the earlier pairwise sweep unreadable.
#
#     The rank weight is swept rather than fixed. kl_rank is KL + w*rank, so a
#     single weight would confound the loss family with its mixing coefficient.
#
#     Both pair features are carried through the cross, not just pair2. The score
#     sweep left their order unsettled -- pair2 won at 96, pair4 at 192 and 336 --
#     and pair4 is the less stable of the two there (effective rank 15.7-19.1
#     against pair2's 22.7-23.2, and its best epoch was the first or second in
#     three of four horizons). Running both under each loss is what separates
#     "pair4 responds differently to the objective" from seed noise.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
DATASET="${DATASET:-ETTh1}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
STEPS="${STEPS:-E1 E2}"

if [[ " ${STEPS} " == *" E1 "* ]]; then
  echo "###################### E1 capacity ######################"
  DATASET="${DATASET}" PRED_LENS="${PRED_LENS}" GPU="${CUDA_VISIBLE_DEVICES}" \
  CONFIGS="32:64 128:256 256:512 512:1024" \
  TAG="_fo" LOSS="weighted_topk_ce" GRAD_MODE="full_online" CKPT_METRIC="retrieved_mse10" \
  OUT_CSV="./metrics/e1_capacity_full_online/capacity.csv" \
    bash scripts/run_stage1_capacity_horizons.sh || echo "[FAILED] E1"
fi

if [[ " ${STEPS} " == *" E2 "* ]]; then
  echo "###################### E2 loss ######################"
  LOG_ROOT="./logs/e2_loss/${DATASET}"; mkdir -p "${LOG_ROOT}"
  # arm = score:loss:feature
  # arm = score:loss:feature[:rank_weight]
  ARMS=(${ARMS:-cosine:kl:pair4 cosine:topk_coverage:pair4 cosine:weighted_topk_ce:pair4 \
                 cosine:kl_expected_mse:pair4 cosine:kl_rank:pair4:0.1 cosine:kl_rank:pair4:1.0 \
                 pairwise_mlp:kl:pair2 pairwise_mlp:weighted_topk_ce:pair2 \
                 pairwise_mlp:kl:pair4 pairwise_mlp:weighted_topk_ce:pair4 \
                 asymmetric:kl:pair4 asymmetric:weighted_topk_ce:pair4})
  for PRED in ${PRED_LENS}; do
    DIR="${LOG_ROOT}/pred${PRED}"; mkdir -p "${DIR}"
    for ARM in "${ARMS[@]}"; do
      IFS=':' read -r SCORE LOSS FEAT RANK_W <<< "${ARM}"
      RANK_W="${RANK_W:-0.1}"
      # asymmetric is a metric, not a pair scorer: it goes through
      # --stage1_retrieval_metric while the score stays cosine.
      METRIC_ARGS=(); SCORE_FLAG="${SCORE}"
      if [ "${SCORE}" = asymmetric ]; then
        SCORE_FLAG=cosine
        METRIC_ARGS=(--stage1_retrieval_metric asymmetric
                     --stage1_metric_output cosine --stage1_metric_layer_norm 0)
      fi
      SHORT="$([ "${SCORE}" = cosine ] && echo cos || ([ "${SCORE}" = asymmetric ] && echo asym || echo "${FEAT}"))_${LOSS}"
      RANK_ARGS=()
      if [ "${LOSS}" = "kl_rank" ]; then
        SHORT="${SHORT}_w${RANK_W}"
        RANK_ARGS=(--stage1_rank_weight "${RANK_W}" --stage1_rank_top_k 10 \
                   --stage1_rank_margin 0.1 --stage1_rank_min_mse_gap 0.0)
      fi
      LOG="${DIR}/${SHORT}.log"; MARKER="${DIR}/${SHORT}.done"
      if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ]; then echo "[skip] ${PRED}/${SHORT}"; continue; fi
      echo "=============================================================="
      echo "[E2 ${DATASET}/pred${PRED}/${SHORT}] score=${SCORE} loss=${LOSS}"
      echo "=============================================================="
      {
        echo "### RUN ${DATASET}/pred${PRED}/${SHORT} score=${SCORE} loss=${LOSS} $(date -Is)"
        python -u run.py --task_name stage1_relation --is_training 1 \
          --model_id "carts_e2_${SHORT}_${DATASET}_${PRED}" --model RelationStage1 \
          --data "${DATASET}" \
          --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
          --data_path "${DATASET}.csv" --features M \
          --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
          --enc_in 7 --batch_size 32 --num_workers 0 \
          --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
          --patch_len 16 --stride 16 --seed "${SEED:-0}" --candidate_mask raft \
          --relation_input_space delta_last --relation_teacher_space delta_last \
          --source_mode auto --relation_top_n 1 --target_mode all \
          --relation_encoder_type mlp --relation_self_fill linear \
          --learning_rate 1e-3 --train_epochs "${EPOCHS:-10}" --patience "${PATIENCE:-5}" \
          --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
          --teacher_mse_space normalized --stage1_teacher_mode mse \
          --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k 10 \
          --stage1_retrieval_score "${SCORE_FLAG}" --stage1_pairwise_feature "${FEAT}" \
          "${METRIC_ARGS[@]}" \
          --stage1_full_memory_gradient_mode full_online \
          "${RANK_ARGS[@]}" \
          --stage1_checkpoint_metric retrieved_mse10 --stage1_probe_vis 0 \
          --des "e2_${SHORT}_${DATASET}_sl${PRED}_pl${PRED}" || exit 21
        echo "### RUN COMPLETE ${DATASET}/pred${PRED}/${SHORT} $(date -Is)"
      } 2>&1 | tee "${LOG}"
      if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "${LOG}"; then
        touch "${MARKER}"; echo "[ok] ${PRED}/${SHORT}"
      else
        echo "[FAILED] ${PRED}/${SHORT}"
      fi
    done
  done
fi
echo "queue finished $(date -Is)"
