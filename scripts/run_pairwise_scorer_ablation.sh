#!/bin/bash
# Is cosine the bottleneck, or the data? A learnable pair score, tested directly.
#
# Over the full bank the incumbent cosine correlates with future-MSE at rho 0.61;
# inside its own Top-100 the correlation is 0.03 and the Top-10 candidates sit
# within 0.004 cosine of each other. Coarse retrieval works, fine ordering has no
# signal left in that space. These arms replace the fixed dot product with a
# learned function of the pair, keeping everything else identical.
#
#   e1_cosine_kl     cosine, KL                      (baseline, retrained here)
#   e2_pair2_kl      pair MLP on [z_q, z_k], KL      (score function effect)
#   e3_pair4_kl      + difference and magnitude, KL  (pair feature effect)
#   e4_pair2_wce     pair2, weighted Top-K CE        (objective effect)
#   e5_pair4_wce     pair4, weighted Top-K CE
#
# Every arm re-encodes the selected candidates with the current encoder, so the
# shared encoder gets gradient from the candidate branch as well as the query.
#
# MINING=reference (default) picks the training candidates with a frozen
# checkpoint, so every arm's loss runs over the *same* candidate ids and a
# difference between arms is a difference in score function or objective.
# MINING=self lets each arm mine with its own score -- a fair question about the
# finished system, but not a score-function ablation, because "better score" and
# "different candidates seen" would move together.
#
# e1_cosine_kl is retrained here rather than reusing selected100_reencode_kl: the
# training pool now carries random negatives, and the baseline has to see the
# same pool for the comparison to mean anything.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SMOKE="${SMOKE:-0}"; FORCE="${FORCE:-0}"; SEED="${SEED:-0}"
DATASET="${DATASET:-ETTh1}"
if [ "${SMOKE}" = "1" ]; then
  PRED_LENS=(${PRED_LENS:-96}); EPOCHS=1; PATIENCE=1; TAG="smoke"
  ARMS=(${ARMS:-e3_pair4_kl})
else
  PRED_LENS=(${PRED_LENS:-96 192 336 720}); EPOCHS="${EPOCHS:-10}"
  PATIENCE="${PATIENCE:-5}"; TAG="full"
  ARMS=(${ARMS:-e1_cosine_kl e2_pair2_kl e3_pair4_kl e4_pair2_wce e5_pair4_wce})
fi
MINE_TOP_M="${MINE_TOP_M:-100}"; ORACLE_INJECT_K="${ORACLE_INJECT_K:-10}"
# Mining returns only Top-M neighbours; evaluation ranks the whole bank. Cosine
# extrapolates there by construction, a learned scorer does not -- without these
# the pair scorer came out anti-correlated over the full memory (Spearman -0.46).
RANDOM_NEGATIVES="${RANDOM_NEGATIVES:-128}"
MINING="${MINING:-reference}"
reference_ckpt() {
  ls -d checkpoints/stage1/$1/seq$2_pred$2/*s1_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1
}
LOG_ROOT="./logs/pairwise_scorer_${TAG}"; mkdir -p "${LOG_ROOT}"

arm_score() { case "$1" in e1_*) echo cosine ;; *) echo pairwise_mlp ;; esac; }
arm_feat()  { case "$1" in *pair2*) echo pair2 ;; *) echo pair4 ;; esac; }
arm_loss()  { case "$1" in *_wce) echo weighted_topk_ce ;; *) echo kl ;; esac; }

FAILED=(); OK=(); SKIP=()
for PRED in "${PRED_LENS[@]}"; do
  SEQ="${PRED}"
  LOG_DIR="${LOG_ROOT}/${DATASET}/pred${PRED}"; mkdir -p "${LOG_DIR}"
  for ARM in "${ARMS[@]}"; do
    RUN_ID="${DATASET}/pred${PRED}/${ARM}"
    LOG="${LOG_DIR}/${ARM}.log"; MARKER="${LOG_DIR}/${ARM}.done"
    if [ "${FORCE}" != "1" ] && [ -f "${MARKER}" ]; then
      echo "[skip] ${RUN_ID}"; SKIP+=("${RUN_ID}"); continue
    fi
    SCORE="$(arm_score "${ARM}")"; FEAT="$(arm_feat "${ARM}")"; LOSS="$(arm_loss "${ARM}")"
    REF_ARGS=()
    if [ "${MINING}" = "reference" ]; then
      REF="$(reference_ckpt "${DATASET}" "${PRED}")"
      if [ -z "${REF}" ]; then
        echo "[miss] no frozen reference Stage-1 for ${DATASET}/${PRED}"; FAILED+=("${RUN_ID} (no reference)"); continue
      fi
      REF_ARGS=(--stage1_mining_score reference --stage1_pool_reference_ckpt "${REF}")
    fi
    ID="carts_ps_${ARM}_${DATASET}_${PRED}"; DES="ps_${ARM}_${DATASET}_sl${SEQ}_pl${PRED}"

    echo "=============================================================="
    echo "[${RUN_ID}] score=${SCORE} feature=${FEAT} loss=${LOSS} mining=${MINING} gpu=${CUDA_VISIBLE_DEVICES}"
    echo "=============================================================="
    {
      echo "### RUN ${RUN_ID} score=${SCORE} feature=${FEAT} loss=${LOSS} $(date -Is)"
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
        --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k "${ORACLE_INJECT_K}" \
        --stage1_retrieval_score "${SCORE}" --stage1_pairwise_feature "${FEAT}" \
        --stage1_candidate_subset_mode selected_reencode \
        --stage1_candidate_mine_top_m "${MINE_TOP_M}" \
        --stage1_candidate_oracle_inject_k "${ORACLE_INJECT_K}" \
        --stage1_candidate_random_negatives "${RANDOM_NEGATIVES}" \
        "${REF_ARGS[@]}" \
        --stage1_checkpoint_metric recall10 --stage1_probe_vis 0 \
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
