#!/bin/bash
set -euo pipefail

# Self-only retrieval: how good can the Top-K get without any cross-channel
# relation? relation_top_n=1 keeps only the self source, because the relation
# graph ranks self first and top_n counts the total source list including self.
# That makes this the same retrieval unit TS-RAG uses (per-variable, retrieved
# from that variable's own history), so the arms differ only in the retriever.
#
#   no_retrieval base forecast head only, retrieval switched off. The floor every
#                other arm has to beat. Needed here because the earlier
#                no-retrieval baselines only cover pred_len 96/192/336/720.
#   identity     no encoder at all: the L2-normalised raw window is the key, so
#                the score is plain cosine between windows. This is the control
#                that says whether learning a representation buys anything.
#   random       the same MLP encoder as the trained arms but left at its random
#                init and frozen. Sitting between identity and the trained arms,
#                it separates "having this architecture" from "training it".
#   identity_l2  identity, scored with the negative mean squared distance on
#                un-normalised windows instead of cosine
#   chronos      frozen chronos-t5-base, nothing in the retrieval path trains
#   chronos_l2   chronos scored with l2 on un-normalised embeddings, which is
#                what TS-RAG retrieves with (faiss IndexFlatL2). Normalising
#                throws away the embedding norm, so this is the arm that says
#                whether that norm carried anything useful.
#   2stage_ema   Stage-1 learns the encoder against an EMA-target teacher,
#                Stage-2 freezes it
#   2stage_mse   same, future-MSE teacher
#   e2e_ema      no Stage-1; the encoder trains inside Stage-2 against the
#                EMA retrieval KL
#   e2e_mse      same, future-MSE teacher
#
# The e2e arms need lambda > 0: Top-K is not differentiable, so without the KL
# the encoder can only reweight candidates it already picked and barely trains.
# Primary metric is student_relation_oracle_recall_at_{1,5,10}, not MSE.

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

DATASET="${1:-ETTh1}"
case "${DATASET}" in
  ETTh1) DATA_PATH="ETTh1.csv" ;;
  ETTm1) DATA_PATH="ETTm1.csv" ;;
  *) echo "Usage: $0 {ETTh1|ETTm1} [arm ...]" >&2; exit 2 ;;
esac
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/${DATASET}/self_topk"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# 720 is left out: chronos-t5-base truncates its input to context_length=512, so
# at seq_len 720 it would retrieve from the last 512 steps while every other arm
# read all 720, and the arms would stop being comparable.
PRED_LENS=(${PRED_LENS:-96 192 336})
if [ "$#" -gt 0 ]; then ARMS=("$@"); else ARMS=(${ARMS:-no_retrieval identity identity_l2 random random_l2 chronos chronos_l2 2stage_ema 2stage_ema_l2 2stage_mse 2stage_mse_l2 e2e_ema e2e_ema_l2 e2e_mse e2e_mse_l2}); fi
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
STUDENT_TEMP="${STUDENT_TEMP:-0.10}"
TEACHER_TEMP="${TEACHER_TEMP:-0.07}"
TAU_TOPK="${TAU_TOPK:-0.10}"
TOP_K="${TOP_K:-10}"
LAMBDA="${LAMBDA:-1.0}"
# FORCE=1 re-runs everything: finished logs are truncated instead of skipped and
# a two-stage arm retrains Stage-1 instead of reusing the checkpoint on disk.
FORCE="${FORCE:-0}"
RELATION_TOP_N=1          # self source only - this is what makes the run self-only

# run.py shortens long experiment names with its own sha256 rule, so the
# checkpoint directory is found by globbing on the model id instead of trying to
# reproduce that rule here - the model id alone is unique per arm and pred_len.
find_stage1_ckpt() {
  ls -1 "$1"/stage1_"$2"_*/checkpoint.pth 2>/dev/null | head -1 || true
}

# Arm is the outer loop so the ARMS order is respected globally: every cheap
# frozen arm finishes across all pred_lens before the expensive end-to-end ones
# start, instead of e2e at pred_len 96 running ahead of no_retrieval at 192.
for ARM in "${ARMS[@]}"; do
for PRED_LEN in "${PRED_LENS[@]}"; do
  SEQ_LEN="${PRED_LEN}"

  COMMON_ARGS=(
    --data "${DATASET}"
    --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/
    --data_path "${DATA_PATH}"
    --features M
    --seq_len "${SEQ_LEN}"
    --label_len 0
    --pred_len "${PRED_LEN}"
    --enc_in 7
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
    --relation_top_n "${RELATION_TOP_N}"
    --target_mode all
    --relation_encoder_type mlp
    --relation_self_fill linear
  )

  STAGE2_COMMON=(
    --task_name stage2_relation
    --is_training 1
    --model RelationStage2
    --learning_rate 1e-2
    --train_epochs "${TRAIN_EPOCHS}"
    --patience "${PATIENCE}"
    --base_head_mode shared_target_linear
    --memory_cache_mode precompute
    --memory_chunk_size 1024
    --top_k "${TOP_K}"
    --tau_topk "${TAU_TOPK}"
    --stage2_relation_fusion gate
    --relation_mixer_input retrieved
    --fusion_mode raft_concat
    --oracle_candidate_eval 1
  )

    # A non-default tau_topk gets its own experiment name so a sweep over it
    # cannot overwrite the tau=0.10 logs and checkpoints already on disk.
    TAU_TAG=""
    if [ "$(awk -v t="${TAU_TOPK}" 'BEGIN{print (t==0.10) ? 1 : 0}')" != "1" ]; then
      TAU_TAG="_tauk${TAU_TOPK//./p}"
    fi
    EXPERIMENT="self_${ARM}${TAU_TAG}"
    LOG_PATH="${LOG_DIR}/${ARM}${TAU_TAG}_seq${SEQ_LEN}_pred${PRED_LEN}.log"
    if [ "${FORCE}" != "1" ] && grep -q 'Stage2 Test Final' "${LOG_PATH}" 2>/dev/null; then
      echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] already finished, skipping"
      continue
    fi
    # Stage-2 appends to the log, so a forced re-run has to start from an empty
    # file or the old and new runs end up interleaved in it.
    [ "${FORCE}" = "1" ] && : > "${LOG_PATH}"

    # ---- Stage-1, only for the two-stage arms -------------------------------
    STAGE1_ARGS=()
    case "${ARM}" in
      2stage_mse|2stage_ema|2stage_mse_l2|2stage_ema_l2)
        case "${ARM}" in 2stage_mse*) S1_TEACHER=mse ;; *) S1_TEACHER=ema_target ;; esac
        case "${ARM}" in *_l2) SIM=l2 ;; *) SIM=cosine ;; esac
        # tau_topk only affects Stage-2, so Stage-1 is named without the tau tag
        # and one checkpoint is shared across a tau sweep. Retraining it per tau
        # would waste time and, worse, fold Stage-1 init variance into the tau
        # comparison.
        S1_MODEL_ID="CARTS_stage1_self_${ARM}_${DATASET}_${PRED_LEN}"
        S1_DES="stage1_self_${ARM}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}"
        S1_DIR="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}"
        S1_CKPT="$(find_stage1_ckpt "${S1_DIR}" "${S1_MODEL_ID}")"

        if [ "${FORCE}" != "1" ] && [ -n "${S1_CKPT}" ]; then
          echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] Stage-1 checkpoint exists, reusing: ${S1_CKPT}"
        else
          echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] Stage-1 (teacher=${S1_TEACHER})"
          # Wrapped so a Stage-1 crash records the arm and moves on instead of
          # taking the rest of the sweep down with it.
          if ! python -u run.py \
            --task_name stage1_relation \
            --is_training 1 \
            --model RelationStage1 \
            --model_id "${S1_MODEL_ID}" \
            --learning_rate 1e-3 \
            --train_epochs "${TRAIN_EPOCHS}" \
            --patience "${PATIENCE}" \
            --tau_student "${STUDENT_TEMP}" \
            --tau_teacher "${TEACHER_TEMP}" \
            --teacher_mse_space normalized \
            --stage1_teacher_mode "${S1_TEACHER}" \
            --retrieval_similarity "${SIM}" \
            --stage1_loss_mode kl \
            --stage1_probe_vis 0 \
            --stage1_ema_momentum_base 0.99 \
            --stage1_ema_momentum_final 0.9995 \
            --des "${S1_DES}" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "${LOG_PATH}"
          then
            echo "[FAILED] ${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM}: Stage-1 exited non-zero, see ${LOG_PATH}" \
              | tee -a "${LOG_DIR}/_failures.txt" >&2
            continue
          fi
          S1_CKPT="$(find_stage1_ckpt "${S1_DIR}" "${S1_MODEL_ID}")"
          if [ -z "${S1_CKPT}" ]; then
            # One broken arm must not take the other 45 runs down with it.
            echo "[FAILED] ${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM}: no Stage-1 checkpoint under ${S1_DIR}" \
              | tee -a "${LOG_DIR}/_failures.txt" >&2
            continue
          fi
        fi
        # Stage-2 must score with the metric Stage-1 was fit under, otherwise the
        # encoder is evaluated on a similarity it never optimised for.
        STAGE1_ARGS=(--stage2_retrieval_backbone stage1
                     --stage1_encoder_init checkpoint
                     --stage1_ckpt_path "${S1_CKPT}"
                     --stage2_retrieval_encoder online
                     --retrieval_similarity "${SIM}"
                     --freeze_stage1_encoder 1
                     --refresh_memory_every_epoch 0)
        ;;
    esac

    # ---- Stage-2 ------------------------------------------------------------
    case "${ARM}" in
      no_retrieval)
        # Retrieval off entirely, so there is no relation graph to load and no
        # oracle candidates to score; source_mode=all and oracle_candidate_eval=0
        # override what COMMON_ARGS/STAGE2_COMMON set for the retrieval arms.
        BACKBONE_ARGS=(--stage1_encoder_init none
                       --freeze_stage1_encoder 1
                       --disable_retrieval 1
                       --source_mode all
                       --oracle_candidate_eval 0
                       --use_aux_base_loss 0
                       --use_aux_ret_loss 0
                       --beta_entropy_reg 0
                       --refresh_memory_every_epoch 0)
        ;;
      identity|identity_l2)
        [ "${ARM}" = "identity_l2" ] && SIM=l2 || SIM=cosine
        BACKBONE_ARGS=(--stage2_retrieval_backbone identity
                       --stage1_encoder_init none
                       --freeze_stage1_encoder 1
                       --retrieval_similarity "${SIM}"
                       --refresh_memory_every_epoch 0)
        ;;
      random|random_l2)
        # No Stage-1 checkpoint is read; the encoder keeps its random init and
        # freeze_stage1_encoder=1 detaches the query, so nothing in the
        # retrieval path receives a gradient.
        [ "${ARM}" = "random_l2" ] && SIM=l2 || SIM=cosine
        BACKBONE_ARGS=(--stage2_retrieval_backbone stage1
                       --stage1_encoder_init random
                       --freeze_stage1_encoder 1
                       --retrieval_similarity "${SIM}"
                       --refresh_memory_every_epoch 0)
        ;;
      chronos|chronos_l2|chronos_eos|chronos_tsrag)
        # Three axes separate this repo's Chronos setup from TS-RAG's. Distance
        # and pooling get their own arm so a loss can be attributed; chronos_tsrag
        # moves all three at once for the faithful reproduction.
        #   chronos        mean pooling, cosine, delta_last   (repo default)
        #   chronos_l2     ... l2                             (isolates distance)
        #   chronos_eos    ... EOS pooling                    (isolates pooling)
        #   chronos_tsrag  EOS + l2 + absolute                (TS-RAG as published)
        case "${ARM}" in
          chronos)       SIM=cosine; POOL=mean; INSPACE=delta_last ;;
          chronos_l2)    SIM=l2;     POOL=mean; INSPACE=delta_last ;;
          chronos_eos)   SIM=cosine; POOL=eos;  INSPACE=delta_last ;;
          chronos_tsrag) SIM=l2;     POOL=eos;  INSPACE=absolute   ;;
        esac
        # T5 self-attention over a chunk costs chunk*C heads*L^2. At L=336 a
        # 1024-window chunk asks for 36 GiB and OOMs, so the chunk shrinks with
        # the sequence length to keep that term roughly constant.
        if   [ "${SEQ_LEN}" -le 128 ]; then CHRONOS_CHUNK=1024
        elif [ "${SEQ_LEN}" -le 256 ]; then CHRONOS_CHUNK=256
        else                                CHRONOS_CHUNK=96
        fi
        BACKBONE_ARGS=(--retrieval_similarity "${SIM}"
                       --chronos_pooling "${POOL}"
                       --relation_input_space "${INSPACE}"
                       --memory_chunk_size "${CHRONOS_CHUNK}"
                       --stage2_retrieval_backbone chronos
                       --stage1_encoder_init none
                       --chronos_model_id "${CHRONOS_MODEL_ID:-amazon/chronos-t5-base}"
                       --chronos_dtype bfloat16
                       --chronos_finetune 0
                       --chronos_random_init 0
                       --freeze_stage1_encoder 1
                       --refresh_memory_every_epoch 0)
        ;;
      2stage_mse|2stage_ema|2stage_mse_l2|2stage_ema_l2)
        BACKBONE_ARGS=("${STAGE1_ARGS[@]}")
        ;;
      e2e_mse|e2e_ema|e2e_mse_l2|e2e_ema_l2)
        case "${ARM}" in e2e_mse*) KL_TEACHER=future_mse ;; *) KL_TEACHER=ema ;; esac
        case "${ARM}" in *_l2) SIM=l2 ;; *) SIM=cosine ;; esac
        # The retrieval KL scores with the same similarity the Top-K uses, so an
        # l2 arm trains and retrieves under the same metric - unlike the
        # two-stage arms, where Stage-1 would still be fit under cosine.
        BACKBONE_ARGS=(--stage2_retrieval_backbone stage1
                       --stage1_encoder_init random
                       --freeze_stage1_encoder 0
                       --retrieval_kl_weight "${LAMBDA}"
                       --retrieval_kl_teacher "${KL_TEACHER}"
                       --retrieval_similarity "${SIM}"
                       --tau_student "${STUDENT_TEMP}"
                       --tau_teacher "${TEACHER_TEMP}"
                       --refresh_memory_every_epoch 1)
        ;;
      *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
    esac

    echo "[${DATASET}][seq${SEQ_LEN}_pred${PRED_LEN}][${ARM}] Stage-2 (self-only, top_n=1)"
    # BACKBONE_ARGS goes last on purpose: argparse keeps the last occurrence, so
    # an arm can override anything COMMON_ARGS/STAGE2_COMMON set (no_retrieval
    # needs that for --source_mode and --oracle_candidate_eval).
    if ! python -u run.py \
      "${STAGE2_COMMON[@]}" \
      --model_id "CARTS_stage2_${EXPERIMENT}_${DATASET}_${PRED_LEN}" \
      --des "stage2_${EXPERIMENT}_${DATASET}_seq${SEQ_LEN}_pred${PRED_LEN}_topk${TOP_K}" \
      "${COMMON_ARGS[@]}" \
      "${BACKBONE_ARGS[@]}" \
      2>&1 | tee -a "${LOG_PATH}"
    then
      echo "[FAILED] ${DATASET} seq${SEQ_LEN}_pred${PRED_LEN} ${ARM}: Stage-2 exited non-zero, see ${LOG_PATH}" \
        | tee -a "${LOG_DIR}/_failures.txt" >&2
      continue
    fi
done   # PRED_LEN
done   # ARM

if [ -s "${LOG_DIR}/_failures.txt" ]; then
  echo
  echo "=== failed runs (${DATASET}) ==="
  cat "${LOG_DIR}/_failures.txt"
fi

echo "self-only Top-K logs: ${LOG_DIR}"
