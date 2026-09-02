#!/bin/bash
# End-to-end retrieval + score-separation losses.
#
# The bottleneck study ended on Case D: the pool is big enough, Top-K aggregation
# is not lossy, the utility target is broadly stable -- and better candidates still
# buy no forecast. The one unexplained link is that Stage-2's Top-K weights are
# uniform (weight entropy = ln(10)), so a better ordering has nowhere to go.
#
# Arms, in the order they must be run:
#
#   existing          stage-wise CARTS, unchanged                (canonical baseline)
#   tau003 / tau001   baseline retriever, sharper tau_topk       (CONTROL: is it just sharpness?)
#   e2e_forecast      EMA off, forecast loss reaches the encoder (E2E coupling alone)
#   e2e_kl            + existing KL                              (loss effect inside E2E)
#   e2e_rank          + RankNet                                  (order only)
#   e2e_margin        + residual margin ranking                  (MAIN: separation)
#   e2e_adaptive      + teacher-gap adaptive margin
#   stagewise_margin  Stage-1 ranking, Stage-2 separate          (CONTROL: rank vs coupling)
#
# The tau control matters as much as the main arm: weights are flat because a
# 0.004 score gap is divided by tau=0.10. Lowering tau produces sharper weights
# without changing the retriever at all, so without it "margin ranking helped"
# cannot be told apart from "sharper weights helped".
#
# Usage
#   SMOKE=1 bash scripts/run_e2e_margin_rank_ablation.sh
#   ARMS="existing e2e_forecast e2e_margin" bash scripts/run_e2e_margin_rank_ablation.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SMOKE="${SMOKE:-0}"; FORCE="${FORCE:-0}"; SEED="${SEED:-0}"
PRED=96; SEQ=96
if [ "${SMOKE}" = "1" ]; then
  DATASETS=(${DATASETS:-ETTh1}); S1_EPOCHS=1; S2_EPOCHS=1; PATIENCE=1; TAG="smoke"
  ARMS=(${ARMS:-e2e_margin})
else
  DATASETS=(${DATASETS:-ETTm1 ETTh1}); S1_EPOCHS="${S1_EPOCHS:-10}"
  S2_EPOCHS="${S2_EPOCHS:-10}"; PATIENCE="${PATIENCE:-5}"; TAG="full"
  ARMS=(${ARMS:-existing tau003 tau001 e2e_forecast e2e_kl e2e_rank e2e_margin e2e_adaptive stagewise_margin})
fi
RANK_WEIGHT="${RANK_WEIGHT:-0.1}"
RANK_MARGIN="${RANK_MARGIN:-0.05}"
LOG_ROOT="./logs/e2e_margin_rank_${TAG}"; OUT="./metrics/e2e_margin_rank"
mkdir -p "${LOG_ROOT}" "${OUT}"

baseline_stage1() { ls -d checkpoints/stage1/$1/seq${SEQ}_pred${PRED}/*s1_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1; }

# arm -> stage2 knobs
arm_e2e()        { case "$1" in e2e_*) echo 1 ;; *) echo 0 ;; esac; }
arm_rank()       { case "$1" in e2e_rank|e2e_rank_v2) echo ranknet ;; e2e_margin|e2e_margin_v2) echo margin ;; e2e_adaptive|e2e_adaptive_v2) echo adaptive_margin ;; *) echo none ;; esac; }
arm_rank_w()     { case "$1" in e2e_rank*|e2e_margin*|e2e_adaptive*) echo "${RANK_WEIGHT}" ;; *) echo 0.0 ;; esac; }
arm_kl()         { case "$1" in e2e_kl) echo 1.0 ;; *) echo 0.0 ;; esac; }
arm_freeze()     { case "$1" in e2e_*) echo 0 ;; *) echo 1 ;; esac; }
arm_ema()        { case "$1" in existing|tau*) echo 1 ;; *) echo 0 ;; esac; }
arm_tau()        { case "$1" in tau003) echo 0.003 ;; tau001) echo 0.001 ;; *) echo 0.10 ;; esac; }
# v2 corrections, applied only to the *_v2 arms. The audit of the v1 loss found
# 3.7% of its pairs inside the Top-K it was built to decompress, carrying 1.9% of
# the loss, against pairs already 24x wider -- and a margin 5.7x the gaps it was
# meant to widen. These three knobs are exactly those three findings.
arm_gamma()      { case "$1" in *_v2) echo "${RANK_GAMMA:-0.5}" ;; *) echo -1.0 ;; esac; }
arm_margin_mode(){ case "$1" in *_v2) echo topk_relative ;; *) echo absolute ;; esac; }
arm_sigma_mode() { case "$1" in e2e_rank_v2) echo topk_relative ;; *) echo fixed ;; esac; }
arm_margin_val() { case "$1" in *_v2) echo "${RANK_MARGIN_V2:-2.0}" ;; *) echo "${RANK_MARGIN}" ;; esac; }
# stagewise_margin trains Stage-1 with the residual teacher; every other arm
# reuses the canonical Future-KL Stage-1 so only the Stage-2 side differs.
arm_trains_stage1() { case "$1" in stagewise_margin) echo 1 ;; *) echo 0 ;; esac; }

FAILED=(); OK=(); SKIP=()
for DATASET in "${DATASETS[@]}"; do
  REF_S1="$(baseline_stage1 "${DATASET}")"
  [ -z "${REF_S1}" ] && { echo "[miss] baseline Stage-1 for ${DATASET}"; continue; }
  RESIDUAL_CACHE="./cache/residual_teacher/${DATASET}_pred${PRED}.pt"
  [ -f "${RESIDUAL_CACHE}" ] || { echo "[miss] residual cache ${RESIDUAL_CACHE}"; continue; }
  LOG_DIR="${LOG_ROOT}/${DATASET}"; mkdir -p "${LOG_DIR}"

  for ARM in "${ARMS[@]}"; do
    RUN_ID="${DATASET}/${ARM}"; LOG="${LOG_DIR}/${ARM}.log"; MARKER="${LOG_DIR}/${ARM}.done"
    if [ "${FORCE}" != "1" ] && [ -f "${MARKER}" ]; then
      echo "[skip] ${RUN_ID}"; SKIP+=("${RUN_ID}"); continue
    fi
    E2E="$(arm_e2e "${ARM}")"; RANK="$(arm_rank "${ARM}")"; RANK_W="$(arm_rank_w "${ARM}")"
    KL="$(arm_kl "${ARM}")"; FREEZE="$(arm_freeze "${ARM}")"; EMA="$(arm_ema "${ARM}")"
    TAU="$(arm_tau "${ARM}")"; TRAIN_S1="$(arm_trains_stage1 "${ARM}")"
    GAMMA="$(arm_gamma "${ARM}")"; MARGIN_MODE="$(arm_margin_mode "${ARM}")"
    SIGMA_MODE="$(arm_sigma_mode "${ARM}")"; MARGIN_VAL="$(arm_margin_val "${ARM}")"

    S1_CKPT="${REF_S1}"
    S2_ID="carts_em2_${ARM}_${DATASET}"; S2_DES="em2_${ARM}_${DATASET}"

    COMMON=(
      --data "${DATASET}" --root_path "../Dataset/Time-Series-Library_dataset/ETT-small/"
      --data_path "${DATASET}.csv" --features M
      --seq_len "${SEQ}" --label_len 0 --pred_len "${PRED}"
      --enc_in 7 --batch_size 32 --num_workers 0
      --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256
      --patch_len 16 --stride 16 --seed "${SEED}" --candidate_mask raft
      --relation_input_space delta_last --relation_teacher_space delta_last
      --relation_value_space delta_last
      --source_mode auto --relation_top_n 1 --target_mode all
      --relation_encoder_type mlp --relation_self_fill linear
    )

    echo "=============================================================="
    echo "[${RUN_ID}] e2e=${E2E} rank=${RANK} alpha=${RANK_W} kl=${KL} tau=${TAU} ema=${EMA} freeze=${FREEZE}"
    echo "=============================================================="
    {
      echo "### RUN ${RUN_ID} e2e=${E2E} rank=${RANK} alpha=${RANK_W} kl=${KL} tau=${TAU} $(date -Is)"

      if [ "${TRAIN_S1}" = "1" ]; then
        S1_ID="carts_em1_${ARM}_${DATASET}"; S1_DES="em1_${ARM}_${DATASET}"
        S1_SETTING="stage1_${S1_ID}_RelationStage1_${DATASET}_ftM_sl${SEQ}_ll0_pl${PRED}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${S1_DES}_0"
        S1_CKPT="./checkpoints/stage1/${DATASET}/seq${SEQ}_pred${PRED}/${S1_SETTING}/checkpoint.pth"
        echo "### STAGE1 (residual teacher, full bank)"
        python -u run.py --task_name stage1_relation --is_training 1 \
          --model_id "${S1_ID}" --model RelationStage1 "${COMMON[@]}" \
          --learning_rate 1e-3 --train_epochs "${S1_EPOCHS}" --patience "${PATIENCE}" \
          --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
          --teacher_mse_space normalized --stage1_teacher_mode mse --stage1_loss_mode kl \
          --stage1_residual_teacher 1 --stage1_residual_teacher_cache "${RESIDUAL_CACHE}" \
          --stage1_checkpoint_metric loss --stage1_probe_vis 0 --des "${S1_DES}" || exit 21
        [ -f "${S1_CKPT}" ] || { echo "### ERROR missing ${S1_CKPT}"; exit 22; }
      fi

      echo "### STAGE2 stage1_ckpt=${S1_CKPT}"
      python -u run.py --task_name stage2_relation --is_training 1 \
        --model_id "${S2_ID}" --model RelationStage2 \
        --base_head_mode shared_target_linear "${COMMON[@]}" \
        --learning_rate 1e-2 --train_epochs "${S2_EPOCHS}" --patience "${PATIENCE}" \
        --stage1_ckpt_path "${S1_CKPT}" --freeze_stage1_encoder "${FREEZE}" \
        --memory_cache_mode precompute --refresh_memory_every_epoch 1 \
        --memory_chunk_size 1024 --top_k 10 --tau_topk "${TAU}" \
        --relation_mixer_input retrieved --fusion_mode residual --gate_mode scalar \
        --stage2_e2e "${E2E}" --stage2_rank_loss "${RANK}" --stage2_rank_weight "${RANK_W}" \
        --stage2_rank_margin "${MARGIN_VAL}" --stage2_residual_cache "${RESIDUAL_CACHE}" \
        --stage2_rank_topk_gamma "${GAMMA}" \
        --stage2_rank_margin_mode "${MARGIN_MODE}" \
        --stage2_rank_sigma_mode "${SIGMA_MODE}" \
        --retrieval_kl_weight "${KL}" --retrieval_kl_teacher future_mse \
        --use_ema_teacher "${EMA}" \
        --des "${S2_DES}" || exit 23
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
for run in "${FAILED[@]:-}"; do [ -n "${run}" ] && echo "  FAILED ${run}"; done
