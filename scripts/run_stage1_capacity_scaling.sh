#!/bin/bash
# STEP 4 -- is the full-scale train-fitting failure a capacity problem?
#
# ETTh1/pred96 only, full-bank KL, everything else identical. The MLP relation
# encoder is Linear(input, d_ff) -> GELU -> Linear(d_ff, d_model), so scaling
# d_model alone would only widen the output projection; d_ff is scaled with it
# to actually grow the backbone.
#
#   train R@10 rises with capacity  -> capacity-limited, generalization is next
#   train R@10 flat                 -> not capacity; the target itself is suspect
#
# No Stage-2 here: only whether the encoder can fit its own training queries.
#
# The first run of this used the detached key bank, so the encoder was trained
# only through the query branch -- with the candidate side frozen there may be
# nothing for extra capacity to do, which would produce a flat curve for a reason
# that has nothing to do with capacity. TAG/LOSS/GRAD_MODE/CKPT_METRIC are
# overridable so the same sweep can be rerun under the corrected conditions
# without overwriting the original: TAG feeds the model_id, so a tagged rerun
# writes its own checkpoints and its own CSV.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASET="${DATASET:-ETTh1}"
PRED_LEN="${PRED_LEN:-96}"
SEQ_LEN="${PRED_LEN}"
# "d_model:d_ff" pairs -- the hidden width grows with the embedding width.
CONFIGS=(${CONFIGS:-128:256 256:512 512:1024})
EPOCHS="${STAGE1_EPOCHS:-10}"
PATIENCE="${STAGE1_PATIENCE:-5}"
SEED="${SEED:-0}"
MAX_QUERIES="${MAX_QUERIES:-512}"
TAG="${TAG:-}"                       # non-empty keeps this sweep's runs separate
LOSS="${LOSS:-kl}"
GRAD_MODE="${GRAD_MODE:-bank}"
CKPT_METRIC="${CKPT_METRIC:-recall10}"
LOG_DIR="${LOG_DIR:-./logs/stage1_capacity_scaling}"
OUT_CSV="${OUT_CSV:-./metrics/stage1_capacity_scaling.csv}"
mkdir -p "${LOG_DIR}" "$(dirname "${OUT_CSV}")"
# Only wipe on an explicit rerun: a horizon sweep appends to one CSV, and
# silently deleting it would destroy the settings already measured.
[ "${FORCE:-0}" = "1" ] && rm -f "${OUT_CSV}"

for CONFIG in "${CONFIGS[@]}"; do
  D_MODEL="${CONFIG%%:*}"
  D_FF="${CONFIG##*:}"
  ID="carts_s1_cap${D_MODEL}${TAG}_${DATASET}_${PRED_LEN}"
  DES="s1_cap${D_MODEL}${TAG}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
  SETTING="stage1_${ID}_RelationStage1_${DATASET}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm${D_MODEL}_nh4_el2_dl1_df${D_FF}_expand2_dc4_fc1_ebtimeF_dtTrue_${DES}_0"
  CK="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}/${SETTING}/checkpoint.pth"

  echo "=============================================================="
  echo "[capacity] d_model=${D_MODEL} d_ff=${D_FF}"
  echo "=============================================================="
  if [ "${FORCE:-0}" = "1" ] || [ ! -f "${CK}" ]; then
    python -u run.py \
      --task_name stage1_relation --is_training 1 \
      --model_id "${ID}" --model RelationStage1 \
      --data "${DATASET}" \
      --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path "${DATASET}.csv" \
      --features M --seq_len "${SEQ_LEN}" --label_len 0 --pred_len "${PRED_LEN}" \
      --enc_in 7 --batch_size 32 --num_workers 0 \
      --d_model "${D_MODEL}" --d_ff "${D_FF}" --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed "${SEED}" \
      --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_loss_mode "${LOSS}" --stage1_coverage_top_k 10 \
      --stage1_full_memory_gradient_mode "${GRAD_MODE}" \
      --stage1_checkpoint_metric "${CKPT_METRIC}" --stage1_probe_vis 0 \
      --des "${DES}" 2>&1 | tee "${LOG_DIR}/d${D_MODEL}.log"
  else
    echo "[skip] checkpoint exists: ${CK}"
  fi

  if [ ! -f "${CK}" ]; then
    echo "[FAILED] missing checkpoint for d_model=${D_MODEL}"
    continue
  fi
  for SPLIT in train val; do
    python -u scripts/diagnose_stage1_retrieval.py \
      --checkpoint "${CK}" --retriever learned --split "${SPLIT}" \
      --max_queries "${MAX_QUERIES}" --csv "${OUT_CSV}" > /dev/null 2>&1 \
      || echo "[FAILED] diagnostic ${SPLIT} d_model=${D_MODEL}"
  done
done

echo "csv: ${OUT_CSV}"
