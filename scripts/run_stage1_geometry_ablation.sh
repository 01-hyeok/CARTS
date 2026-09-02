#!/bin/bash
# STEP 2 -- cosine vs unnormalized L2 in the learned embedding space.
#
# On normalized vectors ||zq-zk||^2 = 2 - 2cos, so swapping the score alone
# changes nothing. The real ablation is dropping the explicit F.normalize before
# scoring, which --retrieval_similarity l2 already does inside RelationEncoder
# (LayerNorm and everything else in the encoder stays). Teacher, loss and
# candidate set are unchanged: geometry is the only variable.
#
# Runs pred96 first and only expands to every horizon if L2 clearly wins.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96})
OUT_CSV="${OUT_CSV:-./metrics/next_retrieval_diagnosis/stage1_geometry.csv}"
LOG_DIR="${LOG_DIR:-./logs/next_retrieval_diagnosis/geometry}"
EPOCHS="${STAGE1_EPOCHS:-10}"; PATIENCE="${STAGE1_PATIENCE:-5}"
MAX_QUERIES="${MAX_QUERIES:-512}"
mkdir -p "${LOG_DIR}" "$(dirname "${OUT_CSV}")"

for DATASET in "${DATASETS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    for SIM in cosine l2; do
      ID="carts_s1_geom${SIM}_${DATASET}_${PRED_LEN}"
      DES="s1_geom${SIM}_${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      SETTING="stage1_${ID}_RelationStage1_${DATASET}_ftM_sl${SEQ_LEN}_ll0_pl${PRED_LEN}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${DES}_0"
      CK="./checkpoints/stage1/${DATASET}/seq${SEQ_LEN}_pred${PRED_LEN}/${SETTING}/checkpoint.pth"
      echo "[geometry] ${DATASET}/pred${PRED_LEN} similarity=${SIM}"
      if [ "${FORCE:-0}" = "1" ] || [ ! -f "${CK}" ]; then
        python -u run.py --task_name stage1_relation --is_training 1 \
          --model_id "${ID}" --model RelationStage1 --data "${DATASET}" \
          --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
          --data_path "${DATASET}.csv" --features M \
          --seq_len "${SEQ_LEN}" --label_len 0 --pred_len "${PRED_LEN}" \
          --enc_in 7 --batch_size 32 --num_workers 0 \
          --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
          --patch_len 16 --stride 16 --seed "${SEED:-0}" --candidate_mask raft \
          --relation_input_space delta_last --relation_teacher_space delta_last \
          --source_mode auto --relation_top_n 1 --target_mode all \
          --relation_encoder_type mlp --relation_self_fill linear \
          --retrieval_similarity "${SIM}" \
          --learning_rate 1e-3 --train_epochs "${EPOCHS}" --patience "${PATIENCE}" \
          --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
          --teacher_mse_space normalized --stage1_teacher_mode mse --stage1_loss_mode kl \
          --stage1_checkpoint_metric recall10 --stage1_probe_vis 0 \
          --des "${DES}" > "${LOG_DIR}/${DATASET}_${PRED_LEN}_${SIM}.log" 2>&1 \
          || { echo "  [FAILED] training"; continue; }
      else
        echo "  [skip] checkpoint exists"
      fi
      [ -f "${CK}" ] || { echo "  [FAILED] missing checkpoint"; continue; }
      for SPLIT in train val test; do
        python -u scripts/diagnose_stage1_retrieval.py --checkpoint "${CK}" \
          --retriever learned --split "${SPLIT}" --max_queries "${MAX_QUERIES}" \
          --csv "${OUT_CSV}" > /dev/null 2>&1 || echo "  [FAILED] diagnostic ${SPLIT}"
      done
    done
  done
done
echo "csv: ${OUT_CSV}"
