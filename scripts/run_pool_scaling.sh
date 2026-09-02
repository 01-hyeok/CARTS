#!/bin/bash
# EXPERIMENT 2 -- does the residual teacher's advantage survive a bigger pool?
#
# The teacher ablation ran on a fixed Top-100 pool and paid a large restriction
# cost. This scales the pool up with the teacher as the only other variable, so
# "the teacher helps" and "the pool hurts" stop being one number.
#
# At every M, Future-KL and Residual-KL are trained on the SAME candidate ids --
# mined by a frozen reference encoder that never moves during training.
# M=0 means the full memory bank, which the residual teacher can reach because
# its scores are one matmul rather than a stored per-pool matrix.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

SMOKE="${SMOKE:-0}"; FORCE="${FORCE:-0}"; SEED="${SEED:-0}"
if [ "${SMOKE}" = "1" ]; then
  DATASETS=(${DATASETS:-ETTh1}); POOLS=(${POOLS:-100}); TEACHERS=(${TEACHERS:-residual})
  STAGE1_EPOCHS=1; STAGE2_EPOCHS=1; PATIENCE=1; TAG="smoke"
else
  DATASETS=(${DATASETS:-ETTh1 ETTm1}); POOLS=(${POOLS:-100 500 1000 2000 0})
  TEACHERS=(${TEACHERS:-future residual})
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"; STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
  PATIENCE="${PATIENCE:-5}"; TAG="full"
fi
PRED=96; SEQ=96
LOG_ROOT="./logs/pool_scaling_${TAG}"; OUT="./metrics/retrieval_bottleneck"
mkdir -p "${LOG_ROOT}" "${OUT}"

reference_stage1() {
  ls -d checkpoints/stage1/$1/seq${SEQ}_pred${PRED}/*s1_full_bank_kl*/checkpoint.pth 2>/dev/null | head -1
}

FAILED=(); OK=(); SKIP=()
for DATASET in "${DATASETS[@]}"; do
  REF_S1="$(reference_stage1 "${DATASET}")"
  [ -z "${REF_S1}" ] && { echo "[miss] reference Stage-1 for ${DATASET}"; continue; }
  RESIDUAL_CACHE="./cache/residual_teacher/${DATASET}_pred${PRED}.pt"
  [ -f "${RESIDUAL_CACHE}" ] || { echo "[miss] residual cache ${RESIDUAL_CACHE}"; continue; }
  LOG_DIR="${LOG_ROOT}/${DATASET}"; mkdir -p "${LOG_DIR}"

  for POOL in "${POOLS[@]}"; do
    for TEACHER in "${TEACHERS[@]}"; do
      ARM="${TEACHER}_kl_m${POOL}"
      RUN_ID="${DATASET}/${ARM}"; LOG="${LOG_DIR}/${ARM}.log"; MARKER="${LOG_DIR}/${ARM}.done"
      if [ "${FORCE}" != "1" ] && [ -f "${MARKER}" ]; then
        echo "[skip] ${RUN_ID}"; SKIP+=("${RUN_ID}"); continue
      fi
      RESIDUAL_FLAG=0; [ "${TEACHER}" = "residual" ] && RESIDUAL_FLAG=1

      S1_ID="carts_ps1_${ARM}_${DATASET}"; S1_DES="ps1_${ARM}_${DATASET}"
      S2_ID="carts_ps2_${ARM}_${DATASET}"; S2_DES="ps2_${ARM}_${DATASET}"
      S1_SETTING="stage1_${S1_ID}_RelationStage1_${DATASET}_ftM_sl${SEQ}_ll0_pl${PRED}_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_${S1_DES}_0"
      S1_CKPT="./checkpoints/stage1/${DATASET}/seq${SEQ}_pred${PRED}/${S1_SETTING}/checkpoint.pth"

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
      echo "[${RUN_ID}] teacher=${TEACHER} pool=${POOL} gpu=${CUDA_VISIBLE_DEVICES}"
      echo "=============================================================="
      {
        echo "### RUN ${RUN_ID} teacher=${TEACHER} pool=${POOL} $(date -Is)"
        echo "### STAGE1"
        python -u run.py --task_name stage1_relation --is_training 1 \
          --model_id "${S1_ID}" --model RelationStage1 "${COMMON[@]}" \
          --learning_rate 1e-3 --train_epochs "${STAGE1_EPOCHS}" --patience "${PATIENCE}" \
          --top_k 10 --tau_student 0.10 --tau_teacher 0.1 \
          --teacher_mse_space normalized --stage1_teacher_mode mse --stage1_loss_mode kl \
          --stage1_residual_teacher "${RESIDUAL_FLAG}" \
          --stage1_residual_teacher_cache "${RESIDUAL_CACHE}" \
          --stage1_pool_size "${POOL}" \
          --stage1_pool_reference_ckpt "${REF_S1}" \
          --stage1_checkpoint_metric recall10 \
          --stage1_probe_vis 0 --des "${S1_DES}" || exit 21
        [ -f "${S1_CKPT}" ] || { echo "### ERROR missing ${S1_CKPT}"; exit 22; }

        echo "### STAGE2 stage1_ckpt=${S1_CKPT}"
        python -u run.py --task_name stage2_relation --is_training 1 \
          --model_id "${S2_ID}" --model RelationStage2 \
          --base_head_mode shared_target_linear "${COMMON[@]}" \
          --learning_rate 1e-2 --train_epochs "${STAGE2_EPOCHS}" --patience "${PATIENCE}" \
          --stage1_ckpt_path "${S1_CKPT}" --freeze_stage1_encoder 1 \
          --memory_cache_mode precompute --refresh_memory_every_epoch 1 \
          --memory_chunk_size 1024 --top_k 10 --tau_topk 0.10 \
          --relation_mixer_input retrieved --fusion_mode residual --gate_mode scalar \
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
done
echo "completed=${#OK[@]} skipped=${#SKIP[@]} failed=${#FAILED[@]}"
for run in "${FAILED[@]:-}"; do [ -n "${run}" ] && echo "  FAILED ${run}"; done
