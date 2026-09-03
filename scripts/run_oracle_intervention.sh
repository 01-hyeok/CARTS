#!/usr/bin/env bash
# EXPERIMENT 1 -- Test-matched Oracle Intervention.
#
# No training. One Stage-2 checkpoint is loaded once and every arm runs under
# it, so the only thing that differs between arms is which ten candidates
# retrieval is handed. The support is the cosine Top-P100 -- a fixed common
# candidate support, not a neutral pool and not a full-memory oracle.
#
# Config is not chosen here: it is recovered from the checkpoint's own training
# run, because a relation_top_n that disagrees with the checkpoint would change
# the branch structure and silently stop being the same model.
set -eu
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS

# R2-relation is deliberately absent: under this checkpoint's self-only relation
# graph it is the same experiment as the target-space set oracle, so it would
# cost a full pass to reproduce R2-U exactly. It stays a unit-test invariant.
ARMS="${ARMS:-R0,R1,R2-U,R2-W,R3}"
POOL="${POOL:-100}"
OUT="${OUT:-logs/oracle_intervention}"
mkdir -p "$OUT"

for PRED in ${PREDS:-96 720}; do
  # The corrected-wiring KL+Cosine Stage-2 run, fixed in advance as the
  # reference so the checkpoint is not chosen by looking at test scores.
  CK=$(ls -d ./checkpoints/stage2/ETTh1/seq${PRED}_pred${PRED}/*s2ls_fixsel_cosine_kl_stage2*/ 2>/dev/null | head -1)checkpoint.pth
  S1=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_kl_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] stage2 ckpt pred${PRED}"; continue; }
  [ -f "$S1" ] || { echo "[miss] stage1 ckpt pred${PRED}"; continue; }

  LOG="${OUT}/ETTh1_H${PRED}_P${POOL}.log"
  echo "=== ETTh1 H${PRED}  stage2=$(basename "$(dirname "$CK")") ==="
  { echo "### RUN oracle_intervention ETTh1 H${PRED} $(date -Is)"
    echo "### stage2_ckpt ${CK}"
    echo "### stage1_ckpt ${S1}"
    echo "### git $(git rev-parse HEAD)"
    python -u run.py \
      --task_name stage2_relation --is_training 0 \
      --model RelationStage2 --data ETTh1 \
      --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
      --data_path ETTh1.csv --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" --enc_in 7 \
      --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
      --stage1_ckpt_path "$S1" --stage1_encoder_init checkpoint \
      --freeze_stage1_encoder 1 --stage2_e2e 0 \
      --stage2_ckpt_path "$CK" \
      --oracle_intervention_arms "$ARMS" \
      --oracle_intervention_pool "$POOL" \
      --oracle_intervention_out "$OUT" \
      --model_id "carts_oracle_int_ETTh1_${PRED}" \
      --des "oracle_int_ETTh1_sl${PRED}_pl${PRED}"
    echo "### RUN COMPLETE $(date -Is)"
  } 2>&1 | tee "$LOG"
done
echo "oracle intervention finished $(date -Is)"
