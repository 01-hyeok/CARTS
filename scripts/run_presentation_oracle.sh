#!/usr/bin/env bash
# TABLE 2: Individual Oracle vs Set Oracle over FULL memory, TEST split, K=10.
# Sequential on one GPU. Existing outputs are skipped, never overwritten.
set -eu
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
OUT=logs/presentation_202609/oracle_full/ETTh1
mkdir -p "$OUT"

# horizon batch   (batch shrinks with horizon: the full bank is materialised per query)
SPECS='96 16
720 8
192 12
336 8'

printf '%s\n' "${SPECS_OVERRIDE:-$SPECS}" | while read -r PRED BS; do
  [ -n "${PRED:-}" ] || continue
  D="$OUT/H${PRED}"; mkdir -p "$D"
  [ -f "$D/.done" ] && { echo "[skip] ETTh1 H${PRED} (already complete)"; continue; }
  CK=$(ls -d ./checkpoints/stage2/ETTh1/seq${PRED}_pred${PRED}/*s2ls_fixsel_cosine_kl_stage2*/ 2>/dev/null | head -1)checkpoint.pth
  S1=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_kl_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] && [ -f "$S1" ] || { echo "[miss] ETTh1 H${PRED}"; continue; }

  cat > "$D/command.txt" <<CMD
source /data/pjh_workspace/ts-env/bin/activate
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u run.py --task_name stage2_relation --is_training 0 \\
  --model RelationStage2 --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \\
  --data_path ETTh1.csv --features M --seq_len ${PRED} --label_len 0 --pred_len ${PRED} --enc_in 7 \\
  --batch_size ${BS} --num_workers 0 --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \\
  --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \\
  --relation_input_space delta_last --relation_teacher_space delta_last \\
  --source_mode auto --relation_top_n 1 --target_mode all \\
  --relation_encoder_type mlp --relation_self_fill linear \\
  --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \\
  --stage1_ckpt_path ${S1} --stage1_encoder_init checkpoint \\
  --freeze_stage1_encoder 1 --stage2_e2e 0 --stage2_ckpt_path ${CK} \\
  --oracle_intervention_arms R0,R1,R2-U,R2-W,R3 --oracle_intervention_pool 0 \\
  --oracle_intervention_out ${D} --des oracle_full_ETTh1_sl${PRED}_pl${PRED}
CMD
  echo "=== ETTh1 H${PRED} FULL bs=${BS} ==="
  { echo "### git $(git rev-parse HEAD)"; echo "### stage2_ckpt ${CK}"; echo "### stage1_ckpt ${S1}"
    bash "$D/command.txt"; echo "### RUN COMPLETE $(date -Is)"
  } 2>&1 | tee "$D/stdout.log"
  grep -q '### RUN COMPLETE' "$D/stdout.log" && touch "$D/.done"
done
echo "presentation oracle finished $(date -Is)"
