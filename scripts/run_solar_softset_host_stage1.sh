#!/usr/bin/env bash
# Solar canonical Stage-1 S0_wce reference, built via the EXACT SAME pipeline
# as scripts/run_soft_set_mse_stage1.sh (architecture/config/training
# protocol), restricted to the single S0_wce arm (wce_weight=1,
# set_mse_weight=0) -- the only arm TRACK-A-TF-ORACLE-LEARNABILITY01 needs,
# per user instruction that Solar does not need the WCE-weight loss sweep,
# only a canonical-protocol Stage-1 reference to build the frozen Stage-2
# host from. See scripts/run_soft_set_mse_stage1.sh for the S1-S4 arms this
# intentionally skips.
#
# This is a separate script (not a modification of run_soft_set_mse_stage1.sh)
# so ETTh1/Weather's existing canonical pipeline is untouched. Every flag
# below is copied verbatim from run_soft_set_mse_stage1.sh's S0_wce branch;
# only the dataset spec (Solar, enc_in=137, relation graph) and the
# arm-restriction are Solar-specific.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="${LOG_ROOT:-logs/soft_set_mse}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/soft_set_mse}"
mkdir -p "$LOG_ROOT"

# dataset  pred  enc_in  data_key  root                                       data_path    tau_set(inert for S0_wce)
SPECS='Solar 96  137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt  0.01
Solar 720 137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt  0.01'

printf '%s\n' "${SPECS_OVERRIDE:-$SPECS}" | while read -r DS PRED ENC DKEY ROOT DPATH TAU; do
  [ -n "${DS:-}" ] || continue
  GRAPH_ARGS=(--relation_graph_path metrics/relation_graphs/solar/pearson_self_top1.json)
  DIR="$LOG_ROOT/${DS}/pred${PRED}"; mkdir -p "$DIR"

  ARM=S0_wce
  WCE_W=1.0; SET_W=0.0
  LOG="$DIR/${ARM}.log"; MARKER="$DIR/${ARM}.done"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$MARKER" ]; then
    echo "[skip] ${DS}/pred${PRED}/${ARM}"; continue
  fi
  MODEL_ID="carts_softset_${DS}_${PRED}_${ARM}"
  echo "=============================================================="
  echo "[${DS}/pred${PRED}/${ARM}] wce_weight=${WCE_W} set_mse_weight=${SET_W} tau_set=${TAU}"
  echo "=============================================================="
  {
    echo "### RUN ${DS}/pred${PRED}/${ARM} $(date -Is)"
    echo "### git $(git rev-parse HEAD)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "$MODEL_ID" --model RelationStage1 \
      --data "$DKEY" --root_path "$ROOT" --data_path "$DPATH" --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" --enc_in "$ENC" \
      --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      "${GRAPH_ARGS[@]}" \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 5 \
      --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_loss_mode wce_soft_set_mse --stage1_coverage_top_k 10 \
      --stage1_wce_weight "$WCE_W" --stage1_set_mse_weight "$SET_W" \
      --stage1_set_tau "$TAU" --stage1_set_mse_normalization mean \
      --stage1_full_memory_gradient_mode full_online \
      --stage1_checkpoint_metric hard_aggregate_mse10 \
      --stage1_probe_vis 0 \
      --checkpoints "$CKPT_ROOT" \
      --des "softset_${ARM}_${DS}_sl${PRED}_pl${PRED}" || exit 21
    echo "### RUN COMPLETE ${DS}/pred${PRED}/${ARM} $(date -Is)"
  } 2>&1 | tee "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ] && grep -q '### RUN COMPLETE' "$LOG"; then
    touch "$MARKER"; echo "[ok] ${DS}/pred${PRED}/${ARM}"
  else
    echo "[FAIL] ${DS}/pred${PRED}/${ARM}"
  fi
done
echo "Solar softset Stage-1 (S0_wce only) finished $(date -Is)"
