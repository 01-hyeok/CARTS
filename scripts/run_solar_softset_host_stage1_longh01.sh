#!/usr/bin/env bash
# TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- Solar canonical Stage-1 S0_wce
# reference for H192/H336/H720, built via the EXACT SAME pipeline as
# scripts/run_solar_softset_host_stage1.sh (every flag copied verbatim
# from its S0_wce branch), with ONE addition:
# --relation_target_chunk_size 20, the EXISTING, already-implemented
# memory-safety knob (run.py's --relation_target_chunk_size, default 0
# = all 137 channels forward+backward together before one optimizer
# step) that restricts each batch's forward+backward to a 20-channel
# subset, cycling across batches. This is "channel-wise backward"
# (OOM mitigation order step 1), NOT a change to loss definition,
# architecture, or any other hyperparameter -- confirmed via a smoke
# test that Solar_720 (which OOM'd at the full-137-channel default,
# see logs/soft_set_mse/Solar/pred720/S0_wce.log) completes a full
# epoch at chunk_size=20 using <31GiB peak VRAM (vs the prior attempt's
# OOM at 69GiB+13GiB requested on an 80GiB card).
#
# H192/H336 are included even though they are smaller than H720 (so
# may not need chunking) -- using the SAME chunk_size for all three
# keeps the "effective batch semantics" as close to uniform across
# horizons as this mitigation allows, rather than tuning a different
# value per horizon with no evidence either direction is needed.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="${LOG_ROOT:-logs/soft_set_mse}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/soft_set_mse}"
mkdir -p "$LOG_ROOT"

CHUNK="${RELATION_TARGET_CHUNK_SIZE:-20}"

SPECS='Solar 192 137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt  0.01
Solar 336 137 Solar ../Dataset/Time-Series-Library_dataset/Solar/  solar_AL.txt  0.01
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
  echo "[${DS}/pred${PRED}/${ARM}] wce_weight=${WCE_W} set_mse_weight=${SET_W} tau_set=${TAU} chunk=${CHUNK}"
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
      --relation_target_chunk_size "$CHUNK" \
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
echo "Solar softset Stage-1 long-horizon (H192/H336/H720, chunk=${CHUNK}) finished $(date -Is)"
