#!/usr/bin/env bash
# TRACK-V Phase B (H192/H336) prerequisite: trains the missing
# `soft_set_mse S0_wce` Stage-1 REFERENCE checkpoint for
# ETTh1/Weather x H192/H336 (never trained before -- only seq96_pred96
# and seq720_pred720 existed under checkpoints/soft_set_mse/stage1/).
#
# S0_wce ONLY (not the full S0-S4 WCE-vs-SetMSE ablation that
# `run_soft_set_mse_stage1.sh` normally sweeps): `--stage1_set_mse_weight
# 0.0` means `weighted_set = set_mse_weight * set_loss` is EXACTLY zero
# (models/RelationStage1.py:5191) regardless of `--stage1_set_tau`'s
# value -- the SetMSE term contributes nothing to gradient or loss for
# this arm, verified by direct source read before launching this script
# (no blind hyperparameter guess). `--stage1_set_support_weight`
# defaults to 0.0 too (run.py:236), so no other tau-dependent term is
# live either. Therefore a SINGLE shared tau (0.015, same value already
# used for ETTh1 H96) is used for ALL FOUR new cells here -- not because
# it was separately calibrated for each, but because it is PROVABLY
# inert for S0_wce, so using one consistent value everywhere is strictly
# correct (not an approximation) and avoids a spurious per-dataset
# hyperparameter difference that would have no justification.
#
# Everything else is byte-identical to `run_soft_set_mse_stage1.sh`'s
# own S0_wce invocation template.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="logs/soft_set_mse"
CKPT_ROOT="checkpoints/soft_set_mse"
mkdir -p "$LOG_ROOT"

TAU=0.015  # shared, provably inert for S0_wce -- see header

# dataset  pred  enc_in  data_key  root                                            data_path
SPECS='ETTh1 192  7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv
ETTh1 336 7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv
Weather 192  21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv
Weather 336 21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv'

printf '%s\n' "$SPECS" | while read -r DS PRED ENC DKEY ROOT DPATH; do
  [ -n "${DS:-}" ] || continue
  GRAPH_ARGS=()
  if [ "$DS" = Weather ]; then
    GRAPH_ARGS=(--relation_graph_path metrics/relation_graphs/weather/pearson_self_top1.json)
  fi
  DIR="$LOG_ROOT/${DS}/pred${PRED}"; mkdir -p "$DIR"
  ARM=S0_wce
  LOG="$DIR/${ARM}.log"; MARKER="$DIR/${ARM}.done"
  if [ -f "$MARKER" ]; then
    echo "[skip] ${DS}/pred${PRED}/${ARM} already done"; continue
  fi
  MODEL_ID="carts_softset_${DS}_${PRED}_${ARM}"
  echo "=============================================================="
  echo "[${DS}/pred${PRED}/${ARM}] wce_weight=1.0 set_mse_weight=0.0 tau_set=${TAU} (inert)"
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
      --stage1_wce_weight 1.0 --stage1_set_mse_weight 0.0 \
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
echo "H192/H336 S0_wce reference-checkpoint training finished $(date -Is)"
