#!/usr/bin/env bash
# Stage-1 soft_set_mse experiment: does aggregate-aligned supervision beat WCE?
#
#   S0  WCE only                         wce_weight=1  set_mse_weight=0
#   S1  SetMSE only                      wce_weight=0  set_mse_weight=1
#   S2  WCE + Set  lambda=10             wce_weight=1  set_mse_weight=10
#   S3  WCE + Set  lambda=30             wce_weight=1  set_mse_weight=30
#   S4  WCE + Set  lambda=50             wce_weight=1  set_mse_weight=50
#
# Reuses the existing wce_soft_set_mse loss mode end to end (soft_set_mse(),
# hard_aggregate_metrics()); S1 is the same mode with --stage1_wce_weight 0,
# not a new loss. Checkpoint selection is --stage1_checkpoint_metric
# hard_aggregate_mse10 for every arm, per the experiment spec: none is selected
# on its own training objective.
#
# Only the loss changes across arms -- encoder, scorer (plain cosine, no
# --stage1_retrieval_metric), d_model, optimizer, epochs and candidate masking
# are identical for every arm on a given dataset/horizon.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS
source /data/pjh_workspace/ts-env/bin/activate

LOG_ROOT="${LOG_ROOT:-logs/soft_set_mse}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/soft_set_mse}"
mkdir -p "$LOG_ROOT"

# dataset  pred  enc_in  data_key  root                                            data_path   tau_set
SPECS='ETTh1 96  7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv    0.015
ETTh1 720 7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/  ETTh1.csv    0.02
Weather 96  21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv  0.0025
Weather 720 21 custom ../Dataset/Time-Series-Library_dataset/weather/       weather.csv  0.00105'

printf '%s\n' "${SPECS_OVERRIDE:-$SPECS}" | while read -r DS PRED ENC DKEY ROOT DPATH TAU; do
  [ -n "${DS:-}" ] || continue
  GRAPH_ARGS=()
  if [ "$DS" = Weather ]; then
    GRAPH_ARGS=(--relation_graph_path metrics/relation_graphs/weather/pearson_self_top1.json)
  fi
  DIR="$LOG_ROOT/${DS}/pred${PRED}"; mkdir -p "$DIR"

  for ARM in S0_wce S1_set_only S2_lam10 S3_lam30 S4_lam50; do
    case "$ARM" in
      S0_wce)      WCE_W=1.0; SET_W=0.0  ;;
      S1_set_only) WCE_W=0.0; SET_W=1.0  ;;
      S2_lam10)    WCE_W=1.0; SET_W=10.0 ;;
      S3_lam30)    WCE_W=1.0; SET_W=30.0 ;;
      S4_lam50)    WCE_W=1.0; SET_W=50.0 ;;
    esac
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
done
echo "soft_set_mse Stage-1 sweep finished $(date -Is)"
