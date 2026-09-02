#!/usr/bin/env bash
# Weather in the original CARTS relation setting: relation_top_n 3, so each
# target retrieves through its own channel plus its two most correlated ones.
#
# The score/loss sweep on weather ran self-only (top1), which was chosen to keep
# the comparison with ETTh1 clean. That makes its numbers incomparable to the
# original weather baseline, which is cross-channel. This re-runs the baseline
# arm (cosine + kl) under the original graph so the sweep has a reference point
# in the setting the project actually uses.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
ROOT=../Dataset/Time-Series-Library_dataset/weather/
GRAPH=metrics/relation_graphs/weather/pearson_self_top3.json
S1DIR=logs/weather_top3/stage1; S2DIR=logs/weather_top3/stage2
mkdir -p "$S1DIR" "$S2DIR"

for PRED in 96 192 336 720; do
  TAG="w3_cosine_kl_weather_${PRED}"
  LOG="${S1DIR}/pred${PRED}.log"
  if [ ! -f "${S1DIR}/pred${PRED}.done" ]; then
  { echo "### RUN stage1 top3 pred${PRED} $(date -Is)"
    python -u run.py --task_name stage1_relation --is_training 1 \
      --model_id "carts_${TAG}" --model RelationStage1 \
      --data custom --root_path "$ROOT" --data_path weather.csv --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
      --enc_in 21 --batch_size 32 --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 3 --target_mode all \
      --relation_graph_path "$GRAPH" \
      --relation_encoder_type mlp --relation_self_fill linear \
      --learning_rate 1e-3 --train_epochs 10 --patience 5 \
      --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
      --teacher_mse_space normalized --stage1_teacher_mode mse \
      --stage1_loss_mode kl --stage1_coverage_top_k 10 \
      --stage1_full_memory_gradient_mode full_online \
      --stage1_checkpoint_metric retrieved_mse10 --stage1_probe_vis 0 \
      --des "${TAG}_sl${PRED}_pl${PRED}" || exit 21
    echo "### RUN COMPLETE stage1 top3 pred${PRED} $(date -Is)"; } 2>&1 | tee "$LOG"
  grep -q '### RUN COMPLETE' "$LOG" && touch "${S1DIR}/pred${PRED}.done"
  fi

  CK=$(ls -d ./checkpoints/stage1/custom/seq${PRED}_pred${PRED}/*${TAG}*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[skip] no stage1 ckpt for pred${PRED}"; continue; }
  for MODE in stage2 e2e; do
    [ "$MODE" = e2e ] && E=1 || E=0
    L2="${S2DIR}/pred${PRED}_${MODE}.log"
    [ -f "${S2DIR}/pred${PRED}_${MODE}.done" ] && continue
    { echo "### RUN ${MODE} top3 pred${PRED} $(date -Is)"; echo "### ckpt $CK"
      python -u run.py --task_name stage2_relation --is_training 1 \
        --model_id "carts_w3_s2_${MODE}_weather_${PRED}" --model RelationStage2 \
        --data custom --root_path "$ROOT" --data_path weather.csv --features M \
        --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
        --enc_in 21 --batch_size 32 --num_workers 0 \
        --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
        --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
        --relation_input_space delta_last --relation_teacher_space delta_last \
        --source_mode auto --relation_top_n 3 --target_mode all \
        --relation_graph_path "$GRAPH" \
        --relation_encoder_type mlp --relation_self_fill linear \
        --learning_rate 1e-3 --train_epochs 10 --patience 3 \
        --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
        --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
        --freeze_stage1_encoder "$([ "$MODE" = stage2 ] && echo 1 || echo 0)" \
        --stage2_e2e "$E" --refresh_memory_every_epoch "$E" --stage2_e2e_full_online "$E" \
        --des "w3_s2_${MODE}_weather_sl${PRED}_pl${PRED}" || exit 21
      echo "### RUN COMPLETE ${MODE} top3 pred${PRED} $(date -Is)"; } 2>&1 | tee "$L2"
    grep -q '### RUN COMPLETE' "$L2" && touch "${S2DIR}/pred${PRED}_${MODE}.done"
  done
done
echo "weather top3 baseline finished $(date -Is)"
