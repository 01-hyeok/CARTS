#!/usr/bin/env bash
# Step-0 tau calibration for Weather, same method as run_tau_calibration.sh
# (now backed by exp_stage1_relation.py:tau_calibration_diag, reconstructed and
# acceptance-tested against the ETTh1 ground-truth logs -- see
# tests/test_tau_calibration.py). Run on CPU by default: this reads a frozen
# checkpoint's score distribution over 4 train batches, cheap enough that it
# should never contend with a GPU job that is mid-run.
#
# Unlike the ETTh1 driver, no --stage1_retrieval_metric override is used here:
# the soft_set_mse arms for this experiment all score with plain cosine (the
# changed variable is the loss, not the scorer), so calibration reads the same
# score the training arms will actually use.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-}"
export CARTS_TAUCAL_DIAG=1 CARTS_TAUCAL_BATCHES="${BATCHES:-4}"
export CARTS_TAUCAL_TAUS="${TAUS:-0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.05,0.07,0.1}"
DIR=logs/tau_calibration_weather; mkdir -p "$DIR"
for PRED in ${PREDS:-96 720}; do
  CK=$(ls -d ./checkpoints/stage1/custom/seq${PRED}_pred${PRED}/*w1_cosine_wce_weather*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] pred${PRED}"; continue; }
  python -u run.py --task_name stage1_relation --is_training 0 \
    --model RelationStage1 --data custom \
    --root_path ../Dataset/Time-Series-Library_dataset/weather/ \
    --data_path weather.csv --features M \
    --seq_len "$PRED" --label_len 0 --pred_len "$PRED" \
    --enc_in 21 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
    --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
    --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_graph_path metrics/relation_graphs/weather/pearson_self_top1.json \
    --relation_encoder_type mlp --relation_self_fill linear \
    --top_k 10 --tau_student 0.10 --tau_teacher 0.1 --tau_topk 0.1 \
    --teacher_mse_space normalized --stage1_teacher_mode mse \
    --stage1_loss_mode weighted_topk_ce --stage1_coverage_top_k 10 \
    --stage1_full_memory_gradient_mode full_online --stage1_probe_vis 0 \
    --stage1_ckpt_path "$CK" --stage1_encoder_init checkpoint \
    --model_id "carts_w1_cosine_wce_weather_${PRED}" \
    --des "w1_cosine_wce_weather_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "\[taucal\]|Traceback|Error" | tee "$DIR/pred${PRED}.log"
done
