#!/usr/bin/env bash
# Post-hoc: does tau_topk explain the flat fusion weights?
#
# Every one of the 48 Stage-2 runs reports alpha_entropy_norm in [0.994, 0.999],
# i.e. the ten retrieved futures are averaged almost uniformly. tau_topk=0.1
# divides cosine scores whose spread inside the Top-10 is ~0.04, so the softmax
# cannot separate them. This re-runs test only, on the already-trained
# checkpoint, varying tau. Training saw tau=0.1, so this measures how much of
# the loss is inference-time flattening rather than a bad ranking.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PRED="${PRED:-96}"; ARM="${ARM:-cosine_kl}"
SCORE="${ARM%%_*}"; LOSS="${ARM##*_}"
CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*e2_cos_${LOSS}_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
OUT="${OUT:-logs/tau_sweep}"; mkdir -p "$OUT"
for TAU in 0.001 0.005 0.01 0.02 0.05 0.1; do
  echo "===== tau=${TAU} pred=${PRED} ${ARM} ====="
  python -u run.py --task_name stage2_relation --is_training 0 \
    --model_id "carts_s2ls_${ARM}_stage2_ETTh1_${PRED}" --model RelationStage2 \
    --data ETTh1 --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
    --data_path ETTh1.csv --features M \
    --seq_len "${PRED}" --label_len 0 --pred_len "${PRED}" \
    --enc_in 7 --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
    --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
    --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all \
    --relation_encoder_type mlp --relation_self_fill linear \
    --top_k 10 --tau_topk "${TAU}" \
    --fusion_mode residual --gate_mode scalar \
    --stage1_ckpt_path "${CK}" --stage1_encoder_init checkpoint \
    --freeze_stage1_encoder 1 --stage2_e2e 0 \
    --des "s2ls_${ARM}_stage2_ETTh1_sl${PRED}_pl${PRED}" 2>&1 \
    | grep -E "Stage2 Test \||final_mse|final_mae|Error|Traceback" | tail -3
done
