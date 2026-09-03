#!/usr/bin/env bash
# EXP-1 + EXP-2: selection-rule intervention over two supports.
#
#   P100  a cosine-induced fixed common candidate support
#   FULL  every valid candidate in memory
#
# Running the same arms over both supports is what separates the bottlenecks:
# a gap that survives widening the support is a ranking or objective problem,
# a gap that only appears once the support is widened was the support all along.
#
# Nothing is trained. Config per dataset is recovered from the checkpoint's own
# training run rather than copied between datasets -- the two differ in channel
# count, memory size, data class and checkpoint generation.
set -eu
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS

ARMS="${ARMS:-R0,R1,R2-U,R2-W,R3}"
OUT="${OUT:-logs/oracle_intervention}"
mkdir -p "$OUT"

run_one () {
  local DS="$1" PRED="$2" POOL="$3" BS="$4"
  local DATA ROOT DPATH ENC CKDIR S1PAT CKPAT LABEL
  case "$DS" in
    ETTh1)
      DATA=ETTh1; ROOT=../Dataset/Time-Series-Library_dataset/ETT-small/
      DPATH=ETTh1.csv; ENC=7; CKDIR=ETTh1
      # ETTh1 has a post-wiring-fix generation; use it.
      CKPAT='*s2ls_fixsel_cosine_kl_stage2*'; S1PAT='*e2_cos_kl_ETTh1*' ;;
    Weather)
      DATA=custom; ROOT=../Dataset/Time-Series-Library_dataset/weather/
      DPATH=weather.csv; ENC=21; CKDIR=custom
      # Weather has no 'fixsel' generation. For a cosine-configured arm the
      # wiring fix is a verified no-op (ETTh1 cosine arms are identical to six
      # decimals pre- and post-fix), so this checkpoint is the KL+Cosine
      # reference. See EXPERIMENT_LOG.md EXP-2.
      CKPAT='*s2ls_cosine_kl_stage2_weather*'; S1PAT='*w1_cosine_kl_weather*' ;;
    *) echo "unknown dataset $DS"; return 1 ;;
  esac

  local CK S1
  CK=$(ls -d ./checkpoints/stage2/${CKDIR}/seq${PRED}_pred${PRED}/${CKPAT}/ 2>/dev/null | head -1)checkpoint.pth
  S1=$(ls -d ./checkpoints/stage1/${CKDIR}/seq${PRED}_pred${PRED}/${S1PAT}/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] || { echo "[miss] stage2 ${DS} H${PRED}"; return 0; }
  [ -f "$S1" ] || { echo "[miss] stage1 ${DS} H${PRED}"; return 0; }

  [ "$POOL" -le 0 ] && LABEL=FULL || LABEL="P${POOL}"
  local LOG="${OUT}/${DATA}_H${PRED}_${LABEL}.log"
  [ -f "${OUT}/${DATA}_H${PRED}_${LABEL}.done" ] && { echo "[skip] ${DS} H${PRED} ${LABEL}"; return 0; }

  echo "=== ${DS} H${PRED} ${LABEL} bs=${BS} ==="
  { echo "### RUN oracle_intervention ${DS} H${PRED} ${LABEL} $(date -Is)"
    echo "### stage2_ckpt ${CK}"
    echo "### stage1_ckpt ${S1}"
    echo "### git $(git rev-parse HEAD)"
    python -u run.py \
      --task_name stage2_relation --is_training 0 \
      --model RelationStage2 --data "$DATA" \
      --root_path "$ROOT" --data_path "$DPATH" --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" --enc_in "$ENC" \
      --batch_size "$BS" --num_workers 0 \
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
      --model_id "carts_oracle_int_${DS}_${PRED}" \
      --des "oracle_int_${DS}_sl${PRED}_pl${PRED}"
    echo "### RUN COMPLETE $(date -Is)"
  } 2>&1 | tee "$LOG"
  grep -q '### RUN COMPLETE' "$LOG" && touch "${OUT}/${DATA}_H${PRED}_${LABEL}.done"
}

# One spec per line: dataset  horizon  pool(0=full)  batch
# Newline-delimited rather than a quoted word list, so an externally supplied
# SPECS is not split into individual words the way a shell array default is.
DEFAULT_SPECS='ETTh1 96 100 32
ETTh1 720 100 32
Weather 96 100 32
Weather 720 100 32
ETTh1 96 0 16
ETTh1 720 0 8
Weather 96 0 8
Weather 720 0 4'

printf '%s\n' "${SPECS:-$DEFAULT_SPECS}" | while read -r DS PRED POOL BS; do
  [ -n "${DS:-}" ] || continue
  run_one "$DS" "$PRED" "$POOL" "$BS"
done
echo "oracle intervention suite finished $(date -Is)"
