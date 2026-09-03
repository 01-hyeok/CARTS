#!/usr/bin/env bash
# Presentation queue: one GPU, strictly sequential, resumable.
#
#   P1  ETTh1 FULL oracle          -> TABLE 2   (own script; waited on, not restarted)
#   P2  (skipped) Weather pair2 excluded -- repeated full-GPU OOM
#   P3  Weather Stage-2 4 arms     -> TABLE 1 (cosine/asymmetric x kl/wce)
#   P4  Weather FULL oracle        -> extra, requested after the two tables
#
# Every stage writes a .done marker and is skipped if it already exists, so a
# rerun after a failure resumes instead of repeating finished work.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd /data/pjh_workspace/CARTS

# Single-instance lock. The user was explicit: one GPU, strictly sequential,
# never two of these running at once. A killed parent can leave an orphaned
# child still training (a prior run of this exact failure mode duplicated a
# Weather Stage-1 job and OOM'd both copies), so this is enforced with flock
# rather than trusted to careful process management.
exec 9>/tmp/carts_presentation_queue.lock
flock -n 9 || { echo "[queue] another instance holds the lock; exiting"; exit 1; }
source /data/pjh_workspace/ts-env/bin/activate

HZ="${HZ:-96 720 192 336}"          # presentation horizon order
ROOT=logs/presentation_202609
mkdir -p "$ROOT"

# --- P1: ETTh1 FULL oracle. The script skips horizons that already carry a
# .done marker, so calling it is safe whether or not a previous pass finished.
# It is NOT guarded by a pgrep wait: `pgrep -f` matches any shell whose command
# line merely mentions the script name, including a monitoring loop watching for
# it, and two such loops will wait on each other forever.
echo "[queue] P1 ETTh1 FULL oracle $(date -Is)"
bash scripts/run_presentation_oracle.sh || echo "[queue] P1 reported a failure; continuing"

# --- P2: skipped. pair2 (MLP) on Weather repeatedly saturates the full 79GB
# of GPU 1 even alone (36,696 candidates x 21 channels is far past what ETTh1's
# 8,449 x 7 needed) and every attempt OOM'd. Excluded from TABLE 1 by decision,
# not left as a silent gap -- see PRESENTATION_RESULTS_202609.md.
echo "[queue] P2 skipped: Weather pair2 excluded by decision (repeated OOM) $(date -Is)"

# --- P3: Weather Stage-2, all six arms, identical conditions ----------------
echo "[queue] P3 Weather Stage-2 6-arm $(date -Is)"
# DATASET must match run_stage2_learned_score.sh's own case statement exactly
# ("weather", lowercase) -- the wrong case here is what silently no-op'd this
# stage entirely last time, and it went unnoticed because piping to `tail`
# reports tail's exit code, not the script's. `set -o pipefail` above plus
# checking PIPESTATUS here is what makes that failure visible instead.
DATASET=weather PREDS="$HZ" MODES="stage2" \
  ARMS="cosine:kl cosine:wce asymmetric:kl asymmetric:wce" \
  ORACLE_EVAL=1 LOG_ROOT="$ROOT/weather_forecasting" \
  bash scripts/run_stage2_learned_score.sh 2>&1 | tee -a logs/presentation_queue_p3.out
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "[queue] P3 FAILED, exit=${PIPESTATUS[0]}"

# --- P4: Weather FULL oracle ------------------------------------------------
echo "[queue] P4 Weather FULL oracle $(date -Is)"
OUTW="$ROOT/oracle_full/Weather"; mkdir -p "$OUTW"
for PRED in $HZ; do
  D="$OUTW/H${PRED}"; mkdir -p "$D"
  [ -f "$D/.done" ] && { echo "[skip] Weather H${PRED} oracle"; continue; }
  CK=$(ls -d ./checkpoints/stage2/custom/seq${PRED}_pred${PRED}/*s2ls_cosine_kl_stage2_weather*/ 2>/dev/null | head -1)checkpoint.pth
  S1=$(ls -d ./checkpoints/stage1/custom/seq${PRED}_pred${PRED}/*w1_cosine_kl_weather*/ 2>/dev/null | head -1)checkpoint.pth
  [ -f "$CK" ] && [ -f "$S1" ] || { echo "[miss] Weather H${PRED}"; continue; }
  # 36k candidates x 21 channels: the per-query bank is the memory ceiling here.
  case "$PRED" in 96) BS=6 ;; 192) BS=4 ;; *) BS=2 ;; esac
  echo "=== Weather H${PRED} FULL bs=${BS} ==="
  { echo "### git $(git rev-parse HEAD)"; echo "### stage2_ckpt ${CK}"
    python -u run.py --task_name stage2_relation --is_training 0 \
      --model RelationStage2 --data custom \
      --root_path ../Dataset/Time-Series-Library_dataset/weather/ \
      --data_path weather.csv --features M \
      --seq_len "$PRED" --label_len 0 --pred_len "$PRED" --enc_in 21 \
      --batch_size "$BS" --num_workers 0 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
      --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
      --relation_input_space delta_last --relation_teacher_space delta_last \
      --source_mode auto --relation_top_n 1 --target_mode all \
      --relation_encoder_type mlp --relation_self_fill linear \
      --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
      --stage1_ckpt_path "$S1" --stage1_encoder_init checkpoint \
      --freeze_stage1_encoder 1 --stage2_e2e 0 --stage2_ckpt_path "$CK" \
      --oracle_intervention_arms R0,R1,R2-U,R2-W,R3 --oracle_intervention_pool 0 \
      --oracle_intervention_out "$D" \
      --des "oracle_full_weather_sl${PRED}_pl${PRED}"
    echo "### RUN COMPLETE $(date -Is)"
  } 2>&1 | tee "$D/stdout.log"
  grep -q '### RUN COMPLETE' "$D/stdout.log" && touch "$D/.done"
done

echo "[queue] ALL STAGES FINISHED $(date -Is)"
