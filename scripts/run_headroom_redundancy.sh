#!/usr/bin/env bash
# Step B (Oracle headroom) and Step C (redundancy replacement) in one test-only pass.
#
# B asks whether a long horizon fails because memory stopped holding useful
# candidates or because the retriever stopped finding them: the Oracle Top-10,
# aggregated the way Stage-2 aggregates, bounds the second.
#
# C asks whether the retriever is collecting near-duplicates. An improving swap
# is guided by the true future, so one nearly always exists -- the number only
# reads against its controls: a random swap of the same shape, and the same
# search on arms whose candidate spread already differs.
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export CARTS_SUBSET_DIAG=1 CARTS_REDUNDANCY_DIAG=1
export CARTS_SUBSET_DIAG_BATCHES="${BATCHES:-8}"
export CARTS_SET_TAUS="${TAUS:-0.015}"
export CARTS_REDUNDANCY_DEPTH="${DEPTH:-100}"
OUT="${OUT:-logs/headroom_redundancy}"; mkdir -p "$OUT"

score_args(){ case "$1" in
  cosine) echo "" ;;
  asymmetric) echo "--stage1_retrieval_metric asymmetric --stage1_metric_output cosine --stage1_metric_layer_norm 0" ;;
  pair2) echo "--stage1_retrieval_score pairwise_mlp --stage1_pairwise_feature pair2" ;;
esac; }

run_one(){
  local ds="$1" pred="$2" score="$3" loss="$4" ck="$5"
  local arm="${score}_${loss}"
  local log="${OUT}/${ds}_pred${pred}_${arm}.log"
  [ -s "$log" ] && grep -q '\[swap\] delta_best' "$log" && { echo "[skip] ${ds}/${pred}/${arm}"; return; }
  case "$ds" in
    ETTh1)   local loader=ETTh1; local root=../Dataset/Time-Series-Library_dataset/ETT-small/; local csv=ETTh1.csv; local ch=7; local graph="" ;;
    weather) local loader=custom; local root=../Dataset/Time-Series-Library_dataset/weather/; local csv=weather.csv; local ch=21; local graph="--relation_graph_path metrics/relation_graphs/weather/pearson_self_top1.json" ;;
  esac
  echo "=== ${ds} pred${pred} ${arm} ==="
  python -u run.py --task_name stage2_relation --is_training 0 \
    --model_id "carts_s2ls_${arm}_stage2_${ds}_${pred}" --model RelationStage2 \
    --data "$loader" --root_path "$root" --data_path "$csv" --features M \
    --seq_len "$pred" --label_len 0 --pred_len "$pred" \
    --enc_in "$ch" --batch_size 32 --num_workers 0 \
    --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 \
    --patch_len 16 --stride 16 --seed 0 --candidate_mask raft \
    --relation_input_space delta_last --relation_teacher_space delta_last \
    --source_mode auto --relation_top_n 1 --target_mode all $graph \
    --relation_encoder_type mlp --relation_self_fill linear \
    --top_k 10 --tau_topk 0.1 --fusion_mode residual --gate_mode scalar \
    --stage1_ckpt_path "$ck" --stage1_encoder_init checkpoint \
    --freeze_stage1_encoder 1 --stage2_e2e 0 --oracle_candidate_eval 1 \
    $(score_args "$score") \
    --des "s2ls_${arm}_stage2_${ds}_sl${pred}_pl${pred}" 2>&1 \
    | grep -E "\[step0\]|\[swap\]|Traceback|Error" > "$log"
  grep -q '\[swap\] delta_best' "$log" && echo "[ok] ${ds}/${pred}/${arm}" || echo "[FAIL] ${ds}/${pred}/${arm}"
}

for PRED in ${PREDS:-96 720 192 336}; do
  for A in "cosine kl e2_cos_kl" "cosine wce e2_cos_weighted_topk_ce" \
           "asymmetric kl e2_asym_kl" "asymmetric wce e2_asym_weighted_topk_ce" \
           "pair2 kl e2_pair2_kl" "pair2 wce e2_pair2_weighted_topk_ce"; do
    set -- $A
    CK=$(ls -d ./checkpoints/stage1/ETTh1/seq${PRED}_pred${PRED}/*${3}_ETTh1*/ 2>/dev/null | head -1)checkpoint.pth
    [ -f "$CK" ] && run_one ETTh1 "$PRED" "$1" "$2" "$CK"
  done
  for A in "cosine kl" "cosine wce" "asymmetric kl" "asymmetric wce"; do
    set -- $A
    CK=$(ls -d ./checkpoints/stage1/custom/seq${PRED}_pred${PRED}/*w1_${1}_${2}_weather*/ 2>/dev/null | head -1)checkpoint.pth
    [ -f "$CK" ] && run_one weather "$PRED" "$1" "$2" "$CK"
  done
done
echo "headroom + redundancy finished $(date -Is)"
