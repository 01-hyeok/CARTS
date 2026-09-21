#!/usr/bin/env bash
# TRACK-A-TF-ORACLE-LEARNABILITY01 -- corrected Stage-2 retraining.
#   scripts/run_tf_oracle_learnability01_stage2.sh <cell> [<cell> ...]
#   cell in: ETTh1_96 ETTh1_720 Weather_96 Weather_720 Solar_96
#
# Per cell: build the corrected delta-space cache for each completed Stage-1
# arm (scripts/build_tf_oracle_learnability01_retrieval_cache.py), then
# retrain Stage-2 from a FRESH init, frozen retrieval branch
# (scripts/train_setlossctrl_stage2_retrain02.py, reused UNMODIFIED -- it is
# already fully generic over cell/arm/cache_dir, not CONTROL02-specific).
# Both arms of a cell share the same fresh Stage-2 init.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

STAGE1_OUT="results/TRACK-A-TF-ORACLE-LEARNABILITY01"
OUT="results/EXP-TF-ORACLE-LEARNABILITY01-STAGE2"
LOGS="logs/EXP-TF-ORACLE-LEARNABILITY01-STAGE2"
mkdir -p "$LOGS"

S1_ETTh1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S1_ETTh1_720="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S1_Weather_96="checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S1_Weather_720="checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S1_Solar_96="checkpoints/soft_set_mse/stage1/Solar/seq96_pred96/stage1_carts_softset_Solar_96_S0_wce_RelationStage1_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"

S2_ETTh1_96="checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth"
S2_ETTh1_720="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_Weather_96="checkpoints/stage2/custom/seq96_pred96/stage2_carts_softset_s2_Weather_96_S0_wce_RelationStage2_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl96_pl96_0/checkpoint.pth"
S2_Weather_720="checkpoints/stage2/custom/seq720_pred720/stage2_carts_softset_s2_Weather_720_S0_wce_RelationStage2_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Weather_sl720_pl720_0/checkpoint.pth"
S2_Solar_96="checkpoints/stage2/Solar/seq96_pred96/stage2_carts_softset_s2_Solar_96_S0_wce_RelationStage2_Solar_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_Solar_sl96_pl96_0/checkpoint.pth"

ARMS=(individual_tf_cosine set_tf_cosine)

run_cell() {
  local cell="$1" pred_len s1 s2
  case "$cell" in
    ETTh1_96)    pred_len=96;  s1="$S1_ETTh1_96";    s2="$S2_ETTh1_96" ;;
    ETTh1_720)   pred_len=720; s1="$S1_ETTh1_720";   s2="$S2_ETTh1_720" ;;
    Weather_96)  pred_len=96;  s1="$S1_Weather_96";  s2="$S2_Weather_96" ;;
    Weather_720) pred_len=720; s1="$S1_Weather_720"; s2="$S2_Weather_720" ;;
    Solar_96)    pred_len=96;  s1="$S1_Solar_96";    s2="$S2_Solar_96" ;;
    *) echo "unknown cell: $cell" >&2; return 2 ;;
  esac

  local init="$LOGS/shared_stage2_init_${cell}.pth"
  mkdir -p "$OUT/$cell"
  local first=1
  for arm in "${ARMS[@]}"; do
    local stage1_marker="$STAGE1_OUT/$cell/DONE_${arm}.marker"
    if [[ ! -f "$stage1_marker" ]]; then
      echo "---- [$cell] $arm Stage-1 not complete yet -- skipping ----"
      continue
    fi
    local marker="$OUT/$cell/DONE_${arm}.marker"
    local cache_dir="$OUT/cache/$cell/${arm}"
    local rm_json="$STAGE1_OUT/$cell/retrieval_metrics_${arm}.json"
    if [[ -f "$marker" ]]; then
      echo "---- [$cell] $arm Stage-2 already done, skipping ----"
      first=0
      continue
    fi

    if [[ ! -f "$cache_dir/train.pt" ]]; then
      echo "---- [$cell] building retrieval cache for $arm ----"
      python -u scripts/build_tf_oracle_learnability01_retrieval_cache.py \
        --cell "$cell" --arm_name "$arm" --reference_ckpt "$s1" --stage2_host "$s2" \
        --pred_len "$pred_len" --top_k 10 --chunk_size 4096 --out_dir "$OUT" \
        2>&1 | tee "$LOGS/cache_${cell}_${arm}.log"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        echo "[ABORT] $cell/$arm cache build failed" >&2; return 1
      fi
    fi

    local init_flag="--shared_init_in $init"
    [[ $first -eq 1 ]] && init_flag="--shared_init_out $init"
    first=0

    echo "---- [$cell] Stage-2 retrain: $arm ----"
    python -u scripts/train_setlossctrl_stage2_retrain02.py \
      --cell "$cell" --arm_name "$arm" --stage2_host "$s2" --cache_dir "$cache_dir" \
      --stage1_retrieval_metrics_json "$rm_json" --fragg_tolerance 0.02 \
      --checkpoints checkpoints/exp_tf_oracle_learnability01_stage2 --out_dir "$OUT" \
      --seed 0 --train_epochs 10 --patience 5 $init_flag \
      2>&1 | tee "$LOGS/stage2_${cell}_${arm}.log"

    if [[ ! -f "$marker" ]]; then
      echo "[ABORT] $cell/$arm did not produce its DONE marker -- stopping cell." >&2
      return 1
    fi
  done
  echo "########## [$cell] Stage-2 complete ##########"
}

for cell in "$@"; do
  run_cell "$cell"
done
echo "TRACK-A-TF-ORACLE-LEARNABILITY01 Stage-2 orchestrator finished $(date -Is)"
