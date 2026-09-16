#!/usr/bin/env bash
# TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED -- ETTh1_720, 8 arms, one at a
# time. CORRECTED (delta-space) cache/trainer -- see
# scripts/build_choicece_retrieval_cache01.py and
# scripts/train_choicece_stage2_retrain01.py module docstrings.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

S1_96="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
S2_96="checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"
FACTORIAL_METRICS="results/track_a_factorial_e2e/ETTh1_720"
OUT="results/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED"
INIT="logs/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/shared_stage2_init_ETTh1_720.pth"

mkdir -p logs/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED "$OUT/ETTh1_720"

ARMS=(individual_tf_cosine individual_onpolicy_cosine set_tf_cosine set_onpolicy_cosine
     individual_tf_asymmetric individual_onpolicy_asymmetric set_tf_asymmetric set_onpolicy_asymmetric)

first=1
for arm in "${ARMS[@]}"; do
  marker="$OUT/ETTh1_720/DONE_${arm}.marker"
  cache_dir="$OUT/cache/ETTh1_720/${arm}"
  rm_json="$FACTORIAL_METRICS/retrieval_metrics_${arm}.json"
  if [[ -f "$marker" ]]; then
    echo "---- [ETTh1_720] $arm already done (corrected), skipping ----"
    continue
  fi

  if [[ ! -f "$cache_dir/train.pt" ]]; then
    echo "---- [ETTh1_720] building CORRECTED (delta-space) retrieval cache for $arm ----"
    python -u scripts/build_choicece_retrieval_cache01.py \
      --cell ETTh1_720 --arm_name "$arm" --reference_ckpt "$S1_96" --stage2_host "$S2_96" \
      --pred_len 720 --top_k 10 --chunk_size 4096 --out_dir "$OUT" \
      2>&1 | tee "logs/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/cache_ETTh1_720_${arm}.log"
  fi

  init_flag="--shared_init_in $INIT"
  [[ $first -eq 1 ]] && init_flag="--shared_init_out $INIT"
  first=0

  echo "---- [ETTh1_720] Stage-2 retrain (CORRECTED): $arm ----"
  python -u scripts/train_choicece_stage2_retrain01.py \
    --cell ETTh1_720 --arm_name "$arm" --stage2_host "$S2_96" --cache_dir "$cache_dir" \
    --stage1_retrieval_metrics_json "$rm_json" --fragg_tolerance 0.02 \
    --checkpoints checkpoints/track_a_choicece_stage2_retrain01_corrected --out_dir "$OUT" \
    --seed 1 --train_epochs 10 --patience 5 $init_flag \
    2>&1 | tee "logs/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/stage2_ETTh1_720_${arm}.log"

  if [[ ! -f "$marker" ]]; then
    echo "[ABORT] $arm did not produce its DONE marker -- stopping chain." >&2
    exit 1
  fi
done
echo "########## TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED ETTh1_720: all 8 arms complete ##########"
