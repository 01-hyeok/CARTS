#!/usr/bin/env bash
# TRACK-HARD-EXPERT-V5-FULL01, one cell at a time:
#   bash scripts/run_hard_expert_v5_full01.sh <cell> <ref_ckpt> <pred_len> <seq_len> <base_ckpt>
# Stage1 (10 fixed epochs, no early stopping, checkpoint selected by
# validation Mean-Mixture RetMSE@10) + Mean-Mixture retrieval cache +
# Stage2 (reused unmodified train_r_stage2_lambda01.py). Full-Candidate,
# fresh-every-step, candidate-gradient-ON throughout.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CELL="$1"; REF="$2"; PRED_LEN="$3"; SEQ_LEN="$4"; BASE_CKPT="$5"

OUT_ROOT="results/TRACK-HARD-EXPERT-V5-FULL01/${CELL}"
CKPT_ROOT="checkpoints/track_hard_expert_v5_full01/${CELL}"

echo "=== [${CELL}] Hard-Expert-V5 Stage-1 (10 fixed epochs, Mean-Mixture val criterion) ==="
python -u scripts/train_hard_expert_v5_full01.py \
  --reference_ckpt "${REF}" --cell "${CELL}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" \
  --skip_init_hash_check \
  --out_dir "results/TRACK-HARD-EXPERT-V5-FULL01" --checkpoints "checkpoints/track_hard_expert_v5_full01"

SELECTED_EPOCH=$(python3 -c "import json; print(json.load(open('${OUT_ROOT}/checkpoint_selection.json'))['mean_mixture_best_epoch'])")
SELECTED_CKPT="${CKPT_ROOT}/checkpoint_epoch${SELECTED_EPOCH}.pth"

echo "=== [${CELL}] Hard-Expert-V5 Mean-Mixture retrieval cache (selected checkpoint: ${SELECTED_CKPT}) ==="
python -u scripts/build_v_meanmix_cache01.py --arm V5 --reference_ckpt "${REF}" \
  --retriever_checkpoint "${SELECTED_CKPT}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 --tau_s 0.1 \
  --out_dir "${OUT_ROOT}/cache"

echo "=== [${CELL}] Hard-Expert-V5 Stage-2 (reused unmodified train_r_stage2_lambda01.py) ==="
python -u scripts/train_r_stage2_lambda01.py --reference_ckpt "${REF}" --pred_len "${PRED_LEN}" --seq_len "${SEQ_LEN}" --seed 0 \
  --base_checkpoint "${BASE_CKPT}" --cache_dir "${OUT_ROOT}/cache" --out_dir "${OUT_ROOT}/stage2"

python3 - "$CELL" "$OUT_ROOT" "$SELECTED_EPOCH" "$SELECTED_CKPT" <<'PYEOF'
import json
import sys
cell, out_root, selected_epoch, selected_ckpt = sys.argv[1:5]
sel = json.load(open(f'{out_root}/checkpoint_selection.json'))
stage1 = json.load(open(f'{out_root}/cache/stage1_metrics.json'))
stage2 = json.load(open(f'{out_root}/stage2/metrics.json'))
summary = {
    'cell': cell,
    'selected_epoch': int(selected_epoch), 'selected_checkpoint_path': selected_ckpt,
    'mean_mixture_best_val_retmse10': sel['mean_mixture_best_val_retmse10'],
    'hard_loss_best_epoch': sel['hard_loss_best_epoch'],
    'legacy_rr_best_epoch': sel['legacy_rr_best_epoch'],
    'mean_vs_rr_epochs_agree': sel['mean_vs_rr_epochs_agree'],
    'mean_vs_hard_epochs_agree': sel['mean_vs_hard_epochs_agree'],
    'mean_mixture_test_retmse10': stage1['retmse10'],
    'mean_mixture_test_top10_overlap_vs_roundrobin': stage1['top10_overlap_vs_roundrobin'],
    'stage2_test_mse': stage2['test_mse'], 'stage2_test_mae': stage2.get('test_mae'),
    'stage2_lambda': stage2.get('test_lambda', stage2.get('lambda')),
}
with open(f'{out_root}/final_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'[run_hard_expert_v5] wrote {out_root}/final_summary.json: {summary}')
PYEOF

echo "=== [${CELL}] TRACK-HARD-EXPERT-V5-FULL01 complete ==="
