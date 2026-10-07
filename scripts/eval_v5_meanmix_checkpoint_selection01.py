#!/usr/bin/env python3
"""TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 -- per-epoch V5 Stage1
checkpoint selector using VALIDATION Mean-Mixture RetMSE@10, replacing
the historical Round-Robin validation retMSE@10 criterion.

For every `checkpoint_epoch{N}.pth` under --ckpt_dir (N in --epochs),
loads the frozen model+SlotHeads, evaluates once on the val split, and
reports BOTH:
  - old_rr_val_retmse10   (round_robin_topk_selection -- historical
                            criterion, reported for comparison ONLY)
  - new_mean_val_retmse10 (mean_mixture_topk_selection -- THIS track's
                            checkpoint-selection criterion)
plus Mean AggMSE@10/Recall@10/NDCG@10 as non-selection diagnostics.

best_epoch_mean = argmin_epoch new_mean_val_retmse10. Never touches the
test split. Writes checkpoint_selection.csv/json and selected_checkpoint.json
to --out_dir. No new selection mechanism beyond what
utils/mean_mixture_selection.py and train_t_pure_multislot01.py already
provide -- this script only decides WHICH existing epoch checkpoint to use.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_t_pure_multislot01 import compute_scores_full_grad, round_robin_topk_selection
from utils.full_candidate_bank import compute_scores_full_grad_channel_first
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


@torch.no_grad()
def eval_epoch_on_val(exp, args, model, slot_heads, device, tau_s):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag='val', shuffle=False)
    rr_ret, mean_ret, mean_agg, mean_recall, mean_ndcg = [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
            d_raw = -u

            scores = compute_scores_full_grad_channel_first(model, slot_heads, batch_x, exp.memory_x, c)
            rr_idx = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
            mean_idx, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s, k=TOP_K)

            rr_ret.append(d_raw.gather(1, rr_idx).mean(dim=-1).cpu())
            mean_ret.append(d_raw.gather(1, mean_idx).mean(dim=-1).cpu())

            y_sel = memory_c[mean_idx] + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            mean_agg.append(((r_c - query_future) ** 2).mean(dim=-1).cpu())

            oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), TOP_K, largest=False)
            mean_recall.append(recall_at_k(mean_idx, oracle_idx, TOP_K).cpu())
            mean_ndcg.append(ndcg_at_k(mean_idx, d_raw, cand_mask, TOP_K).cpu())

    return {
        'old_rr_val_retmse10': float(torch.cat(rr_ret).mean()),
        'new_mean_val_retmse10': float(torch.cat(mean_ret).mean()),
        'new_mean_val_agg_mse10': float(torch.cat(mean_agg).mean()),
        'new_mean_val_recall10': float(torch.cat(mean_recall).mean()),
        'new_mean_val_ndcg10': float(torch.cat(mean_ndcg).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--ckpt_dir', required=True, help='directory with checkpoint_epoch{N}.pth files')
    ap.add_argument('--epochs', required=True, help='comma-separated epoch numbers to evaluate, e.g. 1,2,...,10')
    ap.add_argument('--num_slots', type=int, default=5)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    epochs = [int(e) for e in cli.epochs.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed, 'top_k': TOP_K,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear', 'relation_input_space': 'delta_last',
        'relation_teacher_space': 'delta_last', 'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': 16, 'stride': 16,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=cli.num_slots).to(device)
    slot_heads.eval()

    rows = []
    ckpt_dir = Path(cli.ckpt_dir)
    for ep in epochs:
        ckpt_path = ckpt_dir / f'checkpoint_epoch{ep}.pth'
        assert ckpt_path.exists(), f'[ISSUE][ABORT] missing {ckpt_path}'
        bl = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        row = {'epoch': ep, 'checkpoint_path': str(ckpt_path)}
        row.update(eval_epoch_on_val(exp, args, model, slot_heads, device, cli.tau_s))
        rows.append(row)
        print(f'[eval_v5_meanmix_selection] epoch={ep} old_rr_val_retmse10={row["old_rr_val_retmse10"]:.6f} '
             f'new_mean_val_retmse10={row["new_mean_val_retmse10"]:.6f}')

    best_mean = min(rows, key=lambda r: r['new_mean_val_retmse10'])
    best_rr = min(rows, key=lambda r: r['old_rr_val_retmse10'])

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / 'checkpoint_selection.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps({
        'rows': rows,
        'old_rr_best_epoch': best_rr['epoch'],
        'old_rr_best_val_retmse10': best_rr['old_rr_val_retmse10'],
        'new_mean_best_epoch': best_mean['epoch'],
        'new_mean_best_val_retmse10': best_mean['new_mean_val_retmse10'],
        'epoch_changed': best_rr['epoch'] != best_mean['epoch'],
    }, indent=2))
    (out_dir / 'selected_checkpoint.json').write_text(json.dumps({
        'selection_criterion': 'min validation Mean-Mixture RetMSE@10 (test split NEVER used)',
        'selected_epoch': best_mean['epoch'],
        'selected_checkpoint_path': str(ckpt_dir / f'checkpoint_epoch{best_mean["epoch"]}.pth'),
        'tau_s': cli.tau_s,
    }, indent=2))
    print(f'[eval_v5_meanmix_selection] done. old_rr_best_epoch={best_rr["epoch"]} '
         f'new_mean_best_epoch={best_mean["epoch"]} changed={best_rr["epoch"] != best_mean["epoch"]}')


if __name__ == '__main__':
    main()
