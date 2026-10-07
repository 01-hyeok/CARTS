#!/usr/bin/env python3
"""TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 -- Shared-Top-100 (P100)
variant of eval_v5_meanmix_checkpoint_selection01.py. Same purpose
(select the V5 Stage1 epoch checkpoint by VALIDATION Mean-Mixture
RetMSE@10 instead of Round-Robin), but scores are restricted to the
precomputed Top-100 pool (`scripts/precompute_candidate_pool01.py`
output) via the EXISTING UNMODIFIED `compute_scores` /
`CandidatePoolCache` from `scripts/train_retriever_pool01.py` -- never
re-derives the pool, never touches full-memory candidates.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.candidate_pool import (
    CandidatePoolConfig, encode_pooled_candidates, gather_candidate_histories,
    gather_candidate_values, pooled_future_mse,
)
from utils.full_candidate_bank import encode_raw_channel_first
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


def compute_scores_pool_channel_first(model, slot_heads, batch_x, exp, channel, pool_cache, batch_start_idx, device):
    """A1 (channel-first) variant of `compute_scores`'s coarse_topk
    branch (`scripts/train_retriever_pool01.py`) -- identical math, only
    swaps which `encode_fn` is passed to the already-generic
    `encode_pooled_candidates`/direct query encode. Never modifies the
    shared `train_retriever_pool01.py` (other tracks depend on its exact
    historical behavior)."""
    z_q = encode_raw_channel_first(model, batch_x, channel)
    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
    z_k = encode_pooled_candidates(lambda x, c: encode_raw_channel_first(model, x, c), pooled_x, channel)
    z_k = torch.nn.functional.normalize(z_k, dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global


@torch.no_grad()
def eval_epoch_on_val(exp, args, model, slot_heads, pool_cfg, pool_cache, device, tau_s):
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
            scores, valid_mask, pool_idx_global = compute_scores_pool_channel_first(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            rr_idx = round_robin_topk_selection(scores, valid_mask, k=TOP_K)
            mean_idx, _, _ = mean_mixture_topk_selection(scores, valid_mask, tau_s, k=TOP_K)

            rr_ret.append(d_pool.gather(1, rr_idx).mean(dim=-1).cpu())
            mean_ret.append(d_pool.gather(1, mean_idx).mean(dim=-1).cpu())

            y_sel = pooled_memory_c.gather(
                1, mean_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            mean_agg.append(((r_c - query_future) ** 2).mean(dim=-1).cpu())

            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)
            mean_recall.append(recall_at_k(mean_idx, oracle_idx, TOP_K).cpu())
            mean_ndcg.append(ndcg_at_k(mean_idx, d_pool, valid_mask, TOP_K).cpu())

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
    ap.add_argument('--epochs', required=True, help='comma-separated epoch numbers to evaluate')
    ap.add_argument('--num_query_views', type=int, default=5)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_metric', default='delta_last_cosine')
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    epochs = [int(e) for e in cli.epochs.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size,
                                   metric=cli.candidate_pool_metric)
    cli.init_seed = cli.seed
    cli.patch_len = 16
    cli.top_k = TOP_K
    cli.batch_size = 32
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=cli.num_query_views).to(device)
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size,
                     'candidate_pool_metric': cli.candidate_pool_metric}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt', expected_meta)

    rows = []
    ckpt_dir = Path(cli.ckpt_dir)
    for ep in epochs:
        ckpt_path = ckpt_dir / f'checkpoint_epoch{ep}.pth'
        assert ckpt_path.exists(), f'[ISSUE][ABORT] missing {ckpt_path}'
        bl = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        row = {'epoch': ep, 'checkpoint_path': str(ckpt_path)}
        row.update(eval_epoch_on_val(exp, args, model, slot_heads, pool_cfg, pool_cache, device, cli.tau_s))
        rows.append(row)
        print(f'[eval_v5_meanmix_selection_pool] epoch={ep} old_rr_val_retmse10={row["old_rr_val_retmse10"]:.6f} '
             f'new_mean_val_retmse10={row["new_mean_val_retmse10"]:.6f}')

    best_mean = min(rows, key=lambda r: r['new_mean_val_retmse10'])
    best_rr = min(rows, key=lambda r: r['old_rr_val_retmse10'])

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        'selection_criterion': 'min validation Mean-Mixture RetMSE@10 within Top-100 pool (test split NEVER used)',
        'selected_epoch': best_mean['epoch'],
        'selected_checkpoint_path': str(ckpt_dir / f'checkpoint_epoch{best_mean["epoch"]}.pth'),
        'tau_s': cli.tau_s,
    }, indent=2))
    print(f'[eval_v5_meanmix_selection_pool] done. old_rr_best_epoch={best_rr["epoch"]} '
         f'new_mean_best_epoch={best_mean["epoch"]} changed={best_rr["epoch"] != best_mean["epoch"]}')


if __name__ == '__main__':
    main()
