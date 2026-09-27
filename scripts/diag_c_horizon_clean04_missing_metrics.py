#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN04, Stage 0 -- metrics missing from
CLEAN03: A0's own block-wise MSE (its single Global Top-10 set, scored per
block) and Oracle ranking quality (Recall@10, NDCG@10, Oracle Top-10 mean/
median student rank, retrieval regret, student score std/effective rank)
for A0 and, if given, an expert checkpoint.

Reuses `build_query_cache`/`candidate_value`/`state_sha` from
`scripts.train_c_horizon_frozen03`, `BLOCKS`/`block_distance` from
`scripts.train_c_horizon_clean02`, `encode_raw`/`arm_score` from
`scripts.train_factorial_e2e01`. NDCG relevance definition: same as
`scripts.train_horizon_retrieval_expert01.ndcg_at_10` if present there,
else the shifted-negative-distance relevance used throughout this
session's D-track scripts (`rel = -d - min(-d)`, masked to 0 for
invalid) -- documented explicitly below since the repo has more than one
NDCG helper.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_c_horizon_clean02 import BLOCKS, block_distance
from scripts.train_c_horizon_clean04_expertwise import BlockAdapter
from scripts.train_c_horizon_frozen03 import build_query_cache, candidate_value
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment

BLOCK_NAMES = ('block1', 'block2', 'block3')


def ndcg_at_k(model_idx, d, valid_mask, k):
    """Relevance = shifted (>=0) negative distance -- same convention as
    `scripts.train_patch_retrieval_expert01.ndcg_at_k` (this session's
    established D-track definition), reused here verbatim in spirit."""
    tgt = (-d).masked_fill(~valid_mask, float('-inf'))
    rel = tgt - tgt.masked_fill(~valid_mask, float('inf')).min(dim=-1, keepdim=True).values
    rel = rel.masked_fill(~valid_mask, 0.0)
    gains = rel.gather(1, model_idx)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=d.device).float()).unsqueeze(0)
    dcg = (gains * disc).sum(-1)
    ideal = rel.topk(k, dim=-1).values
    idcg = (ideal * disc).sum(-1).clamp_min(1e-12)
    return dcg / idcg


def recall_at_k(model_idx, oracle_idx, k):
    hit = (model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)).any(-1)
    return hit.float().sum(-1) / k


@torch.no_grad()
def compute_ranking_metrics(cache, candidate_emb, memory_c_by_channel, channels, adapter, bname, top_k,
                            batch_size, exp, chunk_size, device, limit_rows=0):
    lo, hi = BLOCKS[bname]
    starts = cache['starts']
    n_rows = starts.size(0) if not limit_rows else min(limit_rows, starts.size(0))
    acc = {'recall10': [], 'ndcg10': [], 'oracle_mean_rank': [], 'oracle_median_rank': [],
          'regret': [], 'score_std': [], 'effective_rank': [], 'block_mse': []}
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            h_q = cache['emb'][c][s:e].to(device)
            h_i = candidate_emb[c]
            z_q = adapter(h_q) if adapter is not None else F.normalize(h_q, dim=-1)
            z_i = adapter(h_i) if adapter is not None else F.normalize(h_i, dim=-1)
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            query_future = by[:, :, c]
            s_b = arm_score(z_q, z_i, None).masked_fill(~cand_mask, float('-inf'))

            d_b = block_distance(memory_c, offset_c, query_future, lo, hi, chunk_size)
            d_b = d_b.masked_fill(~cand_mask, float('inf'))
            oracle_idx = stable_topk_indices(d_b, top_k, largest=False)
            model_idx = stable_topk_indices(s_b, top_k, largest=True)

            acc['recall10'].append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            acc['ndcg10'].append(ndcg_at_k(model_idx, d_b, cand_mask, top_k).cpu())

            s_masked = s_b.masked_fill(~cand_mask, float('-inf'))
            ranks = (-s_masked).argsort(dim=-1).argsort(dim=-1).float() + 1.0  # 1-indexed rank by score desc
            oracle_ranks = ranks.gather(1, oracle_idx)
            acc['oracle_mean_rank'].append(oracle_ranks.mean(-1).cpu())
            acc['oracle_median_rank'].append(oracle_ranks.median(-1).values.cpu())

            model_block_mse = d_b.gather(1, model_idx).mean(-1)
            oracle_block_mse = d_b.gather(1, oracle_idx).mean(-1)
            acc['regret'].append((model_block_mse - oracle_block_mse).cpu())

            std_per_row = torch.tensor(
                [float(s_b[b][cand_mask[b]].std()) for b in range(s_b.size(0))])
            acc['score_std'].append(std_per_row)
            p = torch.softmax(s_b.masked_fill(~cand_mask, float('-inf')), dim=-1)
            eff_rank = 1.0 / (p.square().sum(-1).clamp_min(1e-12))
            acc['effective_rank'].append(eff_rank.cpu())

            y_sel = memory_c[model_idx][:, :, lo:hi] + offset_c.view(-1, 1, 1)
            acc['block_mse'].append(((y_sel.mean(dim=1) - query_future[:, lo:hi]) ** 2).mean(-1).cpu())
    return {k: float(torch.cat(v).mean()) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--g_best_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--split', default='val', choices=('val', 'test'))
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN04')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_ckpt = torch.load(cli.g_best_checkpoint, map_location='cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'top_k': cli.top_k, 'tau_topk': 0.1, 'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(g_ckpt['model_state_dict'])
    model.to(device); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    candidate_emb = {c: encode_raw(model, exp.memory_x, c).detach() for c in channels}
    memory_c_by_channel = {c: candidate_value(exp.memory_y, exp.memory_x_last, c).to(device) for c in channels}
    cache = build_query_cache(model, exp, cli.split, channels, device)

    result = {'cell': cli.cell, 'split': cli.split, 'a0': {}}
    for bname in BLOCK_NAMES:
        result['a0'][bname] = compute_ranking_metrics(cache, candidate_emb, memory_c_by_channel, channels,
                                                       None, bname, cli.top_k, cli.batch_size, exp,
                                                       cli.chunk_size, device)
    global_h720 = sum(result['a0'][b]['block_mse'] * (BLOCKS[b][1] - BLOCKS[b][0]) for b in BLOCK_NAMES) / 720.0
    result['a0_global_h720_mse_from_blocks'] = global_h720

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = 'a0_blockwise_metrics.json' if cli.split == 'val' else 'a0_blockwise_metrics_test.json'
    (out_dir / fn).write_text(json.dumps(result, indent=2))
    print(f'[diag_clean04_missing] {cli.cell}/{cli.split}: a0_global_h720_mse_from_blocks={global_h720:.6f}')
    for bname in BLOCK_NAMES:
        m = result['a0'][bname]
        print(f"  {bname}: block_mse={m['block_mse']:.6f} recall10={m['recall10']:.4f} "
             f"ndcg10={m['ndcg10']:.4f} oracle_mean_rank={m['oracle_mean_rank']:.1f} regret={m['regret']:.6f}")


if __name__ == '__main__':
    main()
