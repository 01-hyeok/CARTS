#!/usr/bin/env python3
"""TRACK-A-HORIZON-RETRIEVAL-EXPERT01 -- pre-training Oracle contrast.

TRACK-A-HORIZON-RETRIEVAL-HEADROOM01's "Global Oracle" ranks candidates by
their OWN individual full-horizon future MSE and averages the Top-10 --
this is NOT the same as the Top-10 SET that minimizes the resulting
aggregate's MSE (averaging can benefit from complementary errors across
candidates, which individual-MSE ranking cannot see). Exact optimization
over C(N,10) is combinatorial and infeasible at this candidate-bank scale
(N in the thousands to tens of thousands); this script runs a GREEDY
approximate search instead -- at each of the 10 steps, add whichever
remaining valid candidate most reduces the running uniform-mean's MSE.
This is provably NOT guaranteed optimal (greedy set selection has no
general optimality guarantee here) and its result is NEVER called "optimal
Global Oracle" in any artifact or report this script produces -- only
"greedy uniform-aggregate Global set".

Run on a bounded, seed=0-selected query subset per cell (not the full
split) -- this is an expensive O(K*N) search per query and is explicitly a
SECONDARY diagnostic, not something the main training experiment should
wait on indefinitely.
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
from scripts.diag_horizon_retrieval_headroom01 import BLOCKS, _gather_mean, _mse
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value


@torch.no_grad()
def greedy_uniform_aggregate_picks(memory_c, offset_c, query_future, valid_mask, k,
                                   chunk_size=4096):
    """Greedy approximate search for the Top-k SET minimizing the resulting
    uniform-mean's MSE (not each candidate's own individual MSE). Returns
    picks [B, k], selected in greedy order (NOT MSE-sorted -- step order is
    the search order, unlike `stable_topk_indices`)."""
    bsz = offset_c.size(0)
    n, h = memory_c.shape
    device = memory_c.device
    selected = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    running_sum = torch.zeros(bsz, h, device=device)
    picks = []
    for step in range(1, k + 1):
        best_mse = torch.full((bsz,), float('inf'), device=device)
        best_idx = torch.zeros(bsz, dtype=torch.long, device=device)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = memory_c[start:end]  # [c, h]
            new_sum = running_sum.unsqueeze(1) + chunk.unsqueeze(0)  # [B, c, h]
            new_mean = new_sum / step + offset_c.view(-1, 1, 1)
            mse = ((new_mean - query_future.unsqueeze(1)) ** 2).mean(-1)  # [B, c]
            invalid = (~valid_mask[:, start:end]) | selected[:, start:end]
            mse = mse.masked_fill(invalid, float('inf'))
            chunk_best_mse, chunk_best_local = mse.min(dim=-1)
            improve = chunk_best_mse < best_mse
            best_mse = torch.where(improve, chunk_best_mse, best_mse)
            best_idx = torch.where(improve, chunk_best_local + start, best_idx)
        picks.append(best_idx)
        running_sum = running_sum + memory_c[best_idx]
        selected = selected.scatter(1, best_idx.unsqueeze(-1), True)
    return torch.stack(picks, dim=1)


def _fixed_subset_batches(exp, n_queries, seed=0):
    _, loader = exp._get_data(flag='test', shuffle=False)
    xs, ys, starts = [], [], []
    for bx, by, bstart in loader:
        xs.append(bx); ys.append(by)
        starts.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
    x_all, y_all, start_all = torch.cat(xs), torch.cat(ys), torch.cat(starts)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_all.size(0), generator=g)[:min(n_queries, x_all.size(0))]
    idx, _ = idx.sort()
    return x_all[idx], y_all[idx], start_all[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--n_queries', type=int, default=256)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01')
    cli = ap.parse_args()

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size,
        'seed': 0, 'top_k': cli.top_k, 'tau_topk': 0.1,
    })
    exp._ensure_memory()
    device = exp.device
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    x_sub, y_sub, start_sub = _fixed_subset_batches(exp, cli.n_queries, cli.subset_seed)
    n = x_sub.size(0)

    global_ind_sum, global_greedy_sum, block_oracle_sum = 0.0, 0.0, 0.0
    n_rows = 0
    for s in range(0, n, cli.batch_size):
        bx = x_sub[s:s + cli.batch_size].float().to(device)
        by = y_sub[s:s + cli.batch_size].float().to(device)
        bstart = start_sub[s:s + cli.batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
            query_future = by[:, :, c]

            u_g = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            u_g = u_g.masked_fill(~cand_mask, float('-inf'))
            picks_ind = stable_topk_indices(u_g, cli.top_k, largest=True)
            yhat_ind = _gather_mean(memory_c, offset_c, picks_ind, 0, 720)
            global_ind_sum += float(_mse(yhat_ind, query_future).sum())

            picks_greedy = greedy_uniform_aggregate_picks(memory_c, offset_c, query_future,
                                                          cand_mask, cli.top_k, cli.chunk_size)
            yhat_greedy = _gather_mean(memory_c, offset_c, picks_greedy, 0, 720)
            global_greedy_sum += float(_mse(yhat_greedy, query_future).sum())

            block_preds = []
            for lo, hi in BLOCKS.values():
                u_b = individual_utility_memsafe(memory_c[:, lo:hi], offset_c,
                                                 query_future[:, lo:hi], cli.chunk_size)
                u_b = u_b.masked_fill(~cand_mask, float('-inf'))
                picks_b = stable_topk_indices(u_b, cli.top_k, largest=True)
                block_preds.append(_gather_mean(memory_c, offset_c, picks_b, lo, hi))
            yhat_block = torch.cat(block_preds, dim=1)
            block_oracle_sum += float(_mse(yhat_block, query_future).sum())
        n_rows += bx.size(0)

    n_channels = max(len(channels), 1)
    denom = n_rows * n_channels
    result = {
        'cell': cli.cell, 'n_queries': n, 'n_channels': n_channels,
        'method': 'greedy approximate uniform-aggregate search (NOT guaranteed optimal)',
        'global_individual_ranked_top10_mse': global_ind_sum / max(denom, 1),
        'global_greedy_uniform_aggregate_top10_mse': global_greedy_sum / max(denom, 1),
        'block_oracle_mse': block_oracle_sum / max(denom, 1),
    }
    result['greedy_vs_individual_ranked_delta'] = (
        result['global_individual_ranked_top10_mse'] - result['global_greedy_uniform_aggregate_top10_mse'])
    result['block_vs_greedy_delta'] = (
        result['global_greedy_uniform_aggregate_top10_mse'] - result['block_oracle_mse'])

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'oracle_contrast_greedy.json').write_text(json.dumps(result, indent=2))
    print(f"[oracle_contrast] {cli.cell}: individual-ranked={result['global_individual_ranked_top10_mse']:.6f} "
         f"greedy-uniform={result['global_greedy_uniform_aggregate_top10_mse']:.6f} "
         f"block_oracle={result['block_oracle_mse']:.6f}")


if __name__ == '__main__':
    main()
