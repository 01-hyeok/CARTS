#!/usr/bin/env python3
"""STEP 1 -- is the retrieved pool already good enough, and only the filtering bad?

The previous diagnosis left one specific contradiction: shuffling the retrieval
hurts (so it carries query-specific information) while using it also hurts (so
best alpha is 0). That is what a pool containing both helpful and harmful
candidates looks like once you average over it.

This measures, inside the pool the current retriever already returns:

  pool quality      how many Top-M candidates actually improve the forecast
  oracle filtering  what the forecast becomes if only the harmful ones are dropped

Nothing is trained here. The utility mask uses query futures, so every filtered
variant is an upper bound, not a deployable model -- it answers whether a
filter is worth learning at all.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap  # noqa: E402
from scripts.analyze_residual_oracle import _pair_mse, prepare  # noqa: E402

POOL_SIZES = (10, 50, 100, 200, 500)
EPS = 1e-8

COLUMNS = [
    'dataset', 'pred_len', 'retriever', 'candidate_pool_m', 'alpha', 'top_k',
    'positive_rate', 'at_least_one_positive_rate',
    'mean_positive_count', 'median_positive_count',
    'mean_utility', 'mean_positive_utility', 'best_available_utility',
    'base_mse', 'base_mae', 'current_retrieval_mse', 'current_retrieval_mae',
    'oracle_positive_uniform_mse', 'oracle_positive_uniform_mae',
    'oracle_positive_weighted_mse', 'oracle_positive_weighted_mae',
    'oracle_utility_weighted_mse', 'oracle_utility_weighted_mae',
    'oracle_best_single_mse', 'oracle_best_single_mae',
    'global_utility_oracle_mse', 'global_utility_oracle_mae',
    'filtering_gain', 'filtering_recovery', 'checkpoint',
]


def _retriever_scores(experiment, model, data, start, stop, channel, cand_mask, retriever):
    if retriever == 'raw_l2':
        qp = data['query_x'][start:stop, :, channel] - data['query_x'][start:stop, -1:, channel]
        kp = data['memory_x'][:, :, channel] - data['memory_x'][:, -1:, channel]
        return -_pair_mse(qp, kp)
    if retriever == 'learned':
        sources = model.source_channels(channel)
        if channel not in sources or experiment.key_bank is None:
            return None
        z_q = model._branch_embedding(data['query_x'][start:stop], channel, channel)
        z_k = experiment.key_bank[channel, sources.index(channel)].to(z_q.device, z_q.dtype)
        return torch.matmul(z_q, z_k.transpose(0, 1))
    raise ValueError(f'unknown retriever: {retriever}')


@torch.no_grad()
def analyse(checkpoint_path, retriever, pool_m, alpha=1.0, top_k=10, tau=0.1,
            max_batches=0, chunk=256):
    experiment, args = load_stage2(checkpoint_path)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    data = prepare(experiment, 'test', max_batches)

    channels = data['query_y'].size(-1)
    n_query = data['query_y'].size(0)
    zeros = torch.zeros_like(data['query_base'])
    corrections = {name: zeros.clone() for name in (
        'current', 'positive_uniform', 'positive_weighted', 'utility_weighted',
        'best_single', 'global_oracle',
    )}
    stats = {key: [] for key in (
        'positive_rate', 'at_least_one', 'positive_count', 'mean_utility',
        'mean_positive_utility', 'best_utility',
    )}
    median_pool = []

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        cand_mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
        cand_mask = cand_mask.to(experiment.device)

        for c in range(channels):
            q_res = data['query_residual'][start:stop, :, c]
            k_res = data['memory_residual'][:, :, c]
            horizon = float(k_res.size(-1))
            utility = (
                2.0 * alpha * torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
                - (alpha ** 2) * k_res.square().mean(-1).unsqueeze(0)
            )

            scores = _retriever_scores(
                experiment, model, data, start, stop, c, cand_mask, retriever
            )
            if scores is None:
                return None
            scores = scores.masked_fill(~cand_mask, float('-inf'))

            width = min(pool_m, k_res.size(0))
            pool = scores.topk(width, dim=-1).indices          # [b, M]
            pool_utility = utility.gather(1, pool)             # [b, M]
            pool_valid = cand_mask.gather(1, pool)
            positive = (pool_utility > 0) & pool_valid

            stats['positive_rate'].append(
                positive.sum(-1).float() / pool_valid.sum(-1).clamp_min(1)
            )
            stats['at_least_one'].append(positive.any(-1).float())
            stats['positive_count'].append(positive.sum(-1).float())
            median_pool.append(positive.sum(-1).float().cpu())
            stats['mean_utility'].append(
                (pool_utility * pool_valid).sum(-1) / pool_valid.sum(-1).clamp_min(1)
            )
            stats['mean_positive_utility'].append(
                (pool_utility * positive).sum(-1) / positive.sum(-1).clamp_min(1)
            )
            stats['best_utility'].append(
                pool_utility.masked_fill(~pool_valid, float('-inf')).max(-1).values
            )

            pool_residual = k_res[pool]                        # [b, M, T]
            # Method 1 -- the retriever's own weighting, nothing removed.
            weights = torch.softmax(
                scores.gather(1, pool).masked_fill(~pool_valid, float('-inf')) / tau, dim=-1
            )
            corrections['current'][start:stop, :, c] = (
                pool_residual * weights.unsqueeze(-1)
            ).sum(1)

            positive_f = positive.float()
            has_positive = positive.any(-1, keepdim=True).float()

            # Method 2 -- drop the harmful ones, average the rest.
            corrections['positive_uniform'][start:stop, :, c] = (
                (pool_residual * positive_f.unsqueeze(-1)).sum(1)
                / positive_f.sum(-1, keepdim=True).clamp_min(EPS)
            ) * has_positive

            # Method 3 -- same removal, but keep the retriever's weights.
            masked_w = weights * positive_f
            masked_w = masked_w / masked_w.sum(-1, keepdim=True).clamp_min(EPS)
            corrections['positive_weighted'][start:stop, :, c] = (
                (pool_residual * masked_w.unsqueeze(-1)).sum(1) * has_positive
            )

            # Method 4 -- weight by utility itself (analysis upper bound).
            u_w = pool_utility.clamp_min(0.0) * pool_valid
            u_w = u_w / u_w.sum(-1, keepdim=True).clamp_min(EPS)
            corrections['utility_weighted'][start:stop, :, c] = (
                (pool_residual * u_w.unsqueeze(-1)).sum(1) * has_positive
            )

            # Method 5 -- the single best candidate, or nothing if none helps.
            best = pool_utility.masked_fill(~pool_valid, float('-inf')).argmax(-1)
            corrections['best_single'][start:stop, :, c] = (
                pool_residual[torch.arange(pool.size(0), device=pool.device), best]
                * has_positive
            )

            # Global reference -- utility Top-K over the whole bank.
            g_idx = utility.masked_fill(~cand_mask, float('-inf')).topk(
                min(top_k, k_res.size(0)), dim=-1
            ).indices
            corrections['global_oracle'][start:stop, :, c] = k_res[g_idx].mean(1)

    base = data['query_base']
    true = data['query_y']
    base_mse, base_mae = mse_mae(base, true)
    out = {}
    for name, correction in corrections.items():
        out[name] = mse_mae(base + alpha * correction, true)

    mean = lambda key: float(torch.cat(stats[key]).mean())
    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len),
        'retriever': retriever, 'candidate_pool_m': pool_m,
        'alpha': alpha, 'top_k': top_k,
        'positive_rate': mean('positive_rate'),
        'at_least_one_positive_rate': mean('at_least_one'),
        'mean_positive_count': mean('positive_count'),
        'median_positive_count': float(torch.cat(median_pool).median()),
        'mean_utility': mean('mean_utility'),
        'mean_positive_utility': mean('mean_positive_utility'),
        'best_available_utility': mean('best_utility'),
        'base_mse': base_mse, 'base_mae': base_mae,
        'current_retrieval_mse': out['current'][0], 'current_retrieval_mae': out['current'][1],
        'oracle_positive_uniform_mse': out['positive_uniform'][0],
        'oracle_positive_uniform_mae': out['positive_uniform'][1],
        'oracle_positive_weighted_mse': out['positive_weighted'][0],
        'oracle_positive_weighted_mae': out['positive_weighted'][1],
        'oracle_utility_weighted_mse': out['utility_weighted'][0],
        'oracle_utility_weighted_mae': out['utility_weighted'][1],
        'oracle_best_single_mse': out['best_single'][0],
        'oracle_best_single_mae': out['best_single'][1],
        'global_utility_oracle_mse': out['global_oracle'][0],
        'global_utility_oracle_mae': out['global_oracle'][1],
        'checkpoint': checkpoint_path,
    }
    best_filter = min(
        row['oracle_positive_uniform_mse'], row['oracle_positive_weighted_mse'],
        row['oracle_best_single_mse'],
    )
    row['filtering_gain'] = base_mse - best_filter
    row['filtering_recovery'] = (
        (base_mse - best_filter) / (base_mse - row['global_utility_oracle_mse'] + EPS)
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--retriever', default='learned', choices=['learned', 'raw_l2'])
    parser.add_argument('--pool_m', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = analyse(args.checkpoint, args.retriever, args.pool_m,
                  args.alpha, args.top_k, max_batches=args.max_batches)
    if row is None:
        print('retriever unavailable for this checkpoint')
        return
    print(f"=== {row['dataset']}/{row['pred_len']} {args.retriever} M={args.pool_m} ===")
    for key in COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_row(args.csv, row, COLUMNS)
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
