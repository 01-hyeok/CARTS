#!/usr/bin/env python3
"""OBSERVABILITY 2 -- do queries with similar pasts need the same candidates?

No training. Every query is scored against one *shared* candidate pool, which is
what makes two queries' utility rankings comparable at all -- per-query pools
would put each query in its own label space. Then queries are paired by
target-past similarity and the pairs are binned by distance:

    if past similarity -> utility ranking agreement, the information is in the
    past and the problem is representation (H1)

    if near-identical pasts disagree about which candidate is best as much as
    random pairs do, the identity is not determined by the past (H2)

Random pairs are the control. Utilities come from the production Stage-2 helper,
so this is the same quantity the reranker was trained on.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_forecast_utility_alignment import base_forecast_via_forward  # noqa: E402
from utils.past_neighborhood import pair_mse  # noqa: E402
from utils.retrieval_diagnostics import append_row, unwrap  # noqa: E402
from utils.utility_teacher import _dummy_cache, load_stage2_reference  # noqa: E402

BINS = ((0.0, 0.01, '0-1%'), (0.01, 0.05, '1-5%'), (0.05, 0.10, '5-10%'),
        (0.10, 0.50, '10-50%'))
SUMMARY_COLUMNS = [
    'dataset', 'pred_len', 'similarity', 'bin', 'pairs', 'pool_size',
    'utility_spearman', 'utility_pearson', 'top1_match', 'top10_overlap',
    'top50_overlap', 'residual_mse', 'residual_cosine', 'residual_norm_diff',
    'past_distance', 'checkpoint',
]


def spearman_rows(a, b):
    """Per-row Spearman between two score vectors over a shared pool."""
    rank = lambda x: x.argsort(dim=-1).argsort(dim=-1).double()
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def pearson_rows(a, b):
    ca = a.double() - a.double().mean(-1, keepdim=True)
    cb = b.double() - b.double().mean(-1, keepdim=True)
    return (ca * cb).sum(-1) / (ca.norm(dim=-1) * cb.norm(dim=-1)).clamp_min(1e-12)


def overlap_at(a, b, k):
    top_a = a.topk(min(k, a.size(-1)), dim=-1).indices
    top_b = b.topk(min(k, b.size(-1)), dim=-1).indices
    hit = (top_a.unsqueeze(-1) == top_b.unsqueeze(-2)).any(-1)
    return hit.float().mean(-1)


@torch.no_grad()
def collect(checkpoint, pool_size, max_queries, candidate_chunk, split):
    experiment, args = load_stage2_reference(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)

    # One shared pool for every query, evenly spaced over the bank: per-query
    # validity is still applied, but the label space is common.
    total = memory_y.size(0)
    width = min(pool_size, total)
    pool = torch.linspace(0, total - 1, width).round().long().unique().to(device)

    parts = {'utility': [], 'past': [], 'residual': [], 'z_query': [], 'valid': []}
    seen = 0
    for batch_x, batch_y, batch_start_idx in loader:
        if max_queries and seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        if max_queries and seen + batch_x.size(0) > max_queries:
            keep = max_queries - seen
            batch_x, batch_y, batch_start_idx = (
                batch_x[:keep], batch_y[:keep], batch_start_idx[:keep])
        seen += batch_x.size(0)
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        utility, _ = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=pool,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, batch_x.size(0), device, batch_x.dtype),
            candidate_chunk=candidate_chunk,
        )
        query_base = base_forecast_via_forward(model, batch_x, memory_y, memory_x_last)
        parts['utility'].append(utility.permute(0, 2, 1).cpu())          # [B, C, P]
        parts['past'].append(batch_x.cpu())
        parts['residual'].append((batch_y - query_base).cpu())
        parts['z_query'].append(torch.stack(
            [model._branch_embedding(batch_x, c, c) for c in model.target_channels()],
            dim=1).cpu())
        parts['valid'].append(cand_mask.index_select(1, pool).cpu())
    return {k: torch.cat(v) for k, v in parts.items()}, args, width


def analyse(data, similarity, bins=BINS, max_pairs=200000):
    """Pair every query with every other, bin by past distance, summarise."""
    channels = data['utility'].size(1)
    rows = []
    for channel in range(channels):
        if similarity == 'stage1_cosine':
            z = torch.nn.functional.normalize(data['z_query'][:, channel], dim=-1)
            distance = 1.0 - z @ z.transpose(0, 1)
        elif similarity == 'raw_pearson':
            past = data['past'][:, :, channel]
            centered = past - past.mean(-1, keepdim=True)
            centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            distance = 1.0 - centered @ centered.transpose(0, 1)
        else:
            distance = pair_mse(data['past'][:, :, channel], data['past'][:, :, channel])
        n = distance.size(0)
        distance.fill_diagonal_(float('inf'))

        utility = data['utility'][:, channel]                            # [Q, P]
        residual = data['residual'][:, :, channel]
        order = distance.argsort(dim=-1)
        for low, high, label in bins:
            start, stop = int(low * (n - 1)), max(int(high * (n - 1)), int(low * (n - 1)) + 1)
            picked = order[:, start:stop]
            rows.append(_summarise(utility, residual, distance, picked, label, channel))
        # Random control: a uniformly drawn partner for each query.
        generator = torch.Generator().manual_seed(0)
        picked = torch.randint(0, n, (n, max(1, stop - start)), generator=generator)
        rows.append(_summarise(utility, residual, distance, picked, 'random', channel))
    return rows


def _summarise(utility, residual, distance, picked, label, channel):
    queries, width = picked.shape
    left = torch.arange(queries).unsqueeze(1).expand(-1, width).reshape(-1)
    right = picked.reshape(-1)
    keep = torch.isfinite(distance[left, right])
    left, right = left[keep], right[keep]
    a, b = utility[left], utility[right]
    ra, rb = residual[left], residual[right]
    return {
        'bin': label, 'channel': channel, 'pairs': int(left.numel()),
        'utility_spearman': float(spearman_rows(a, b).mean()),
        'utility_pearson': float(pearson_rows(a, b).mean()),
        'top1_match': float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        'top10_overlap': float(overlap_at(a, b, 10).mean()),
        'top50_overlap': float(overlap_at(a, b, 50).mean()),
        'residual_mse': float((ra - rb).square().mean()),
        'residual_cosine': float(torch.nn.functional.cosine_similarity(ra, rb, dim=-1).mean()),
        'residual_norm_diff': float((ra.norm(dim=-1) - rb.norm(dim=-1)).abs().mean()),
        'past_distance': float(distance[left, right].mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pool_size', type=int, default=500)
    p.add_argument('--max_queries', type=int, default=512)
    p.add_argument('--candidate_chunk', type=int, default=25)
    p.add_argument('--split', default='test')
    p.add_argument('--similarities', default='raw_l2,stage1_cosine')
    p.add_argument('--csv', default='')
    p.add_argument('--summary_csv', default='')
    a = p.parse_args()

    data, args, width = collect(a.checkpoint, a.pool_size, a.max_queries,
                                a.candidate_chunk, a.split)
    for similarity in a.similarities.split(','):
        rows = analyse(data, similarity)
        grouped = {}
        for row in rows:
            grouped.setdefault(row['bin'], []).append(row)
        for label, group in grouped.items():
            mean = lambda key: sum(r[key] for r in group) / len(group)
            summary = {
                'dataset': args.data, 'pred_len': int(args.pred_len),
                'similarity': similarity, 'bin': label,
                'pairs': sum(r['pairs'] for r in group), 'pool_size': width,
                'checkpoint': a.checkpoint,
            }
            for key in ('utility_spearman', 'utility_pearson', 'top1_match',
                        'top10_overlap', 'top50_overlap', 'residual_mse',
                        'residual_cosine', 'residual_norm_diff', 'past_distance'):
                summary[key] = mean(key)
            print(f"[{args.data} {similarity} {label:>8}] "
                  f"spearman={summary['utility_spearman']:+.4f} "
                  f"top1={summary['top1_match']:.3f} "
                  f"top10={summary['top10_overlap']:.3f} "
                  f"residual_mse={summary['residual_mse']:.4f}")
            if a.summary_csv:
                append_row(a.summary_csv, summary, SUMMARY_COLUMNS)
            if a.csv:
                for row in group:
                    append_row(a.csv, {**summary, **row}, SUMMARY_COLUMNS + ['channel'])


if __name__ == '__main__':
    main()
