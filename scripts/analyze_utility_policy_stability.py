#!/usr/bin/env python3
"""EXPERIMENT 3 -- is candidate utility a fixed target, or does it move?

The utility teacher was measured against one Stage-2. Every arm then trained its
own Stage-2, so the thing the retriever was taught to want may no longer be what
its own forecaster rewards. If the ranking moves, a fixed precomputed teacher is
supervising a target that no longer exists by the time it matters.

Same queries, same candidates, same production forward -- only the Stage-2
parameters differ. Nothing is retrained.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, load_stage2, rank_correlations, unwrap  # noqa: E402
from utils.utility_teacher import _dummy_cache  # noqa: E402

OVERLAP_KS = (1, 5, 10)
COLUMNS = [
    'dataset', 'pred_len', 'baseline', 'method', 'queries', 'channels', 'pool_size',
    'pearson', 'spearman',
    *[f'overlap_at_{k}' for k in OVERLAP_KS],
    'ndcg_at_10', 'best_candidate_agreement',
    'sign_agreement', 'positive_positive', 'positive_to_negative', 'negative_to_positive',
    'baseline_positive_rate', 'method_positive_rate',
    'baseline_mean_utility', 'method_mean_utility',
    'baseline_best_utility', 'method_best_utility', 'checkpoint',
]


@torch.no_grad()
def measure(checkpoint, pool, starts_wanted, max_queries, candidate_chunk, split='test'):
    """U(q, k) for one Stage-2, over a candidate pool fixed by the caller."""
    experiment, args = load_stage2(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    _, loader = experiment._get_data(flag=split, shuffle=False)
    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    pool = pool.to(device)

    utilities, starts = [], []
    seen = 0
    for batch_x, batch_y, batch_start_idx in loader:
        if max_queries and seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        if max_queries and seen + batch_x.size(0) > max_queries:
            keep = max_queries - seen
            batch_x, batch_y, batch_start_idx = batch_x[:keep], batch_y[:keep], batch_start_idx[:keep]
        seen += batch_x.size(0)
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        utility, _ = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=pool,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, batch_x.size(0), device, batch_x.dtype),
            candidate_chunk=candidate_chunk,
        )
        utilities.append(utility.permute(0, 2, 1).cpu())     # [B, C, K]
        starts.extend(int(v) for v in batch_start_idx.cpu().tolist())

    if starts_wanted is not None and starts != starts_wanted:
        raise ValueError(
            'checkpoints disagree on which windows the split contains; the '
            'comparison would not be over the same queries'
        )
    return torch.cat(utilities), starts, args


def compare(baseline, method, valid):
    """Ranking, sign and best-candidate agreement between two utility tables."""
    flat_base = baseline.reshape(-1, baseline.size(-1))
    flat_method = method.reshape(-1, method.size(-1))
    if flat_base.shape != flat_method.shape:
        raise ValueError(f'utility shapes differ: {tuple(flat_base.shape)} vs {tuple(flat_method.shape)}')
    mask = valid.reshape(-1, valid.size(-1))

    pearson, spearman = rank_correlations(flat_base, flat_method, mask)
    out = {'pearson': pearson, 'spearman': spearman}

    floor = torch.finfo(flat_base.dtype).min
    for k in OVERLAP_KS:
        width = min(k, flat_base.size(-1))
        top_base = flat_base.masked_fill(~mask, floor).topk(width, dim=-1).indices
        top_method = flat_method.masked_fill(~mask, floor).topk(width, dim=-1).indices
        hit = (top_base.unsqueeze(-1) == top_method.unsqueeze(-2)).any(-1)
        out[f'overlap_at_{k}'] = float(hit.float().mean())
    out['best_candidate_agreement'] = out['overlap_at_1']

    width = min(10, flat_base.size(-1))
    graded = flat_base.clamp_min(0.0).masked_fill(~mask, 0.0)
    by_method = flat_method.masked_fill(~mask, floor).topk(width, dim=-1).indices
    ideal = graded.topk(width, dim=-1).indices
    discount = 1.0 / torch.log2(torch.arange(width, dtype=torch.float64) + 2.0)
    dcg = (graded.gather(1, by_method).double() * discount).sum(-1)
    idcg = (graded.gather(1, ideal).double() * discount).sum(-1)
    keep = idcg > 0
    out['ndcg_at_10'] = float((dcg[keep] / idcg[keep]).mean()) if keep.any() else float('nan')

    # Sign flips matter more than rank shuffles: a candidate crossing zero turns
    # from something to retrieve into something to avoid.
    base_positive = (flat_base > 0) & mask
    method_positive = (flat_method > 0) & mask
    total = mask.sum().clamp_min(1).float()
    out['sign_agreement'] = float(((base_positive == method_positive) & mask).sum() / total)
    out['positive_positive'] = float((base_positive & method_positive).sum() / total)
    out['positive_to_negative'] = float((base_positive & ~method_positive & mask).sum() / total)
    out['negative_to_positive'] = float((~base_positive & method_positive & mask).sum() / total)
    out['baseline_positive_rate'] = float(base_positive.sum() / total)
    out['method_positive_rate'] = float(method_positive.sum() / total)
    out['baseline_mean_utility'] = float(flat_base[mask].mean())
    out['method_mean_utility'] = float(flat_method[mask].mean())
    out['baseline_best_utility'] = float(flat_base.masked_fill(~mask, floor).max(-1).values.mean())
    out['method_best_utility'] = float(flat_method.masked_fill(~mask, floor).max(-1).values.mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='reference Stage-2 checkpoint')
    parser.add_argument('--method', required=True, nargs='+', help='name=stage2_checkpoint')
    parser.add_argument('--pool_size', type=int, default=500)
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--candidate_chunk', type=int, default=25)
    parser.add_argument('--out_dir', default='./metrics/retrieval_bottleneck')
    args = parser.parse_args()

    # A pool fixed once, spread evenly over the bank, so every checkpoint scores
    # exactly the same candidates rather than each its own favourites.
    probe, saved = load_stage2(args.baseline)
    probe._ensure_memory()
    total = probe.memory_y.size(0)
    width = min(args.pool_size, total)
    pool = torch.linspace(0, total - 1, width).round().long().unique()
    del probe

    baseline_utility, starts, base_args = measure(
        args.baseline, pool, None, args.max_queries, args.candidate_chunk)
    valid = torch.ones_like(baseline_utility, dtype=torch.bool)
    out_dir = Path(args.out_dir)

    for spec in args.method:
        name, _, path = spec.partition('=')
        method_utility, _, _ = measure(path, pool, starts, args.max_queries, args.candidate_chunk)
        row = {
            'dataset': base_args.data, 'pred_len': int(base_args.pred_len),
            'baseline': 'future_kl_full', 'method': name,
            'queries': baseline_utility.size(0), 'channels': baseline_utility.size(1),
            'pool_size': int(pool.numel()), 'checkpoint': path,
        }
        row.update(compare(baseline_utility, method_utility, valid))
        append_row(out_dir / 'utility_policy_stability.csv', row, COLUMNS)
        print(f"{row['dataset']}/{row['pred_len']} {name:<22} "
              f"rho={row['spearman']:+.3f} ov@1={row['overlap_at_1']:.3f} "
              f"ov@10={row['overlap_at_10']:.3f} ndcg@10={row['ndcg_at_10']:.3f} "
              f"sign={row['sign_agreement']:.3f} "
              f"flip+-={row['positive_to_negative']:.3f}")
    print(f'wrote {out_dir}/utility_policy_stability.csv')


if __name__ == '__main__':
    main()
