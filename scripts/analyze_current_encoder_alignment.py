#!/usr/bin/env python3
"""STEP 1 -- what is the *current* Stage-1 encoder actually aligned with?

It was trained on Future-MSE and does badly at reproducing that Oracle. The
question this answers is whether it nevertheless lands in a downstream-useful
neighbourhood: the student score is compared against all three targets over the
same pool, with no retraining.

Reads the precomputed teacher cache, so the pool and the utility measurement are
identical to the ones the training arms will use.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, gap_recovery, rank_correlations  # noqa: E402
from utils.utility_teacher import load_cache  # noqa: E402

COLUMNS = [
    'dataset', 'pred_len', 'split', 'queries', 'channels', 'pool_m',
    'student_future_pearson', 'student_future_spearman',
    'student_residual_pearson', 'student_residual_spearman',
    'student_utility_pearson', 'student_utility_spearman',
    'student_future_ndcg_at_10', 'student_future_ndcg_at_50',
    'student_residual_ndcg_at_10', 'student_residual_ndcg_at_50',
    'student_utility_ndcg_at_10', 'student_utility_ndcg_at_50',
    'future_recall_at_10', 'retrieved_future_mse_at_10', 'future_gap_recovery_at_10',
    'retrieved_utility_at_1', 'retrieved_utility_at_5', 'retrieved_utility_at_10',
    'positive_utility_rate_at_10', 'utility_gap_recovery_at_10',
    'best_utility', 'random_utility_at_10', 'pool_positive_rate', 'base_mse',
    'cache',
]


def graded_relevance(target, valid, mode):
    """Non-negative relevance for NDCG.

    Utility is signed and zero is meaningful -- a harmful candidate has no
    relevance -- so it clamps. The similarity targets are negative MSEs whose
    zero point carries nothing, and clamping would flatten every grade to zero,
    so they are shifted onto the query's own worst valid candidate instead.
    """
    if mode == 'clamp':
        return target.clamp_min(0.0).masked_fill(~valid, 0.0)
    floor = target.masked_fill(~valid, float('inf')).min(-1, keepdim=True).values
    return (target - floor).clamp_min(0.0).masked_fill(~valid, 0.0)


def ndcg(score, relevance, valid, depth, mode='shift'):
    """NDCG of `score`'s ranking under a relevance grade of the target."""
    width = min(depth, int(valid.sum(-1).min()))
    if width < 1:
        return float('nan')
    graded = graded_relevance(relevance, valid, mode)
    by_score = score.masked_fill(~valid, float('-inf')).topk(width, dim=-1).indices
    ideal = graded.topk(width, dim=-1).indices
    discount = 1.0 / torch.log2(torch.arange(width, dtype=torch.float64) + 2.0)
    dcg = (graded.gather(1, by_score).double() * discount).sum(-1)
    idcg = (graded.gather(1, ideal).double() * discount).sum(-1)
    keep = idcg > 0
    return float((dcg[keep] / idcg[keep]).mean()) if keep.any() else float('nan')


def topk_mean(score, values, valid, depth):
    width = min(depth, int(valid.sum(-1).min()))
    idx = score.masked_fill(~valid, float('-inf')).topk(width, dim=-1).indices
    return values.gather(1, idx)


def flatten(cache, key):
    """[N, C, M] -> [N*C, M]; each (query, channel) is its own ranking problem."""
    return cache[key].reshape(-1, cache[key].size(-1))


def analyse(cache_path, seed=0):
    cache = load_cache(cache_path)
    meta = cache['meta']
    student = flatten(cache, 'student')
    future = flatten(cache, 'future')
    residual = flatten(cache, 'residual')
    utility = flatten(cache, 'utility')
    valid = flatten(cache, 'valid')

    row = {
        'dataset': meta['dataset'], 'pred_len': meta['pred_len'], 'split': meta['split'],
        'queries': meta['queries'], 'channels': meta['channels'], 'pool_m': meta['pool_m'],
        'base_mse': float(cache['base_mse'].mean()),
        'pool_positive_rate': float((utility[valid] > 0).float().mean()),
        'cache': str(cache_path),
    }
    for name, target in (('future', future), ('residual', residual), ('utility', utility)):
        pearson, spearman = rank_correlations(student, target, valid)
        row[f'student_{name}_pearson'] = pearson
        row[f'student_{name}_spearman'] = spearman
        mode = 'clamp' if name == 'utility' else 'shift'
        row[f'student_{name}_ndcg_at_10'] = ndcg(student, target, valid, 10, mode)
        row[f'student_{name}_ndcg_at_50'] = ndcg(student, target, valid, 50, mode)

    # Future side: the target Stage-1 was actually trained on.
    depth = min(10, int(valid.sum(-1).min()))
    oracle_future = future.masked_fill(~valid, float('-inf')).topk(depth, dim=-1).indices
    student_future = student.masked_fill(~valid, float('-inf')).topk(depth, dim=-1).indices
    hit = (oracle_future.unsqueeze(-1) == student_future.unsqueeze(-2)).any(-1)
    row['future_recall_at_10'] = float(hit.float().mean())
    row['retrieved_future_mse_at_10'] = float(-topk_mean(student, future, valid, 10).mean())

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.rand(student.shape, generator=generator).masked_fill(~valid, float('-inf'))
    random_future = float(-topk_mean(shuffled, future, valid, 10).mean())
    oracle_future_mse = float(-future.gather(1, oracle_future).mean())
    row['future_gap_recovery_at_10'] = gap_recovery(
        row['retrieved_future_mse_at_10'], random_future, oracle_future_mse)

    # Utility side: the target that actually predicts downstream gain.
    for k in (1, 5, 10):
        row[f'retrieved_utility_at_{k}'] = float(topk_mean(student, utility, valid, k).mean())
    row['positive_utility_rate_at_10'] = float(
        (topk_mean(student, utility, valid, 10) > 0).float().mean())
    row['best_utility'] = float(utility.masked_fill(~valid, float('-inf')).max(-1).values.mean())
    row['random_utility_at_10'] = float(topk_mean(shuffled, utility, valid, 10).mean())
    oracle_utility = float(topk_mean(utility, utility, valid, 10).mean())
    row['utility_gap_recovery_at_10'] = gap_recovery(
        row['retrieved_utility_at_10'], row['random_utility_at_10'], oracle_utility,
        higher_is_better=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, nargs='+')
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    for path in args.cache:
        row = analyse(path)
        print(f"=== {row['dataset']}/{row['pred_len']} {row['split']} ===")
        for key in COLUMNS:
            if key == 'cache':
                continue
            value = row[key]
            print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
        if args.csv:
            append_row(args.csv, row, COLUMNS)
    if args.csv:
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
