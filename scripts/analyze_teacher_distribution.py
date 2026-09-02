#!/usr/bin/env python3
"""STEP 2 -- what do the three teachers actually ask the student to learn?

Before training anything, the supervision signals themselves are compared: how
peaked each one is, how much they disagree, and whether the utility teacher is so
sharp or so flat at a given temperature that the KL would be useless.

The temperature sweep is diagnostic only. It exists so a later "utility teacher
did not help" result cannot be blamed on an unexamined tau.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, rank_correlations  # noqa: E402
from utils.utility_teacher import load_cache  # noqa: E402
from scripts.analyze_current_encoder_alignment import flatten, ndcg, topk_mean  # noqa: E402

EPS = 1e-12
TAUS = (0.01, 0.03, 0.05, 0.07, 0.1, 0.2)
COLUMNS = [
    'dataset', 'pred_len', 'split', 'teacher', 'tau', 'queries', 'pool_m',
    'entropy', 'max_entropy', 'normalized_entropy', 'top1_probability',
    'top10_probability_mass', 'effective_candidates',
    'utility_at_teacher_top10', 'positive_rate_at_teacher_top10',
    'utility_ndcg_at_10', 'kl_to_utility', 'overlap_with_utility_at_10',
    'spearman_with_utility', 'cache',
]


def distribution(score, valid, tau):
    """Teacher distribution over valid candidates at one temperature."""
    logits = score.double() / tau
    logits = logits.masked_fill(~valid, float('-inf'))
    # Subtracting the row max keeps exp finite when a sharp tau blows the scale up.
    logits = logits - logits.max(-1, keepdim=True).values
    probability = logits.exp()
    return probability / probability.sum(-1, keepdim=True).clamp_min(EPS)


def summarize(probability, valid):
    entropy = -(probability * (probability + EPS).log()).sum(-1)
    count = valid.sum(-1).double().clamp_min(1.0)
    depth = min(10, int(valid.sum(-1).min()))
    top10 = probability.topk(depth, dim=-1).values.sum(-1)
    return {
        'entropy': float(entropy.mean()),
        'max_entropy': float(count.log().mean()),
        'normalized_entropy': float((entropy / count.log().clamp_min(EPS)).mean()),
        'top1_probability': float(probability.max(-1).values.mean()),
        'top10_probability_mass': float(top10.mean()),
        'effective_candidates': float(entropy.exp().mean()),
    }


def analyse(cache_path, taus=TAUS):
    cache = load_cache(cache_path)
    meta = cache['meta']
    valid = flatten(cache, 'valid')
    utility = flatten(cache, 'utility')
    scores = {name: flatten(cache, name) for name in ('future', 'residual', 'utility')}
    depth = min(10, int(valid.sum(-1).min()))

    rows = []
    for tau in taus:
        reference = distribution(scores['utility'], valid, tau)
        utility_top = reference.masked_fill(~valid, -1.0).topk(depth, dim=-1).indices
        for name, score in scores.items():
            probability = distribution(score, valid, tau)
            row = {
                'dataset': meta['dataset'], 'pred_len': meta['pred_len'],
                'split': meta['split'], 'teacher': name, 'tau': tau,
                'queries': meta['queries'], 'pool_m': meta['pool_m'],
                'cache': str(cache_path),
            }
            row.update(summarize(probability, valid))
            row['utility_at_teacher_top10'] = float(topk_mean(score, utility, valid, 10).mean())
            row['positive_rate_at_teacher_top10'] = float(
                (topk_mean(score, utility, valid, 10) > 0).float().mean())
            row['utility_ndcg_at_10'] = ndcg(score, utility, valid, 10, 'clamp')
            # KL(this teacher || utility teacher): how much supervision would be
            # lost by keeping the incumbent target.
            row['kl_to_utility'] = float(
                (probability * ((probability + EPS).log() - (reference + EPS).log()))
                .sum(-1).mean())
            own_top = probability.masked_fill(~valid, -1.0).topk(depth, dim=-1).indices
            row['overlap_with_utility_at_10'] = float(
                (own_top.unsqueeze(-1) == utility_top.unsqueeze(-2)).any(-1).float().mean())
            row['spearman_with_utility'] = rank_correlations(score, utility, valid)[1]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, nargs='+')
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    for path in args.cache:
        for row in analyse(path):
            print(f"{row['dataset']}/{row['pred_len']} {row['split']} "
                  f"{row['teacher']:>8s} tau={row['tau']:<5} "
                  f"H={row['entropy']:.3f} eff={row['effective_candidates']:6.2f} "
                  f"top1={row['top1_probability']:.3f} "
                  f"U@10={row['utility_at_teacher_top10']:+.4f} "
                  f"KL→U={row['kl_to_utility']:.4f} "
                  f"ov@10={row['overlap_with_utility_at_10']:.3f}")
            if args.csv:
                append_row(args.csv, row, COLUMNS)
    if args.csv:
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
