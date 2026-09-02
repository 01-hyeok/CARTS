#!/usr/bin/env python3
"""Is the target past even sufficient to identify a useful correction?

Separates "the encoder failed to learn it" from "the input never contained it".
For every test query, the nearest pasts in the memory bank are collected and the
dispersion of *their* residuals is measured. See utils/past_neighborhood.py for
the controls; the decisive columns are

    residual_pair_mse_ratio   1.0 means a tight past neighborhood is no more
                              informative about the residual than a random one
    knn_residual_mse          the error of predicting the query residual with
                              its past-neighborhood mean, to compare against
                              what the trained ResDirect arm achieves
    best_candidate_entropy    do near-identical pasts even agree on which
                              historical candidate is best

The neighborhood is drawn from the memory bank under the same temporal validity
mask retrieval uses, so a neighbor can never be a window that overlaps the query.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_residual_oracle import prepare  # noqa: E402
from utils.past_neighborhood import (  # noqa: E402
    finalize, merge_sums, neighborhood_statistics, znorm,
)
from utils.retrieval_diagnostics import append_row, load_stage2  # noqa: E402
from utils.utility_selection import utility_from_residuals  # noqa: E402

METRICS = ('past_distance', 'residual_pair_mse', 'future_pair_mse',
           'past_tail_pair_mse', 'residual_cosine', 'knn_residual_mse',
           'shuffled_residual_pair_mse', 'best_candidate_entropy')

COLUMNS = [
    'dataset', 'pred_len', 'channel', 'channel_name', 'past_metric', 'split',
    'bucket', 'neighbors_mean', 'n_query', 'n_candidates',
    'past_distance', 'past_distance_ratio',
    'residual_pair_mse', 'residual_pair_mse_ratio',
    'residual_cosine', 'residual_cosine_dispersion',
    'future_pair_mse', 'future_pair_mse_ratio',
    'past_tail_pair_mse', 'past_tail_pair_mse_ratio',
    'shuffled_residual_pair_mse', 'shuffled_residual_pair_mse_ratio',
    'knn_residual_mse', 'residual_power', 'residual_explained_fraction',
    'best_candidate_entropy', 'checkpoint',
]


@torch.no_grad()
def best_candidate_ids(memory_residual, subset, starts, memory_bank, alpha,
                       chunk=2048):
    """For each memory window as a pseudo-query, which shared candidate is best.

    A shared candidate subset is what makes the identities comparable across
    queries at all: per-query pools would put every query in its own label space
    and the entropy would be meaningless.
    """
    device = memory_residual.device
    n = memory_residual.size(0)
    subset_residual = memory_residual[subset]
    out = torch.zeros(n, dtype=torch.long, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        utility = utility_from_residuals(
            memory_residual[start:stop], subset_residual, alpha
        )
        mask, _ = memory_bank.valid_mask_batch(starts[start:stop])
        mask = mask.bool().to(device)[:, subset]
        out[start:stop] = utility.masked_fill(~mask, float('-inf')).argmax(-1)
    return out


@torch.no_grad()
def run_channel(experiment, data, channel, fractions, past_metric, chunk,
                identity_subset, best_id, alpha, entropy_sample=32):
    device = experiment.device
    memory_past = data['memory_x'][:, :, channel]
    memory_residual = data['memory_residual'][:, :, channel]
    memory_future = data['memory_y'][:, :, channel]
    query_past_all = data['query_x'][:, :, channel]
    query_residual_all = data['query_residual'][:, :, channel]
    query_future_all = data['query_y'][:, :, channel]

    tail = 0
    query_control = memory_control = None
    if past_metric == 'znorm':
        memory_past_view = znorm(memory_past)
        query_past_view = znorm(query_past_all)
    elif past_metric == 'firsthalf':
        # Sort on the first half, then report the second half as the control:
        # a part of the past the neighbour search never looked at.
        half = memory_past.size(-1) // 2
        memory_past_view = memory_past[:, :half]
        query_past_view = query_past_all[:, :half]
        memory_control = memory_past[:, half:]
        query_control = query_past_all[:, half:]
    else:
        memory_past_view = memory_past
        query_past_view = query_past_all

    generator = torch.Generator(device='cpu').manual_seed(0)
    permutation = torch.randperm(memory_residual.size(0), generator=generator)
    shuffled = memory_residual[permutation.to(device)]

    sums = {}
    n_query = query_past_all.size(0)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
        part = neighborhood_statistics(
            query_past_view[start:stop], memory_past_view,
            query_residual_all[start:stop], memory_residual,
            query_future_all[start:stop], memory_future,
            mask.to(device), fractions,
            best_id=best_id, num_identities=len(identity_subset),
            shuffled_residual=shuffled, tail=tail,
            entropy_sample=entropy_sample,
            query_control=None if query_control is None else query_control[start:stop],
            memory_control=memory_control,
        )
        merge_sums(sums, part)
    rows = finalize(sums, fractions, METRICS)

    # Predicting zero is the do-nothing baseline: its error is the residual's
    # own power, which is what "explained fraction" is measured against.
    residual_power = float(query_residual_all.square().mean())
    for row in rows.values():
        row['residual_power'] = residual_power
        row['residual_explained_fraction'] = 1.0 - row['knn_residual_mse'] / residual_power
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--past_metric', default='raw',
                   choices=['raw', 'znorm', 'firsthalf'])
    p.add_argument('--entropy_sample', type=int, default=32)
    p.add_argument('--fractions', default='0,0.01,0.05,0.1,1.0')
    p.add_argument('--identity_pool', type=int, default=2000)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--csv', default='')
    a = p.parse_args()

    fractions = tuple(float(v) for v in a.fractions.split(','))
    experiment, saved = load_stage2(a.checkpoint)
    experiment._ensure_memory()
    data = prepare(experiment, a.split, a.max_batches)
    device = experiment.device

    channels = data['query_x'].size(-1)
    names = getattr(experiment.memory_bank.sampler.dataset, 'channel_names', None) or \
        [f'ch{i}' for i in range(channels)]
    starts = experiment.memory_bank.memory_starts
    n_memory = data['memory_x'].size(0)
    generator = torch.Generator(device='cpu').manual_seed(0)
    subset = torch.randperm(n_memory, generator=generator)[:min(a.identity_pool, n_memory)]
    subset = subset.to(device)

    for channel in range(channels):
        best_id = best_candidate_ids(
            data['memory_residual'][:, :, channel], subset, starts,
            experiment.memory_bank, a.alpha,
        )
        rows = run_channel(
            experiment, data, channel, fractions, a.past_metric, a.chunk,
            subset, best_id, a.alpha, a.entropy_sample,
        )
        for fraction, row in rows.items():
            label = 'nearest1' if fraction <= 0 else (
                'all' if fraction >= 1.0 else f'{fraction * 100:g}%'
            )
            row.update({
                'dataset': saved.data, 'pred_len': int(saved.pred_len),
                'channel': channel, 'channel_name': names[channel],
                'past_metric': a.past_metric, 'split': a.split, 'bucket': label,
                'neighbors_mean': row.get('past_distance__count', float('nan')),
                'n_query': int(data['query_x'].size(0)), 'n_candidates': n_memory,
                'checkpoint': a.checkpoint,
            })
            print(
                f"ch{channel} {label:>8}: d_past={row['past_distance']:.4f} "
                f"({row['past_distance_ratio']:.3f} of global)  "
                f"residual_ratio={row['residual_pair_mse_ratio']:.3f}  "
                f"knn_mse={row['knn_residual_mse']:.4f} "
                f"(explains {row['residual_explained_fraction'] * 100:.1f}%)  "
                f"entropy={row['best_candidate_entropy']:.3f}"
            )
            if a.csv:
                append_row(a.csv, row, COLUMNS)


if __name__ == '__main__':
    main()
