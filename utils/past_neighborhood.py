"""Past-neighborhood future ambiguity: is the target past even sufficient?

Every negative result so far has had one escape hatch left: maybe the encoder
simply failed to learn the query -> useful-correction mapping. This measures the
information directly, with no encoder in the loop.

Take queries whose target-channel pasts are near-identical and ask how much
their residuals still differ:

    d_past(q, n) -> 0   =>   d_residual(q, n) -> 0 ?

If residual dispersion collapses as the past neighborhood tightens, the past
carries the information and the problem was the model. If it stays near the
global dispersion, identifying the right historical correction from the target
past is irreducibly ambiguous, and no encoder can fix that.

Three controls make the reading safe:

  positive   a held-out part of the past itself. With past_metric='firsthalf'
             the neighborhood is sorted on the first half of the window and the
             dispersion of the *second* half is reported: a quantity the sort key
             never saw, but which similar pasts must still agree on. (The plain
             tail column degenerates into the sort key itself whenever
             pred_len == seq_len, so it reads as a tightness check only.)
  negative   residuals permuted across the neighbor pool. Any apparent collapse
             here is an artifact of averaging over small groups, not signal.
  reference  the same statistics over all valid neighbors (the global bucket).

Nothing here uses a query future as an input: futures and residuals enter only
as the quantities whose dispersion is being measured.
"""

import math

import torch

EPS = 1e-12


def pair_mse(a, b):
    """[B, T] vs [N, T] -> [B, N] mean squared difference."""
    return (
        a.square().mean(-1, keepdim=True)
        + b.square().mean(-1).unsqueeze(0)
        - 2.0 * torch.matmul(a, b.transpose(0, 1)) / a.size(-1)
    ).clamp_min(0.0)


def znorm(x, eps=1e-6):
    """Level- and scale-free view of a window, so 'near past' cannot just mean
    'same level' -- in ETT the offset dominates raw distance."""
    centered = x - x.mean(-1, keepdim=True)
    return centered / centered.std(-1, keepdim=True).clamp_min(eps)


def bucket_widths(valid_counts, fractions):
    """Per-query neighborhood sizes, at least one neighbor each.

    fraction <= 0 means the single nearest neighbor. A percentile bucket is a
    weaker probe than it sounds: at 1% of 8449 candidates the 84th neighbor is
    not a near-duplicate of anything, so the sharpest question -- how different
    is the residual of the *most similar past that exists* -- needs its own
    bucket.
    """
    return {
        fraction: (
            torch.ones_like(valid_counts, dtype=torch.long)
            if fraction <= 0 else
            (valid_counts.float() * fraction).ceil().long().clamp_min(1)
        )
        for fraction in fractions
    }


def _masked_mean(values, mask):
    weight = mask.float()
    return (values * weight).sum(), weight.sum()


def neighborhood_statistics(
    query_past, memory_past, query_residual, memory_residual,
    query_future, memory_future, valid_mask, fractions=(0.01, 0.05, 0.10, 1.0),
    best_id=None, num_identities=0, shuffled_residual=None, tail=0,
    knn_block=512, entropy_sample=32, seed=0, query_control=None,
    memory_control=None,
):
    """Dispersion statistics inside past-nearest neighborhoods, one query chunk.

    Returns running sums keyed by (fraction, metric); the caller accumulates
    across chunks and divides at the end. Sums rather than means because the
    per-query neighborhood sizes differ.
    """
    device = query_past.device
    n_query = query_past.size(0)
    horizon = query_residual.size(-1)
    tail = tail or min(horizon, query_past.size(-1))

    sampler = torch.Generator(device=device)
    sampler.manual_seed(seed)
    distance = pair_mse(query_past, memory_past)
    distance = distance.masked_fill(~valid_mask, float('inf'))
    valid_counts = valid_mask.sum(-1)

    # Every statistic is a lookup into one of these [B, N] tables, so the
    # neighbor windows themselves are never materialised.
    residual_distance = pair_mse(query_residual, memory_residual)
    future_distance = pair_mse(query_future, memory_future)
    # The control window is whatever the caller wants held out; by default the
    # tail of the same window the search sorted on, which only measures
    # tightness. Pass query_control/memory_control for a genuine held-out part.
    if query_control is not None:
        tail_distance = pair_mse(query_control, memory_control)
    else:
        tail_distance = pair_mse(query_past[:, -tail:], memory_past[:, -tail:])
    cosine = torch.matmul(
        torch.nn.functional.normalize(query_residual, dim=-1),
        torch.nn.functional.normalize(memory_residual, dim=-1).transpose(0, 1),
    )
    shuffled_distance = (
        pair_mse(query_residual, shuffled_residual)
        if shuffled_residual is not None else None
    )

    widths = bucket_widths(valid_counts, fractions)
    max_width = int(max(int(w.max()) for w in widths.values()))
    max_width = min(max_width, distance.size(-1))
    order = distance.topk(max_width, dim=-1, largest=False).indices
    position = torch.arange(max_width, device=device).unsqueeze(0)

    gather = lambda table: table.gather(1, order)
    sorted_tables = {
        'past_distance': gather(distance),
        'residual_pair_mse': gather(residual_distance),
        'future_pair_mse': gather(future_distance),
        'past_tail_pair_mse': gather(tail_distance),
        'residual_cosine': gather(cosine),
    }
    if shuffled_distance is not None:
        sorted_tables['shuffled_residual_pair_mse'] = gather(shuffled_distance)

    sums = {}
    for fraction, width in widths.items():
        keep = (position < width.unsqueeze(1)) & torch.isfinite(
            sorted_tables['past_distance']
        )
        for name, table in sorted_tables.items():
            total, count = _masked_mean(table, keep)
            sums[(fraction, name)] = float(total)
            sums[(fraction, name + '__count')] = float(count)

        # k-NN residual prediction: the neighborhood mean as the estimate of the
        # query's own residual. Accumulated in blocks so a 10% neighborhood of a
        # large bank never has to exist as a [B, k, T] tensor.
        summed = torch.zeros(n_query, horizon, device=device)
        counted = torch.zeros(n_query, 1, device=device)
        for start in range(0, max_width, knn_block):
            stop = min(start + knn_block, max_width)
            block_keep = keep[:, start:stop].float().unsqueeze(-1)
            summed += (memory_residual[order[:, start:stop]] * block_keep).sum(1)
            counted += block_keep.sum(1)
        predicted = summed / counted.clamp_min(1.0)
        has_neighbor = (counted.squeeze(-1) > 0)
        error = (predicted - query_residual).square().mean(-1)
        sums[(fraction, 'knn_residual_mse')] = float(error[has_neighbor].sum())
        sums[(fraction, 'knn_residual_mse__count')] = float(has_neighbor.sum())

        if best_id is not None and num_identities:
            # Entropy is compared across buckets, so every bucket must be scored
            # on the same number of members: log(group) normalisation alone
            # would make the widest bucket look like the most agreeing one
            # purely because its ceiling is larger.
            width_i = width.clamp_max(max_width)
            usable = has_neighbor & (width_i >= entropy_sample)
            score = torch.rand(n_query, max_width, device=device, generator=sampler)
            score = score.masked_fill(~keep, -1.0)
            picked = score.topk(min(entropy_sample, max_width), dim=-1).indices
            neighbor_ids = best_id[order.gather(1, picked)]
            counts = torch.zeros(n_query, num_identities, device=device)
            counts.scatter_add_(1, neighbor_ids, torch.ones_like(neighbor_ids, dtype=torch.float))
            group = counts.sum(-1, keepdim=True).clamp_min(1.0)
            probability = counts / group
            entropy = -(probability * (probability + EPS).log()).sum(-1)
            normalised = (entropy / math.log(entropy_sample))[usable]
            sums[(fraction, 'best_candidate_entropy')] = float(normalised.sum())
            sums[(fraction, 'best_candidate_entropy__count')] = float(usable.sum())

    return sums


def merge_sums(total, part):
    for key, value in part.items():
        total[key] = total.get(key, 0.0) + value
    return total


def finalize(sums, fractions, metrics):
    """Sums -> per-fraction means, plus the ratios the verdict is read from."""
    rows = {}
    for fraction in fractions:
        row = {}
        for metric in metrics:
            total = sums.get((fraction, metric))
            count = sums.get((fraction, metric + '__count'), 0.0)
            row[metric] = (total / count) if (total is not None and count) else float('nan')
        row['residual_cosine_dispersion'] = 1.0 - row.get('residual_cosine', float('nan'))
        rows[fraction] = row

    reference = rows.get(1.0, {})
    for fraction, row in rows.items():
        for metric in ('residual_pair_mse', 'future_pair_mse', 'past_tail_pair_mse',
                       'past_distance', 'shuffled_residual_pair_mse'):
            base = reference.get(metric)
            row[metric + '_ratio'] = (
                row[metric] / base if base not in (None, 0.0) and not math.isnan(base)
                else float('nan')
            )
    return rows
