#!/usr/bin/env python3
"""Does Future-MSE similarity actually rank candidates by downstream usefulness?

Stage-1 is trained to retrieve the candidates whose futures look most like the
query's. That is only the right target if a look-alike future is also the one
that improves the Stage-2 forecast. This measures the two orderings against each
other directly.

Utility is defined by *running the model*, not by residual algebra:

    U(q, k, c) = MSE_c(Y_q, no-retrieval) - MSE_c(Y_q, Stage-2 given only k)

produced by `Model.evaluate_candidate_correction`, which injects candidate k into
the retrieval branch and calls the production `forward`. The offset convention,
the mixer and the gate therefore live in exactly one place. The previous
generation of these diagnostics reconstructed the fusion by hand and was
invalidated by a double-counted offset; nothing here reconstructs anything.

Leakage: query futures appear only inside utility, which is a measurement, never
inside a score a retriever could use.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, load_stage2, rank_correlations, unwrap  # noqa: E402

OVERLAP_KS = (1, 5, 10, 50)
COLUMNS = [
    'dataset', 'pred_len', 'score', 'queries', 'channels', 'pool_size',
    'pearson', 'spearman',
    *[f'overlap_at_{k}' for k in OVERLAP_KS],
    *[f'ndcg_at_{k}' for k in OVERLAP_KS],
    'utility_at_score_top1', 'utility_at_score_top10',
    'best_utility', 'mean_utility', 'positive_rate',
    'base_mse', 'checkpoint',
]


def _dummy_cache(model, bsz, device, dtype):
    """Zero retrieval branch, shaped as `build_retrieval_cache` would write it.

    Only used to reach `forward`'s base output, which does not depend on the
    retrieval branch at all -- so the values are irrelevant and the shapes are not.
    """
    slots = model.num_source_slots()
    query_dim = (
        model.relation_emb_dim
        if getattr(model.relation_mixer, 'input_mode', '') == 'retrieved_plus_query'
        else 0
    )
    return {
        'relation_outputs': torch.zeros(bsz, model.channels, slots, model.pred_len,
                                        device=device, dtype=dtype),
        'relation_query_embs': torch.zeros(bsz, model.channels, slots, query_dim,
                                           device=device, dtype=dtype),
    }


@torch.no_grad()
def base_forecast_via_forward(model, x, memory_y, memory_x_last, chunk=512):
    """Base forecast taken from `forward`'s own second output.

    Deliberately not `model.base_head(x)`: that returns a delta-space tensor and
    reproducing the offset restore outside the model is the mistake this whole
    file exists to avoid.
    """
    outputs = []
    for start in range(0, x.size(0), chunk):
        window = x[start:start + chunk]
        outputs.append(model(
            batch_x=window,
            memory_y=memory_y,
            valid_mask=torch.ones(window.size(0), memory_y.size(0),
                                  dtype=torch.bool, device=window.device),
            key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, window.size(0), window.device, window.dtype),
        )[1])
    return torch.cat(outputs)


def _pairwise_neg_mse(query, candidate):
    """-MSE between every query row and every candidate row. [Q, H] x [K, H] -> [Q, K]."""
    return -((query.unsqueeze(1) - candidate.unsqueeze(0)).square().mean(-1))


def _ranking_metrics(score, utility, valid):
    """Top-K overlap and NDCG of a score's ranking against the utility ranking."""
    width = int(valid.sum(-1).min())
    masked_score = score.masked_fill(~valid, float('-inf'))
    masked_utility = utility.masked_fill(~valid, float('-inf'))
    relevance = utility.clamp_min(0.0).masked_fill(~valid, 0.0)
    out = {}
    for k in OVERLAP_KS:
        depth = min(k, width)
        if depth <= 0:
            out[f'overlap_at_{k}'] = float('nan')
            out[f'ndcg_at_{k}'] = float('nan')
            continue
        by_score = masked_score.topk(depth, dim=-1).indices
        by_utility = masked_utility.topk(depth, dim=-1).indices
        hit = (by_score.unsqueeze(-1) == by_utility.unsqueeze(-2)).any(-1)
        out[f'overlap_at_{k}'] = float(hit.float().mean())

        discount = 1.0 / torch.log2(
            torch.arange(depth, device=score.device, dtype=torch.float64) + 2.0
        )
        dcg = (relevance.gather(1, by_score).double() * discount).sum(-1)
        idcg = (relevance.gather(1, by_utility).double() * discount).sum(-1)
        keep = idcg > 0
        out[f'ndcg_at_{k}'] = float((dcg[keep] / idcg[keep]).mean()) if keep.any() else float('nan')

    top1 = masked_score.argmax(-1, keepdim=True)
    out['utility_at_score_top1'] = float(utility.gather(1, top1).mean())
    depth = min(10, width)
    top10 = masked_score.topk(depth, dim=-1).indices if depth > 0 else top1
    out['utility_at_score_top10'] = float(utility.gather(1, top10).mean())
    return out


@torch.no_grad()
def analyse(checkpoint, pool_size=500, max_queries=512, batch_size=32,
            candidate_chunk=25, split='test'):
    experiment, args = load_stage2(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)
    experiment._build_retrieval_cache(split, loader)

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    memory_x = torch.from_numpy(experiment.memory_bank.memory_x).float().to(device)

    # A fixed, evenly spaced pool keeps every query comparable and keeps the
    # pairwise cost bounded; per-query validity is still applied on top.
    total = memory_y.size(0)
    width = min(pool_size, total)
    pool = torch.linspace(0, total - 1, width).round().long().unique().to(device)

    candidate_base = base_forecast_via_forward(
        model, memory_x.index_select(0, pool), memory_y, memory_x_last
    )
    candidate_future = memory_y.index_select(0, pool)
    candidate_residual = candidate_future - candidate_base

    collected = {'utility': [], 'future': [], 'residual': [], 'valid': [], 'base': []}
    seen = 0
    for batch_x, batch_y, batch_start_idx in loader:
        if max_queries and seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(batch_x, batch_y, batch_start_idx)
        if max_queries:
            keep = min(batch_size, max_queries - seen, batch_x.size(0))
            batch_x, batch_y, batch_start_idx = batch_x[:keep], batch_y[:keep], batch_start_idx[:keep]
        seen += batch_x.size(0)

        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        # The real cache carries this batch's query embeddings; only the
        # retrieval values get overridden, so a mixer that also consumes the
        # query still sees exactly what production would give it.
        cache = experiment._cached_retrieval_for_batch(split, batch_start_idx)
        if cache is None:
            if getattr(model.relation_mixer, 'input_mode', '') == 'retrieved_plus_query':
                raise RuntimeError(
                    'this mixer consumes query embeddings, so the diagnostic needs the '
                    'real retrieval cache rather than a zero-filled stand-in'
                )
            cache = _dummy_cache(model, batch_x.size(0), device, batch_x.dtype)
        else:
            cache = {key: value.to(device) for key, value in cache.items()
                     if torch.is_tensor(value)}
        utility, base_mse = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=pool,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last, retrieval_cache=cache,
            candidate_chunk=candidate_chunk,
        )
        query_base = base_forecast_via_forward(model, batch_x, memory_y, memory_x_last)
        query_residual = batch_y - query_base

        channels = batch_y.size(-1)
        future = torch.stack([
            _pairwise_neg_mse(batch_y[:, :, c], candidate_future[:, :, c])
            for c in range(channels)
        ], dim=-1)
        residual = torch.stack([
            _pairwise_neg_mse(query_residual[:, :, c], candidate_residual[:, :, c])
            for c in range(channels)
        ], dim=-1)
        if future.shape != utility.shape or residual.shape != utility.shape:
            raise ValueError(
                f'score/utility shape mismatch: {tuple(future.shape)} '
                f'{tuple(residual.shape)} vs {tuple(utility.shape)}'
            )
        collected['utility'].append(utility.cpu())
        collected['future'].append(future.cpu())
        collected['residual'].append(residual.cpu())
        collected['base'].append(base_mse.cpu())
        collected['valid'].append(cand_mask.index_select(1, pool).cpu())

    utility = torch.cat(collected['utility'])
    valid = torch.cat(collected['valid'])
    base_mse = torch.cat(collected['base'])
    queries, pool_width, channels = utility.shape
    # Flatten (query, channel) into independent ranking problems; the validity
    # mask is per query, so it is repeated across channels rather than reshaped.
    flat_valid = valid.unsqueeze(-1).expand_as(utility).permute(0, 2, 1).reshape(-1, pool_width)
    flat_utility = utility.permute(0, 2, 1).reshape(-1, pool_width)

    rows = []
    for name in ('future', 'residual'):
        flat_score = torch.cat(collected[name]).permute(0, 2, 1).reshape(-1, pool_width)
        pearson, spearman = rank_correlations(flat_score, flat_utility, flat_valid)
        row = {
            'dataset': args.data, 'pred_len': int(args.pred_len), 'score': name,
            'queries': queries, 'channels': channels, 'pool_size': pool_width,
            'pearson': pearson, 'spearman': spearman,
            'best_utility': float(flat_utility.masked_fill(~flat_valid, float('-inf')).max(-1).values.mean()),
            'mean_utility': float(flat_utility[flat_valid].mean()),
            'positive_rate': float((flat_utility[flat_valid] > 0).float().mean()),
            'base_mse': float(base_mse.mean()),
            'checkpoint': checkpoint,
        }
        row.update(_ranking_metrics(flat_score, flat_utility, flat_valid))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pool_size', type=int, default=500)
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--candidate_chunk', type=int, default=25)
    parser.add_argument('--split', default='test')
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    for row in analyse(args.checkpoint, args.pool_size, args.max_queries,
                       candidate_chunk=args.candidate_chunk, split=args.split):
        print(f"=== {row['dataset']}/{row['pred_len']} score={row['score']} ===")
        for key in COLUMNS:
            if key == 'checkpoint':
                continue
            value = row[key]
            print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
        if args.csv:
            append_row(args.csv, row, COLUMNS)
    if args.csv:
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
