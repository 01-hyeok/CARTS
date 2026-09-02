#!/usr/bin/env python3
"""EXPERIMENT 1 -- does Top-K aggregation destroy the gain of better candidates?

The teacher ablation improved every Stage-1 utility metric and improved no
forecast. Stage-2 does not consume one candidate, it consumes a Top-K weighted
mixture, so the obvious suspect is that individually useful candidates stop being
useful together.

Two quantities per query, both measured through the production path:

    individual  U(q, k)      one candidate injected alone
    set         U(q, S_K)    the model's own Top-K, its own weights

The set number needs no injection at all. Setting `top_k = K` and running the
real forward *is* the model selecting S_K with its own scores and aggregating it
with its own softmax -- so there is no hand-built average anywhere, which is the
whole point.

Every method is evaluated on one fixed reference Stage-2, with only the Stage-1
encoder swapped in. That holds the base head, mixer and gate constant so a
difference between methods is a difference in retrieval, not in the forecaster.
Whether that reference matters is Experiment 3's question, not this one.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_ops import retrieve_relation_future  # noqa: E402
from utils.retrieval_diagnostics import append_row, rank_correlations, unwrap  # noqa: E402
from utils.utility_teacher import _dummy_cache, load_stage2_reference  # noqa: E402

KS = (1, 3, 5, 10)
QUERY_COLUMNS = ['dataset', 'pred_len', 'method', 'top_k', 'query', 'channel',
                 'individual_mean', 'individual_sum', 'individual_max',
                 'positive_fraction', 'set_utility', 'base_mse', 'set_mse']
SUMMARY_COLUMNS = [
    'dataset', 'pred_len', 'method', 'top_k', 'queries', 'channels',
    'individual_utility_at_k', 'individual_sum', 'individual_max',
    'positive_rate_at_k', 'set_utility', 'set_utility_positive_rate',
    'base_mse', 'set_mse',
    'interaction_loss', 'interaction_ratio',
    'pearson_mean_vs_set', 'spearman_mean_vs_set',
    'pearson_sum_vs_set', 'spearman_sum_vs_set',
    'pearson_max_vs_set', 'spearman_max_vs_set',
    'pearson_positive_vs_set', 'spearman_positive_vs_set',
    'stage1_checkpoint', 'reference_checkpoint',
]
EPS = 1e-8


@torch.no_grad()
def selected_candidates(model, key_bank, batch_x, valid_mask, top_k):
    """The Top-K ids the production retrieval op would pick, per target channel.

    Uses `retrieve_relation_future` rather than a reimplemented argmax so the
    tie-breaking, masking and similarity all stay the canonical ones.
    """
    picks = []
    for c in model.target_channels():
        sources = model.source_channels(c)
        slot = sources.index(c)
        z_q = model._branch_embedding(batch_x, c, c)
        z_mem = model._branch_memory(key_bank, c, slot, c, z_q.dtype, batch_x.device)
        _, _, top_idx, _, _ = retrieve_relation_future(
            z_q=z_q, z_mem=z_mem,
            memory_value_c=torch.zeros(z_mem.size(0), model.pred_len,
                                       device=batch_x.device, dtype=batch_x.dtype),
            valid_mask=valid_mask, top_k=top_k,
            tau_topk=model.tau_topk, similarity=model.retrieval_similarity,
            soft_all=model.retrieval_soft_all,
        )
        picks.append(top_idx)
    return torch.stack(picks, dim=1)


@torch.no_grad()
def analyse(reference, stage1_path, method, ks=KS, max_queries=512,
            candidate_chunk=10, split='test'):
    experiment, args = load_stage2_reference(reference, stage1_path)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    original_top_k = model.top_k

    collected = {k: {name: [] for name in
                     ('mean', 'sum', 'max', 'positive', 'set', 'base', 'set_mse')}
                 for k in ks}
    seen = 0
    try:
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

            for k in ks:
                pool = selected_candidates(model, experiment.key_bank, batch_x, cand_mask, k)
                utility, base_mse = model.evaluate_candidate_correction(
                    batch_x=batch_x, batch_y=batch_y, candidate_indices=pool,
                    memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
                    memory_x_last=memory_x_last,
                    retrieval_cache=_dummy_cache(model, batch_x.size(0), device, batch_x.dtype),
                    candidate_chunk=candidate_chunk,
                )
                utility = utility.permute(0, 2, 1)          # [B, C, K]

                # The set number: the real model, its own Top-K, its own weights.
                model.top_k = k
                y_final = experiment.model(
                    batch_x=batch_x, memory_y=memory_y, valid_mask=cand_mask,
                    key_bank=experiment.key_bank, memory_x_last=experiment.memory_x_last,
                )[0]
                if y_final.shape != batch_y.shape:
                    raise ValueError(
                        f'forecast shape {tuple(y_final.shape)} != {tuple(batch_y.shape)}')
                set_mse = (y_final - batch_y).square().mean(dim=1)   # [B, C]

                collected[k]['mean'].append(utility.mean(-1).cpu())
                collected[k]['sum'].append(utility.sum(-1).cpu())
                collected[k]['max'].append(utility.max(-1).values.cpu())
                collected[k]['positive'].append((utility > 0).float().mean(-1).cpu())
                collected[k]['set'].append((base_mse - set_mse).cpu())
                collected[k]['base'].append(base_mse.cpu())
                collected[k]['set_mse'].append(set_mse.cpu())
    finally:
        model.top_k = original_top_k

    rows, per_query = [], []
    for k in ks:
        part = {name: torch.cat(value) for name, value in collected[k].items()}
        # Each (query, channel) is one observation; flattening keeps them separate
        # rather than averaging a channel's behaviour away.
        flat = {name: value.reshape(1, -1) for name, value in part.items()}
        valid = torch.ones_like(flat['set'], dtype=torch.bool)
        row = {
            'dataset': args.data, 'pred_len': int(args.pred_len), 'method': method,
            'top_k': k, 'queries': part['set'].size(0), 'channels': part['set'].size(1),
            'individual_utility_at_k': float(part['mean'].mean()),
            'individual_sum': float(part['sum'].mean()),
            'individual_max': float(part['max'].mean()),
            'positive_rate_at_k': float(part['positive'].mean()),
            'set_utility': float(part['set'].mean()),
            'set_utility_positive_rate': float((part['set'] > 0).float().mean()),
            'base_mse': float(part['base'].mean()),
            'set_mse': float(part['set_mse'].mean()),
            'interaction_loss': float((part['sum'] - part['set']).mean()),
            'interaction_ratio': float(
                (part['set'].sum() / (part['sum'].sum().abs() + EPS))),
            'stage1_checkpoint': stage1_path or reference,
            'reference_checkpoint': reference,
        }
        for name in ('mean', 'sum', 'max', 'positive'):
            pearson, spearman = rank_correlations(flat[name], flat['set'], valid)
            label = {'mean': 'mean', 'sum': 'sum', 'max': 'max', 'positive': 'positive'}[name]
            row[f'pearson_{label}_vs_set'] = pearson
            row[f'spearman_{label}_vs_set'] = spearman
        rows.append(row)

        for query in range(part['set'].size(0)):
            for channel in range(part['set'].size(1)):
                per_query.append({
                    'dataset': args.data, 'pred_len': int(args.pred_len),
                    'method': method, 'top_k': k, 'query': query, 'channel': channel,
                    'individual_mean': float(part['mean'][query, channel]),
                    'individual_sum': float(part['sum'][query, channel]),
                    'individual_max': float(part['max'][query, channel]),
                    'positive_fraction': float(part['positive'][query, channel]),
                    'set_utility': float(part['set'][query, channel]),
                    'base_mse': float(part['base'][query, channel]),
                    'set_mse': float(part['set_mse'][query, channel]),
                })
    return rows, per_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True,
                        help='Stage-2 checkpoint providing the fixed forecaster')
    parser.add_argument('--method', required=True, nargs='+',
                        help='name=stage1_checkpoint, or name= to keep the reference encoder')
    parser.add_argument('--ks', default='1,3,5,10')
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--candidate_chunk', type=int, default=10)
    parser.add_argument('--out_dir', default='./metrics/retrieval_bottleneck')
    args = parser.parse_args()

    ks = tuple(int(value) for value in args.ks.split(','))
    out_dir = Path(args.out_dir)
    for spec in args.method:
        name, _, path = spec.partition('=')
        rows, per_query = analyse(args.reference, path, name, ks=ks,
                                  max_queries=args.max_queries,
                                  candidate_chunk=args.candidate_chunk)
        for row in rows:
            append_row(out_dir / 'set_level_summary.csv', row, SUMMARY_COLUMNS)
            print(f"{row['dataset']}/{row['pred_len']} {name:<22} K={row['top_k']:<3} "
                  f"indiv={row['individual_utility_at_k']:+.4f} "
                  f"set={row['set_utility']:+.4f} "
                  f"pos={row['positive_rate_at_k']:.3f} "
                  f"inter_loss={row['interaction_loss']:+.4f} "
                  f"rho(mean,set)={row['spearman_mean_vs_set']:+.3f}")
        for row in per_query:
            append_row(out_dir / 'set_level_query.csv', row, QUERY_COLUMNS)
    print(f'wrote {out_dir}/set_level_summary.csv and set_level_query.csv')


if __name__ == '__main__':
    main()
