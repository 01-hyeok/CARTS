#!/usr/bin/env python3
"""Experiment E, Stage E1: component-teacher validity diagnostic.

No neural encoder is trained here. For every query/channel, computes four
ORACLE distances between the query's real future and every valid
candidate's real future (memory-safe, chunked over candidates -- same
chunking convention as `train_factorial_e2e01.py::individual_utility_
memsafe`, generalized to accept a transform applied to both sides before
squared error):

  shared:   MSE(y_q, y_k)                        (existing production teacher)
  local:    MSE(diff(y_q), diff(y_k))             (1st difference, length H-1)
  trend:    MSE(MA_P(y_q), MA_P(y_k))             (valid moving average, length H-P+1)
  seasonal: MSE(y_q[P:]-y_q[:-P], y_k[P:]-y_k[:-P]) (seasonal difference, length H-P)

P (seasonal period) comes from `period_detection.json` (train-only ACF,
already computed and saved separately -- never re-derived from val/test).

Reuses `exp._candidate_mask` (same candidate validity/no-leak mask as every
other Track-A script this session) and the SAME delta-space
`memory_c`/`offset_c` convention (`scripts.train_margutil01.memory_value`)
for absolute-future reconstruction. No host encoder, no learned model.
"""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_margutil01 import build_experiment, memory_value

COMPONENTS = ('shared', 'local', 'trend', 'seasonal')


def transform(y, component, period):
    """y: [..., H] absolute values (query: [B,H], candidates: [B,C,H] or [C,H])."""
    if component == 'shared':
        return y
    if component == 'local':
        return y[..., 1:] - y[..., :-1]
    if component == 'trend':
        # valid moving average, window=period
        H = y.size(-1)
        csum = torch.cumsum(y, dim=-1)
        csum = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ma = (csum[..., period:] - csum[..., :-period]) / period
        return ma  # length H-period+1
    if component == 'seasonal':
        return y[..., period:] - y[..., :-period]
    raise ValueError(component)


@torch.no_grad()
def component_distance_memsafe(memory_c, offset_c, query_future, component, period, chunk_size):
    """Generalization of individual_utility_memsafe: -utility == MSE distance,
    transform applied to both query and (chunked, absolute-reconstructed)
    candidate futures before squared error. Returns d [B, N] (lower=closer)."""
    bsz = offset_c.size(0)
    n, h = memory_c.shape
    chunk_size = chunk_size or n
    q_t = transform(query_future, component, period)  # [B, H']
    out = offset_c.new_empty(bsz, n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y_c = memory_c[start:end].unsqueeze(0) + offset_c.view(-1, 1, 1)  # [B, c, H]
        y_c_t = transform(y_c, component, period)  # [B, c, H']
        out[:, start:end] = ((y_c_t - q_t.unsqueeze(1)) ** 2).mean(dim=-1)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--out_dir', default='outputs/experiment_E/diagnostics')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_queries', type=int, default=0, help='0 = all')
    cli = ap.parse_args()

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.split
    out_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
    })
    exp._ensure_memory()
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    _, loader = exp._get_data(flag=cli.split, shuffle=False)
    n_seen = 0
    per_comp_topk_picks = {e: [] for e in COMPONENTS}
    per_comp_specialization_sum = {e: {e2: 0.0 for e2 in COMPONENTS} for e in COMPONENTS}
    per_comp_winner_rows = []  # (start, channel, {comp: overall_agg_mse})
    starts_all = []

    for batch_x, batch_y, batch_start_idx in loader:
        if cli.max_queries and n_seen >= cli.max_queries:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        starts_all.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                          else torch.as_tensor(batch_start_idx))
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            d = {}
            for comp in COMPONENTS:
                d[comp] = component_distance_memsafe(memory_c, offset_c, query_future, comp, period,
                                                      cli.chunk_size).masked_fill(~cand_mask, float('inf'))
            picks = {comp: stable_topk_indices(d[comp], cli.top_k, largest=False) for comp in COMPONENTS}
            for comp in COMPONENTS:
                per_comp_topk_picks[comp].append(picks[comp].cpu())

            # cross-component specialization: each teacher's Top-K picks, scored
            # under every OTHER component's own distance metric (mean over picks)
            for e in COMPONENTS:
                for e2 in COMPONENTS:
                    val = d[e2].gather(1, picks[e]).mean(-1).sum()
                    per_comp_specialization_sum[e][e2] += float(val)

            # per-query overall winner: aggregate (uniform mean) of each
            # component's Top-K picks' ABSOLUTE future, scored by OVERALL (shared) MSE
            for comp in COMPONENTS:
                y_sel = (memory_c[picks[comp]] + offset_c.view(-1, 1, 1))  # [B, K, H]
                agg = y_sel.mean(dim=1)
                overall_mse = ((agg - query_future) ** 2).mean(dim=-1)
                for b in range(bsz):
                    per_comp_winner_rows.append((int(batch_start_idx[b]), c, comp, float(overall_mse[b])))
        n_seen += bsz

    # ---- pairwise overlap (Jaccard @ top_k) ----
    overlaps = {}
    for e, e2 in combinations(COMPONENTS, 2):
        pa = torch.cat(per_comp_topk_picks[e])
        pb = torch.cat(per_comp_topk_picks[e2])
        inter = (pa.unsqueeze(-1) == pb.unsqueeze(-2)).any(-1).float().sum(-1)
        union = float(cli.top_k) * 2 - inter
        jaccard = inter / union.clamp_min(1e-9)
        overlaps[f'{e}_vs_{e2}'] = {'mean_jaccard': float(jaccard.mean()), 'median_jaccard': float(jaccard.median())}

    # ---- cross-component specialization matrix (mean distance) ----
    denom = n_seen * len(channels)
    spec_matrix = {e: {e2: per_comp_specialization_sum[e][e2] / denom for e2 in COMPONENTS} for e in COMPONENTS}

    # ---- winner-rate summary from per-query rows ----
    from collections import defaultdict
    by_query = defaultdict(dict)
    for start, c, comp, val in per_comp_winner_rows:
        by_query[(start, c)][comp] = val
    winner_counts = {e: 0 for e in COMPONENTS}
    comp_means = {e: 0.0 for e in COMPONENTS}
    n_rows = len(by_query)
    for key, vals in by_query.items():
        best = min(vals, key=vals.get)
        winner_counts[best] += 1
        for e in COMPONENTS:
            comp_means[e] += vals[e]
    winner_share = {e: winner_counts[e] / max(n_rows, 1) for e in COMPONENTS}
    comp_means = {e: comp_means[e] / max(n_rows, 1) for e in COMPONENTS}
    best_fixed = min(comp_means, key=comp_means.get)
    oracle_mean = sum(min(vals.values()) for vals in by_query.values()) / max(n_rows, 1)

    with open(out_dir / 'per_query_component_metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_start_idx', 'channel'] + [f'overall_agg_mse_{e}' for e in COMPONENTS] + ['winner'])
        for (start, c), vals in by_query.items():
            best = min(vals, key=vals.get)
            w.writerow([start, c] + [vals[e] for e in COMPONENTS] + [best])

    summary = {
        'cell': cli.cell, 'split': cli.split, 'period': period, 'top_k': cli.top_k,
        'n_queries': n_seen, 'n_channels': len(channels), 'n_rows': n_rows,
        'pairwise_top10_jaccard': overlaps,
        'cross_component_specialization_matrix_mean_distance': spec_matrix,
        'component_mean_overall_agg_mse': comp_means,
        'winner_share': winner_share,
        'best_fixed_component': best_fixed, 'best_fixed_mean': comp_means[best_fixed],
        'scale_oracle_mean': oracle_mean,
        'oracle_gain_pct': (comp_means[best_fixed] - oracle_mean) / comp_means[best_fixed] * 100.0,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"[experiment_E_teacher] {cli.cell}/{cli.split}: period={period} "
         f"comp_means={ {k: round(v,4) for k,v in comp_means.items()} } "
         f"winner_share={ {k: round(v,3) for k,v in winner_share.items()} } "
         f"best_fixed={best_fixed} oracle_gain={summary['oracle_gain_pct']:.2f}%")
    print(f"  pairwise_jaccard(top{cli.top_k})={ {k: round(v['mean_jaccard'],3) for k,v in overlaps.items()} }")


if __name__ == '__main__':
    main()
