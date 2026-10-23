#!/usr/bin/env python3
"""TRACK-SEMANTIC-SPECIALIZED-HEADS01 -- PHASE A: semantic oracle
complementarity diagnostic. No model, no training. Reuses
`scripts.diag_experiment_e_teacher01.transform` (Experiment E's own
shared/local/trend/seasonal component-distance definitions) UNMODIFIED
-- restricted here to the Shared-P100 candidate pool (Experiment E's
own script operates on the full candidate bank; this track's whole
premise is P100 comparability with every other track this session).
P (seasonal period) is read from the EXISTING train-only
`period_detection.json` for each dataset -- confirmed reusable across
horizons for the same dataset: `data_provider/data_loader.py`'s own
train-split boundary (`border1s[0]=0`, `border2s[0]`=a fixed constant)
does not depend on seq_len/pred_len, so the train-only ACF period for
ETTh1/Weather is identical regardless of which horizon cell is being
diagnosed -- no new period computation needed.

VALIDATION split only (never test; this diagnostic's output decides
whether Phase B happens at all, so keeping it off test matches this
track's own later rule that Phase B's role assignment must be
train/validation-only, not test-informed).
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_experiment_e_teacher01 import transform
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import gather_candidate_values

COMPONENTS = ('shared', 'local', 'trend', 'seasonal')
FIXED_COMBOS = {
    'G': ('shared',), 'L': ('local',), 'T': ('trend',), 'S': ('seasonal',),
    'GT': ('shared', 'trend'), 'GL': ('shared', 'local'), 'GS': ('shared', 'seasonal'),
    'GTL': ('shared', 'trend', 'local'), 'GTLS': ('shared', 'trend', 'local', 'seasonal'),
}
LABEL = {'shared': 'G', 'local': 'L', 'trend': 'T', 'seasonal': 'S'}


@torch.no_grad()
def component_topk(pooled_future_abs, query_future_abs, component, period, top_k):
    """pooled_future_abs: [B, M, H] absolute candidate futures (P100-restricted).
    query_future_abs: [B, H]. Returns (picks_local [B,top_k], d [B,M])."""
    q_t = transform(query_future_abs, component, period)      # [B, H']
    c_t = transform(pooled_future_abs, component, period)      # [B, M, H']
    d = ((c_t - q_t.unsqueeze(1)) ** 2).mean(dim=-1)           # [B, M]
    picks = d.topk(top_k, dim=-1, largest=False).indices
    return picks, d


@torch.no_grad()
def run_cell(cli, device):
    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size, 'seed': 0,
        'candidate_mask': cli.candidate_mask,
    })
    exp._ensure_memory()
    channels = list(range(int(args.enc_in)))

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt', expected_meta)

    _, loader = exp._get_data(flag='val', shuffle=False)

    overlap_inter = {f'{a}_vs_{b}': [] for a, b in combinations(COMPONENTS, 2)}
    combo_agg = {tag: [] for tag in FIXED_COMBOS}
    per_query_best = []  # best fixed-combo raw AggMSE per query/channel, for restricted-oracle headroom
    per_query_vals = []  # dict of combo->agg_mse per query/channel (for winner-share)
    union_sizes = {tag: [] for tag in ('GT', 'GL', 'GS', 'GTL', 'GTLS')}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]  # absolute
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)
            pooled_future = gather_candidate_values(memory_c, pool_idx_global) + offset_c.view(-1, 1, 1)

            picks, agg_pred = {}, {}
            for comp in COMPONENTS:
                p, _ = component_topk(pooled_future, query_future, comp, period, cli.top_k)
                picks[comp] = p
                y_sel = pooled_future.gather(1, p.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
                agg_pred[comp] = y_sel.mean(dim=1)  # R_comp, raw-future space, [B, H]

            for a, b in combinations(COMPONENTS, 2):
                pa, pb = picks[a], picks[b]
                inter = (pa.unsqueeze(-1) == pb.unsqueeze(-2)).any(-1).float().sum(-1)
                overlap_inter[f'{a}_vs_{b}'].append((inter / cli.top_k).cpu())

            for tag, comps in FIXED_COMBOS.items():
                r = torch.stack([agg_pred[c] for c in comps], dim=0).mean(dim=0)
                agg_mse = ((r - query_future) ** 2).mean(dim=-1)
                combo_agg[tag].append(agg_mse.cpu())

            label_map = {LABEL[c]: picks[c] for c in COMPONENTS}
            for tag in union_sizes:
                comps = [label_map[ch] for ch in tag if ch in label_map]
                all_idx = torch.cat(comps, dim=1)
                usize = torch.tensor([len(set(all_idx[b].tolist())) for b in range(all_idx.size(0))])
                union_sizes[tag].append(usize.float())

            bsz = batch_x.size(0)
            vals_this_batch = {tag: combo_agg[tag][-1] for tag in FIXED_COMBOS}
            for b in range(bsz):
                row = {tag: float(vals_this_batch[tag][b]) for tag in FIXED_COMBOS}
                per_query_vals.append(row)
                per_query_best.append(min(row.values()))

    overlaps = {k: {'mean': float(torch.cat(v).mean()), 'std': float(torch.cat(v).std()),
                   'median': float(torch.cat(v).median()), 'p25': float(torch.cat(v).quantile(0.25)),
                   'p75': float(torch.cat(v).quantile(0.75))}
               for k, v in overlap_inter.items()}
    combo_means = {tag: float(torch.cat(v).mean()) for tag, v in combo_agg.items()}
    union_size_means = {tag: float(torch.cat(v).mean()) for tag, v in union_sizes.items()}

    winner_counts = {tag: 0 for tag in FIXED_COMBOS}
    for row in per_query_vals:
        best_tag = min(row, key=row.get)
        winner_counts[best_tag] += 1
    n_rows = len(per_query_vals)
    winner_share = {tag: winner_counts[tag] / n_rows for tag in FIXED_COMBOS}

    restricted_oracle_mean = sum(per_query_best) / n_rows
    best_fixed_tag = min(combo_means, key=combo_means.get)

    out = {
        'cell': cli.cell, 'period': period, 'n_rows': n_rows,
        'pairwise_top10_overlap': overlaps,
        'fixed_combo_agg_mse': combo_means,
        'best_fixed_combo': best_fixed_tag, 'best_fixed_agg_mse': combo_means[best_fixed_tag],
        'restricted_oracle_agg_mse': restricted_oracle_mean,
        'oracle_gain_pct_vs_best_fixed': (combo_means[best_fixed_tag] - restricted_oracle_mean)
                                        / combo_means[best_fixed_tag] * 100.0,
        'oracle_gain_pct_vs_global': (combo_means['G'] - restricted_oracle_mean) / combo_means['G'] * 100.0,
        'winner_share': winner_share,
        'union_size_mean': union_size_means,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[audit] cell={cli.cell} period_json={cli.period_json}')
    out = run_cell(cli, device)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseA_oracle.json').write_text(json.dumps(out, indent=2))
    print(f'[diag_semantic_oracle] {cli.cell}: period={out["period"]} '
         f'fixed_combo_agg_mse={ {k: round(v,4) for k,v in out["fixed_combo_agg_mse"].items()} } '
         f'best_fixed={out["best_fixed_combo"]} oracle_gain_vs_global={out["oracle_gain_pct_vs_global"]:.2f}%')


if __name__ == '__main__':
    main()
