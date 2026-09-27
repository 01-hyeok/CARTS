#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN03, Stage D0 -- teacher sharpness
diagnostic (spec section 10). Reuses `block_distance`/`BLOCKS` from
`scripts.train_c_horizon_clean02` (== `individual_utility_memsafe` on a
sliced memory_c/query_future, unmodified) and `normalized_teacher_prob`
from `scripts.train_horizon_retrieval_expert01`, unmodified. No model, no
gradient -- pure oracle-distance diagnostic on a fixed train subset. Does
NOT auto-select a temperature (unlike Clean02's calibration script) --
this script only REPORTS sharpness metrics across a fixed tau grid so the
A1/A2/A3 arms can be compared against real Top-K retrieval behavior later.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_c_horizon_clean02 import BLOCKS, block_distance
from scripts.train_horizon_retrieval_expert01 import BLOCK_NAMES, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value

TAUS = (0.01, 0.02, 0.05, 0.10)


def _fixed_train_subset(exp, n_queries, seed=0):
    _, loader = exp._get_data(flag='train', shuffle=False)
    xs, ys, starts = [], [], []
    for bx, by, bstart in loader:
        xs.append(bx); ys.append(by)
        starts.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
    x_all, y_all, start_all = torch.cat(xs), torch.cat(ys), torch.cat(starts)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_all.size(0), generator=g)[:min(n_queries, x_all.size(0))]
    idx, _ = idx.sort()
    return x_all[idx], y_all[idx], start_all[idx]


def sharpness_stats(d, valid_mask, tau):
    p_t = normalized_teacher_prob(d, valid_mask, tau)
    n_valid = valid_mask.sum(-1).clamp_min(2).float()
    ent = -(p_t.masked_fill(~valid_mask, 0.0) * torch.log(p_t.clamp_min(1e-8))).sum(-1)
    norm_ent = ent / torch.log(n_valid)
    eff_pos = 1.0 / (p_t.square().sum(-1).clamp_min(1e-8))
    top1 = p_t.max(dim=-1).values
    top10 = p_t.topk(min(10, p_t.size(-1)), dim=-1).values.sum(-1)
    top30 = p_t.topk(min(30, p_t.size(-1)), dim=-1).values.sum(-1)
    return {'raw_entropy': ent, 'normalized_entropy': norm_ent, 'effective_positives': eff_pos,
           'top1_mass': top1, 'top10_mass': top10, 'top30_mass': top30}


def summarize(x):
    q = torch.quantile(x, torch.tensor([0.10, 0.25, 0.5, 0.75, 0.90]))
    return {'mean': float(x.mean()), 'median': float(q[2]), 'p10': float(q[0]), 'p25': float(q[1]),
           'p75': float(q[3]), 'p90': float(q[4])}


def jaccard(a, b, k):
    inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1)
    union = float(k) * 2 - inter
    return inter / union.clamp_min(1e-9)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--n_queries', type=int, default=512)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN03')
    cli = ap.parse_args()

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
    })
    exp._ensure_memory()
    device = exp.device
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    x_sub, y_sub, start_sub = _fixed_train_subset(exp, cli.n_queries, cli.subset_seed)
    n = x_sub.size(0)

    per_tau_stats = {tau: {name: {k: [] for k in
                                   ('raw_entropy', 'normalized_entropy', 'effective_positives',
                                    'top1_mass', 'top10_mass', 'top30_mass')}
                          for name in ('global', *BLOCK_NAMES)}
                    for tau in TAUS}
    picks_global_top10, picks_block_top10 = [], {b: [] for b in BLOCK_NAMES}

    for s in range(0, n, cli.batch_size):
        bx = x_sub[s:s + cli.batch_size].float().to(device)
        by = y_sub[s:s + cli.batch_size].float().to(device)
        bstart = start_sub[s:s + cli.batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
            query_future = by[:, :, c]
            d_g = block_distance(memory_c, offset_c, query_future, 0, 720, cli.chunk_size)
            d_g = d_g.masked_fill(~cand_mask, float('inf'))
            picks_global_top10.append(stable_topk_indices(d_g, cli.top_k, largest=False).cpu())
            for tau in TAUS:
                st = sharpness_stats(d_g, cand_mask, tau)
                for k, v in st.items():
                    per_tau_stats[tau]['global'][k].append(v.cpu())
            for bname in BLOCK_NAMES:
                lo, hi = BLOCKS[bname]
                d_b = block_distance(memory_c, offset_c, query_future, lo, hi, cli.chunk_size)
                d_b = d_b.masked_fill(~cand_mask, float('inf'))
                picks_block_top10[bname].append(stable_topk_indices(d_b, cli.top_k, largest=False).cpu())
                for tau in TAUS:
                    st = sharpness_stats(d_b, cand_mask, tau)
                    for k, v in st.items():
                        per_tau_stats[tau][bname][k].append(v.cpu())

    result = {'cell': cli.cell, 'n_queries': n, 'n_channels': len(channels), 'taus': list(TAUS),
             'per_tau': {}}
    for tau in TAUS:
        result['per_tau'][tau] = {}
        for name in ('global', *BLOCK_NAMES):
            result['per_tau'][tau][name] = {k: summarize(torch.cat(v))
                                           for k, v in per_tau_stats[tau][name].items()}

    pg = torch.cat(picks_global_top10)
    pb = {b: torch.cat(v) for b, v in picks_block_top10.items()}
    oracle_jaccard = {}
    for a, b in itertools.combinations(BLOCK_NAMES, 2):
        oracle_jaccard[f'{a}_vs_{b}'] = float(jaccard(pb[a], pb[b], cli.top_k).mean())
    for b in BLOCK_NAMES:
        oracle_jaccard[f'global_vs_{b}'] = float(jaccard(pg, pb[b], cli.top_k).mean())
    result['oracle_top10_jaccard'] = oracle_jaccard
    result['note'] = ('Teacher RANKING itself does not change with tau (same distances every time) -- '
                      'only probability CONCENTRATION changes. oracle_top10_jaccard is tau-independent '
                      'by construction (computed from raw distances, not from any softmax).')

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'teacher_sharpness_diagnostic.json').write_text(json.dumps(result, indent=2))
    print(f'[diag_frozen03_teacher] {cli.cell}: oracle_top10_jaccard={ {k: round(v,4) for k,v in oracle_jaccard.items()} }')
    for tau in TAUS:
        row = {name: round(result['per_tau'][tau][name]['top10_mass']['mean'], 4)
              for name in ('global', *BLOCK_NAMES)}
        print(f'  tau={tau}: top10_mass_mean(global,B1,B2,B3)={row}')
    print(f'[diag_frozen03_teacher] wrote {out_dir / "teacher_sharpness_diagnostic.json"}')


if __name__ == '__main__':
    main()
