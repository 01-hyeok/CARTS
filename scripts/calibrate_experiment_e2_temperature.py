#!/usr/bin/env python3
"""Experiment E, E2 -- per-component teacher temperature pre-calibration.

Generalizes `calibrate_patch_retrieval_expert01_temperature.py`'s rule
(among a fixed candidate set, pick the tau_T whose Top-10 teacher-probability
-mass MEDIAN over sampled TRAIN rows is closest to 0.5, ties -> larger tau_T)
to the four E1 component distances (`diag_experiment_e_teacher01.transform`
via `component_distance_memsafe`), since each component transform has its
own natural scale (shared=full MSE, local/seasonal=difference MSE,
trend=moving-average MSE) and therefore needs its own temperature -- a
single dataset-wide tau_T (as in the single-teacher D-track) would not be
valid across all four.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_experiment_e_teacher01 import COMPONENTS, component_distance_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value

CANDIDATE_TAUS = (0.1, 0.05, 0.02, 0.01, 0.005)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--n_queries', type=int, default=512)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--out_dir', default='results/EXPERIMENT-E-E2')
    cli = ap.parse_args()

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
    })
    exp._ensure_memory()
    device = exp.device
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    x_sub, y_sub, start_sub = _fixed_train_subset(exp, cli.n_queries, cli.subset_seed)
    n = x_sub.size(0)

    per_comp_tau = {comp: {tau: {'top10_mass': []} for tau in CANDIDATE_TAUS} for comp in COMPONENTS}
    for s in range(0, n, cli.batch_size):
        bx = x_sub[s:s + cli.batch_size].float().to(device)
        by = y_sub[s:s + cli.batch_size].float().to(device)
        bstart = start_sub[s:s + cli.batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
            query_future = by[:, :, c]
            for comp in COMPONENTS:
                d = component_distance_memsafe(memory_c, offset_c, query_future, comp, period,
                                               cli.chunk_size).masked_fill(~cand_mask, float('inf'))
                for tau in CANDIDATE_TAUS:
                    p_t = normalized_teacher_prob(d, cand_mask, tau)
                    top10_mass = p_t.topk(10, dim=-1).values.sum(-1)
                    per_comp_tau[comp][tau]['top10_mass'].append(top10_mass.cpu())

    result = {'cell': cli.cell, 'n_queries': n, 'n_channels': len(channels), 'period': period,
             'candidate_taus': list(CANDIDATE_TAUS), 'per_component': {}, 'selected_tau_t': {}}
    for comp in COMPONENTS:
        rows = {}
        for tau in CANDIDATE_TAUS:
            mass = torch.cat(per_comp_tau[comp][tau]['top10_mass'])
            rows[tau] = {'top10_mass_median': float(mass.median()), 'top10_mass_mean': float(mass.mean()),
                        'n_rows': int(mass.numel())}
        best_tau = min(CANDIDATE_TAUS, key=lambda t: (abs(rows[t]['top10_mass_median'] - 0.5), -t))
        result['per_component'][comp] = rows
        result['selected_tau_t'][comp] = best_tau
        print(f"[calibrate_e2_temp] {cli.cell}/{comp}: selected_tau_t={best_tau} "
             f"(medians: {[(t, round(rows[t]['top10_mass_median'], 4)) for t in CANDIDATE_TAUS]})")

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'temperature_calibration.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
