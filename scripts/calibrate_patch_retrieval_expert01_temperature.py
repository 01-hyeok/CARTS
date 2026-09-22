#!/usr/bin/env python3
"""TRACK-A-PATCH-RETRIEVAL-EXPERT01 section 2 -- dataset-level teacher
temperature pre-calibration, decoupled from patch size and from any model.

The teacher `p_T(i|q) = softmax(-zscore(d(q,i))/tau_T)` depends only on the
candidate individual future-MSE `d` and the candidate validity mask -- never
on any encoder, so this can (and must, per spec) be measured once per
dataset on TRAIN queries only, before any patch-size model is trained, using
a fixed pre-specified rule (not an optimality search): among
tau_T in {0.1, 0.05, 0.02}, pick the one whose Top-10 teacher-probability-
mass MEDIAN (over the sampled train query/channel rows) is closest to 0.5;
ties broken toward the larger tau_T. The chosen tau_T is then applied to
every patch-size arm trained on that dataset (spec section 2).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value

CANDIDATE_TAUS = (0.1, 0.05, 0.02)


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
    ap.add_argument('--reference_ckpt', required=True,
                    help='any patch_len is fine -- the teacher never touches the encoder')
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--n_queries', type=int, default=512)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/TRACK-A-PATCH-RETRIEVAL-EXPERT01')
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

    per_tau = {tau: {'top10_mass': [], 'n_valid': []} for tau in CANDIDATE_TAUS}
    for s in range(0, n, cli.batch_size):
        bx = x_sub[s:s + cli.batch_size].float().to(device)
        by = y_sub[s:s + cli.batch_size].float().to(device)
        bstart = start_sub[s:s + cli.batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
            query_future = by[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            for tau in CANDIDATE_TAUS:
                p_t = normalized_teacher_prob(d, cand_mask, tau)
                top10_mass = p_t.topk(10, dim=-1).values.sum(-1)
                n_valid = cand_mask.sum(-1).float()
                per_tau[tau]['top10_mass'].append(top10_mass.cpu())
                per_tau[tau]['n_valid'].append(n_valid.cpu())

    rows = {}
    for tau in CANDIDATE_TAUS:
        mass = torch.cat(per_tau[tau]['top10_mass'])
        nv = torch.cat(per_tau[tau]['n_valid'])
        rows[tau] = {
            'tau_t': tau,
            'top10_mass_median': float(mass.median()),
            'top10_mass_mean': float(mass.mean()),
            'n_valid_median': float(nv.median()),
            'n_rows': int(mass.numel()),
        }

    # pre-specified rule: closest median top10_mass to 0.5, ties -> larger tau_T
    best_tau = min(CANDIDATE_TAUS,
                   key=lambda t: (abs(rows[t]['top10_mass_median'] - 0.5), -t))

    result = {
        'cell': cli.cell, 'n_queries': n, 'n_channels': len(channels),
        'candidate_taus': list(CANDIDATE_TAUS), 'per_tau': rows,
        'selection_rule': 'median top10 teacher-prob-mass closest to 0.5, ties -> larger tau_T '
                          '(pre-specified comparison rule, NOT a claim of optimal temperature)',
        'selected_tau_t': best_tau,
    }
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'temperature_calibration.json').write_text(json.dumps(result, indent=2))
    print(f"[calibrate_temp] {cli.cell}: selected_tau_t={best_tau} "
         f"(medians: {[(t, round(rows[t]['top10_mass_median'], 4)) for t in CANDIDATE_TAUS]})")


if __name__ == '__main__':
    main()
