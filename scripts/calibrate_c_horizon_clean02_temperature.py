#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN02 -- section 8 teacher temperature
calibration.

Global teacher temperature is FIXED at tau_T,G=0.10 (spec section 8, not
selected). Each block's temperature is chosen, on a fixed TRAIN query
subset, as the candidate tau_T whose normalized entropy (entropy /
log(n_valid), comparable across different valid-candidate counts) is
closest to the Global teacher's own normalized entropy at tau_T,G=0.10 --
never using val/test. Distances are plain full-MSE-over-the-slice, exactly
`scripts.train_factorial_e2e01.individual_utility_memsafe` called on a
SLICED `memory_c`/`query_future` (mathematically identical to a fresh
block-MSE function since that function is generic over the trailing
horizon length -- no new distance code needed).
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
from scripts.train_horizon_retrieval_expert01 import BLOCK_NAMES, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value

BLOCKS = {'block1': (0, 96), 'block2': (96, 336), 'block3': (336, 720)}
CANDIDATE_TAUS = (0.01, 0.02, 0.05, 0.10, 0.20)
TAU_T_GLOBAL = 0.10


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


def normalized_entropy_and_effpos(d, valid_mask, tau_t):
    p_t = normalized_teacher_prob(d, valid_mask, tau_t)
    n_valid = valid_mask.sum(-1).clamp_min(2).float()  # log(1)=0 guard
    ent = -(p_t.masked_fill(~valid_mask, 0.0) * torch.log(p_t.clamp_min(1e-8))).sum(-1)
    norm_ent = ent / torch.log(n_valid)
    eff_pos = 1.0 / (p_t.square().sum(-1).clamp_min(1e-8))
    return norm_ent, eff_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--n_queries', type=int, default=512)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02')
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

    global_norm_ent, global_eff_pos = [], []
    per_block_tau = {b: {t: {'norm_ent': [], 'eff_pos': []} for t in CANDIDATE_TAUS} for b in BLOCK_NAMES}

    for s in range(0, n, cli.batch_size):
        bx = x_sub[s:s + cli.batch_size].float().to(device)
        by = y_sub[s:s + cli.batch_size].float().to(device)
        bstart = start_sub[s:s + cli.batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
            query_future = by[:, :, c]
            d_g = -individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d_g = d_g.masked_fill(~cand_mask, float('inf'))
            ne, ep = normalized_entropy_and_effpos(d_g, cand_mask, TAU_T_GLOBAL)
            global_norm_ent.append(ne.cpu()); global_eff_pos.append(ep.cpu())
            for bname in BLOCK_NAMES:
                lo, hi = BLOCKS[bname]
                d_b = -individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi],
                                                  cli.chunk_size)
                d_b = d_b.masked_fill(~cand_mask, float('inf'))
                for tau in CANDIDATE_TAUS:
                    neb, epb = normalized_entropy_and_effpos(d_b, cand_mask, tau)
                    per_block_tau[bname][tau]['norm_ent'].append(neb.cpu())
                    per_block_tau[bname][tau]['eff_pos'].append(epb.cpu())

    global_ne_mean = float(torch.cat(global_norm_ent).mean())
    global_ep_mean = float(torch.cat(global_eff_pos).mean())

    result = {'cell': cli.cell, 'n_queries': n, 'n_channels': len(channels),
             'tau_t_global': TAU_T_GLOBAL, 'global_normalized_entropy_mean': global_ne_mean,
             'global_effective_positives_mean': global_ep_mean,
             'selection_rule': 'per block, candidate tau_T in {0.01,0.02,0.05,0.10,0.20} whose '
             'TRAIN-subset normalized entropy mean is closest to the Global teacher\'s own '
             '(tau_T,G=0.10 fixed) normalized entropy mean -- val/test never used',
             'per_block': {}, 'selected_tau_t': {}}
    for bname in BLOCK_NAMES:
        rows = {}
        for tau in CANDIDATE_TAUS:
            ne_mean = float(torch.cat(per_block_tau[bname][tau]['norm_ent']).mean())
            ep_mean = float(torch.cat(per_block_tau[bname][tau]['eff_pos']).mean())
            rows[tau] = {'normalized_entropy_mean': ne_mean, 'effective_positives_mean': ep_mean}
        best_tau = min(CANDIDATE_TAUS, key=lambda t: abs(rows[t]['normalized_entropy_mean'] - global_ne_mean))
        result['per_block'][bname] = rows
        result['selected_tau_t'][bname] = best_tau
        print(f"[calibrate_c_clean02] {cli.cell}/{bname}: selected_tau_t={best_tau} "
             f"(global_norm_ent={global_ne_mean:.4f}, block norm_ents="
             f"{ {t: round(rows[t]['normalized_entropy_mean'],4) for t in CANDIDATE_TAUS} })")

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'teacher_temperature_calibration.json').write_text(json.dumps(result, indent=2))
    print(f'[calibrate_c_clean02] {cli.cell}: wrote {out_dir / "teacher_temperature_calibration.json"}')


if __name__ == '__main__':
    main()
