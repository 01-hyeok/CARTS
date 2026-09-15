#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT03 -- component-level profiling of the Oracle compute
paths ACTUALLY used by the running Weather H96/H720 factorial
(`scripts/train_factorial_e2e01.py`): Individual Oracle (A), Greedy Set
Oracle (E), and the shared Oracle-Choice CE loss (C). Forward-only
(no_grad), read-only against every checkpoint it loads -- matches
`scripts/benchmark_weather_opt01.py`'s own profiling convention (no
backward pass timed, for direct comparability with that report's numbers).

Two arm TYPES profiled separately (individual_utility has a different
component name, `oracle_utility_individual`, than the Set Oracle's
`oracle_utility_set`, so their shares never overwrite one another):
    --arm_type individual   (uses scripts.train_factorial_e2e01.individual_utility)
    --arm_type greedy_set   (uses utils.dense_utility.dense_utility, the reference
                             -- the SAME function OPT01/OPT02 already profiled and
                             optimized; reprofiled here in the Individual-Oracle-
                             adjacent context for a like-for-like OPT03 comparison
                             table, not because its own numbers are new)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_set_difficulty01 import HostScorer
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility
from scripts.train_margutil01 import memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.test_set_oracle_equivalence import load_arm, load_scratch
from utils.dense_utility import candidate_weights, dense_utility


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def profile(arm_type, exp, args, host, model, sc, device, n_iters, top_k, chunk_size, tau_choice):
    _, loader = exp._get_data(flag='train')
    channel = 0
    comp = [f'oracle_utility_{arm_type}']
    timing = {'data_loading': 0.0, 'query_encoder': 0.0, 'candidate_encoder': 0.0,
             'host_scoring': 0.0, comp[0]: 0.0, 'choice_ce': 0.0,
             'greedy_argmax': 0.0, 'other': 0.0}
    call_counts = {comp[0]: 0, 'choice_ce': 0}
    max_shape = {comp[0]: None}
    peak_alloc, peak_reserved = 0, 0
    it = iter(loader)
    n = 0
    t_iter_start = time.perf_counter()
    for _ in range(n_iters):
        t0 = time.perf_counter()
        try:
            batch_x, batch_y, batch_start_idx = next(it)
        except StopIteration:
            it = iter(loader)
            batch_x, batch_y, batch_start_idx = next(it)
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        _sync(device)
        timing['data_loading'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        z_q = encode_raw(model, batch_x, channel)
        _sync(device)
        timing['query_encoder'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        E = encode_raw(model, exp.memory_x, channel)
        _sync(device)
        timing['candidate_encoder'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        host_scores = host.scores(batch_x, channel, cand_mask)
        w_host = candidate_weights(host_scores, cand_mask, tau_choice)
        _sync(device)
        timing['host_scoring'] += time.perf_counter() - t0

        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        q_future = batch_y[:, :, channel]

        selected = torch.zeros_like(cand_mask)
        picks = []
        for t in range(top_k):
            t0 = time.perf_counter()
            if t == 0:
                h_t = z_q
            else:
                m = E[torch.stack(picks, dim=1)].mean(dim=1)
                h_t = sc(z_q, m)
            u_hat = arm_score(h_t, E, None)
            _sync(device)
            timing['other'] += time.perf_counter() - t0

            valid_now = cand_mask & ~selected
            prefix = torch.stack(picks, dim=1) if picks else torch.zeros(
                batch_x.size(0), 0, dtype=torch.long, device=device)

            t0 = time.perf_counter()
            if arm_type == 'individual':
                u_target = individual_utility(futures, q_future)  # called every step (as in production)
            else:
                u_target = -dense_utility(prefix, w_host, futures, q_future, chunk_size=chunk_size)
            _sync(device)
            timing[comp[0]] += time.perf_counter() - t0
            call_counts[comp[0]] += 1
            max_shape[comp[0]] = tuple(futures.shape)

            t0 = time.perf_counter()
            loss_t, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
            _sync(device)
            timing['choice_ce'] += time.perf_counter() - t0
            call_counts['choice_ce'] += 1

            t0 = time.perf_counter()
            pick = u_target.masked_fill(~valid_now, float('-inf')).argmax(dim=-1)
            _sync(device)
            timing['greedy_argmax'] += time.perf_counter() - t0
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)
        n += 1
        if device.type == 'cuda':
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
    total_wall = time.perf_counter() - t_iter_start
    return timing, total_wall, n, peak_alloc, peak_reserved, call_counts, max_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_type', choices=['individual', 'greedy_set'], required=True)
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--n_iters', type=int, default=30)
    ap.add_argument('--n_warmup', type=int, default=3)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_json', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cli.scratch_reference_ckpt:
        exp, args, model, sc = load_scratch(cli.scratch_reference_ckpt, cli.scratch_pred_len, device)
        ckpt_desc = f'SCRATCH({cli.scratch_reference_ckpt}, pred_len={cli.scratch_pred_len})'
    else:
        exp, args, model, sc = load_arm(cli.arm_checkpoint, device)
        exp._ensure_memory()
        ckpt_desc = cli.arm_checkpoint
    host = HostScorer(cli.stage2_host, device)
    tau_choice = host.tau_topk

    # warm-up (discarded)
    profile(cli.arm_type, exp, args, host, model, sc, device, cli.n_warmup, 10, cli.chunk_size, tau_choice)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    timing, total_wall, n, peak_alloc, peak_res, call_counts, max_shape = profile(
        cli.arm_type, exp, args, host, model, sc, device, cli.n_iters, 10, cli.chunk_size, tau_choice)

    total_component = sum(timing.values())
    rows = []
    for k, v in timing.items():
        rows.append({
            'cell': cli.cell, 'arm_type': cli.arm_type, 'component': k,
            'sec_per_iter': v / max(n, 1),
            'pct_of_total_iter_time': 100.0 * v / max(total_component, 1e-12),
            'calls_per_iter': call_counts.get(k, 10 if k not in
                                              ('data_loading', 'query_encoder',
                                               'candidate_encoder', 'host_scoring') else 1),
            'max_tensor_shape': str(max_shape.get(k, '-')),
        })
    Path(cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out = {
        'cell': cli.cell, 'arm_type': cli.arm_type, 'checkpoint': ckpt_desc,
        'chunk_size': cli.chunk_size, 'n_iters': n,
        'total_wall_sec': total_wall, 'sec_per_iter': total_wall / max(n, 1),
        'peak_allocated_mb': peak_alloc / 1e6, 'peak_reserved_mb': peak_res / 1e6,
        'components': rows,
        'note': 'forward-only (no_grad) profiling, matches benchmark_weather_opt01.py convention; '
               'GPU shared with other concurrently running TRACK-A jobs -- see report for contention caveat',
    }
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out_json).write_text(json.dumps(out, indent=2))
    print(f'[profile_opt03] {cli.cell}/{cli.arm_type}: sec_per_iter={out["sec_per_iter"]:.4f} '
         f'peak_alloc={out["peak_allocated_mb"]:.1f}MB checkpoint={ckpt_desc}')
    for r in rows:
        print(f"  {r['component']:>28s}: {r['sec_per_iter']:.5f}s/iter "
             f"({r['pct_of_total_iter_time']:.1f}%) calls/iter={r['calls_per_iter']}")


if __name__ == '__main__':
    main()
