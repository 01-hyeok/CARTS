#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT01 -- component-level profiling + B0/B1 benchmark.

Profiles ~50-100 greedy-K=10 iterations of the REAL Weather Set Oracle
pipeline (encoder -> host scoring -> Set Oracle utility -> argmin loop),
broken into components (spec S1), then compares:

  B0 -- reference `dense_utility` (unmodified production path)
  B1 -- optimized `dense_utility_optimized` (algebraic reformulation)

No training. Read-only w.r.t. every running/completed experiment -- only
loads an existing checkpoint (or builds a fresh scratch encoder for a cell
with no trained Set arm yet) for inference.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.SequentialSetRetriever import SetConditioner
from scripts.diag_set_difficulty01 import HostScorer
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.test_set_oracle_equivalence import load_arm, load_scratch
from utils.dense_utility import candidate_weights, dense_utility
from utils.dense_utility_optimized import dense_utility_optimized, prepare_query_static


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def profile_reference(exp, args, host, model, sc, device, n_iters, top_k=10, chunk_size=4096):
    """Component-level timing of the UNMODIFIED reference pipeline."""
    _, loader = exp._get_data(flag='train')
    channel = 0
    timing = {'data_loading': 0.0, 'query_encoder': 0.0, 'candidate_encoder': 0.0,
             'host_scoring': 0.0, 'oracle_utility': 0.0, 'greedy_argmin': 0.0, 'other': 0.0}
    peak_mem = 0
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
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
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
        w = candidate_weights(host_scores, cand_mask, host.tau_topk)
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
            a = dense_utility(prefix, w, futures, q_future, chunk_size=chunk_size)
            _sync(device)
            timing['oracle_utility'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pick = a.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
            _sync(device)
            timing['greedy_argmin'] += time.perf_counter() - t0
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)
        n += 1
        if device.type == 'cuda':
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
    total_wall = time.perf_counter() - t_iter_start
    return timing, total_wall, n, peak_mem


@torch.no_grad()
def profile_optimized(exp, args, host, model, sc, device, n_iters, top_k=10, chunk_size=4096):
    """Same pipeline, Oracle component replaced by the optimized path, and
    d/d_sq computed once per query (prefix-invariant) instead of implicitly
    re-derived every step."""
    _, loader = exp._get_data(flag='train')
    channel = 0
    timing = {'data_loading': 0.0, 'query_encoder': 0.0, 'candidate_encoder': 0.0,
             'host_scoring': 0.0, 'query_static_prep': 0.0, 'oracle_utility': 0.0,
             'greedy_argmin': 0.0, 'other': 0.0}
    peak_mem = 0
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
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
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
        w = candidate_weights(host_scores, cand_mask, host.tau_topk)
        _sync(device)
        timing['host_scoring'] += time.perf_counter() - t0

        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        q_future = batch_y[:, :, channel]

        t0 = time.perf_counter()
        d, d_sq = prepare_query_static(futures, q_future)   # ONCE per query
        _sync(device)
        timing['query_static_prep'] += time.perf_counter() - t0

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
            a = dense_utility_optimized(prefix, w, d, d_sq, chunk_size=chunk_size)
            _sync(device)
            timing['oracle_utility'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pick = a.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
            _sync(device)
            timing['greedy_argmin'] += time.perf_counter() - t0
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)
        n += 1
        if device.type == 'cuda':
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
    total_wall = time.perf_counter() - t_iter_start
    return timing, total_wall, n, peak_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--n_iters', type=int, default=60)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_profile_ref', required=True)
    ap.add_argument('--out_profile_opt', required=True)
    ap.add_argument('--out_csv_row', default=None, help='append a benchmark row to this CSV')
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

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_ref, wall_ref, n_ref, peak_ref = profile_reference(
        exp, args, host, model, sc, device, cli.n_iters, chunk_size=cli.chunk_size)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_opt, wall_opt, n_opt, peak_opt = profile_optimized(
        exp, args, host, model, sc, device, cli.n_iters, chunk_size=cli.chunk_size)

    def _pct_table(timing, wall):
        total = sum(timing.values())
        return {k: {'seconds': v, 'pct_of_measured': 100.0 * v / max(total, 1e-9)}
               for k, v in timing.items()}

    prof_ref = {'cell': cli.cell, 'checkpoint': ckpt_desc, 'n_iters': n_ref,
               'wall_clock_total_s': wall_ref, 'sec_per_iter': wall_ref / max(n_ref, 1),
               'peak_gpu_mem_bytes': peak_ref, 'components': _pct_table(t_ref, wall_ref)}
    prof_opt = {'cell': cli.cell, 'checkpoint': ckpt_desc, 'n_iters': n_opt,
               'wall_clock_total_s': wall_opt, 'sec_per_iter': wall_opt / max(n_opt, 1),
               'peak_gpu_mem_bytes': peak_opt, 'components': _pct_table(t_opt, wall_opt)}

    Path(cli.out_profile_ref).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out_profile_ref).write_text(json.dumps(prof_ref, indent=2))
    Path(cli.out_profile_opt).write_text(json.dumps(prof_opt, indent=2))

    print(f'\n=== {cli.cell}: B0 reference vs B1 optimized ===')
    print(f"{'Component':<22}{'B0 sec/iter':>14}{'B0 %':>8}{'B1 sec/iter':>14}{'B1 %':>8}")
    for k in t_ref:
        b0 = t_ref[k] / max(n_ref, 1)
        b0p = prof_ref['components'][k]['pct_of_measured']
        b1 = t_opt.get(k, 0.0) / max(n_opt, 1)
        b1p = prof_opt['components'].get(k, {}).get('pct_of_measured', 0.0)
        print(f'{k:<22}{b0:>14.5f}{b0p:>8.2f}{b1:>14.5f}{b1p:>8.2f}')
    speedup = (wall_ref / max(n_ref, 1)) / max(wall_opt / max(n_opt, 1), 1e-9)
    oracle_speedup = (t_ref['oracle_utility'] / max(n_ref, 1)) / max(t_opt['oracle_utility'] / max(n_opt, 1), 1e-9)
    print(f'\nTotal sec/iter: B0={wall_ref/max(n_ref,1):.5f}  B1={wall_opt/max(n_opt,1):.5f}  '
         f'speedup={speedup:.2f}x')
    print(f'Oracle-component-only speedup: {oracle_speedup:.2f}x')
    print(f'Peak GPU mem: B0={peak_ref/1e6:.1f} MB  B1={peak_opt/1e6:.1f} MB')

    if cli.out_csv_row:
        row = {
            'cell': cli.cell, 'version': 'B0_reference', 'sec_per_iter': wall_ref / max(n_ref, 1),
            'sec_per_100_iter': 100 * wall_ref / max(n_ref, 1), 'peak_vram_mb': peak_ref / 1e6,
            'oracle_ms_per_greedy_step': 1000 * t_ref['oracle_utility'] / max(n_ref, 1) / 10,
        }
        row2 = {
            'cell': cli.cell, 'version': 'B1_optimized', 'sec_per_iter': wall_opt / max(n_opt, 1),
            'sec_per_100_iter': 100 * wall_opt / max(n_opt, 1), 'peak_vram_mb': peak_opt / 1e6,
            'oracle_ms_per_greedy_step': 1000 * t_opt['oracle_utility'] / max(n_opt, 1) / 10,
        }
        exists = Path(cli.out_csv_row).exists()
        with open(cli.out_csv_row, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if not exists:
                w.writeheader()
            w.writerow(row)
            w.writerow(row2)


if __name__ == '__main__':
    main()
