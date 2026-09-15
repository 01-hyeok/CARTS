#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT02 -- B0 (reference) vs B1 (OPT01, unchunked
optimized) vs B2 (OPT02, chunked optimized) benchmark.

Timer boundary (spec C, stated explicitly): each `torch.cuda.synchronize()`
brackets ONLY the Set Oracle component call itself
(`dense_utility`/`dense_utility_optimized`) inside the K=10 greedy loop --
`data_loading`/`query_encoder`/`candidate_encoder`/`host_scoring` are timed
identically across B0/B1/B2 (same code, run once per outer iteration,
outside the K loop) and are included in `total_wall` but not separately
re-measured per version, since OPT01/OPT02 do not touch them. `peak
allocated`/`peak reserved` VRAM are read via
`torch.cuda.max_memory_allocated()` / `torch.cuda.max_memory_reserved()`,
reset via `torch.cuda.reset_peak_memory_stats()` before each version's run.

Same warm-up policy for all three versions: the first iteration of each
version is run and discarded (JIT/caching warm-up) before timed iterations
begin, so all three see comparable warm CUDA-cache state.
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
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import memory_value
from scripts.test_set_oracle_equivalence import load_arm, load_scratch
from utils.dense_utility import candidate_weights, dense_utility
from utils.dense_utility_optimized import (dense_utility_optimized, prepare_query_static,
                                           prepare_query_static_chunked)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def run_version(version, candidate_chunk_size, exp, args, host, model, sc, device,
                n_iters, n_warmup, top_k=10):
    """version in {'B0', 'B1', 'B2'}. Returns (oracle_time_s, total_time_s,
    n_measured, peak_allocated_bytes, peak_reserved_bytes)."""
    _, loader = exp._get_data(flag='train')
    channel = 0
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    for _ in range(n_warmup):
        batch_x, batch_y, batch_start_idx = next_batch()
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        z_q = encode_raw(model, batch_x, channel)
        E = encode_raw(model, exp.memory_x, channel)
        host_scores = host.scores(batch_x, channel, cand_mask)
        w = candidate_weights(host_scores, cand_mask, host.tau_topk)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        q_future = batch_y[:, :, channel]
        if version == 'B0':
            d = d_sq = None
        elif version == 'B1':
            d, d_sq = prepare_query_static(futures, q_future)
        else:
            d = None
            d_sq = prepare_query_static_chunked(futures, q_future,
                                                candidate_chunk_size=candidate_chunk_size)
        selected = torch.zeros_like(cand_mask)
        picks = []
        for t in range(top_k):
            h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
            u_hat = arm_score(h_t, E, None)
            valid_now = cand_mask & ~selected
            prefix = torch.stack(picks, dim=1) if picks else torch.zeros(
                batch_x.size(0), 0, dtype=torch.long, device=device)
            if version == 'B0':
                a = dense_utility(prefix, w, futures, q_future, chunk_size=candidate_chunk_size)
            elif version == 'B1':
                a = dense_utility_optimized(prefix, w, d=d, d_sq=d_sq, chunk_size=candidate_chunk_size)
            else:
                a = dense_utility_optimized(prefix, w, futures=futures, query_future=q_future,
                                            d_sq=d_sq, candidate_chunk_size=candidate_chunk_size)
            pick = a.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    oracle_time, n = 0.0, 0
    t_wall_start = time.perf_counter()
    for _ in range(n_iters):
        batch_x, batch_y, batch_start_idx = next_batch()
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        z_q = encode_raw(model, batch_x, channel)
        E = encode_raw(model, exp.memory_x, channel)
        host_scores = host.scores(batch_x, channel, cand_mask)
        w = candidate_weights(host_scores, cand_mask, host.tau_topk)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        q_future = batch_y[:, :, channel]

        if version == 'B0':
            d = d_sq = None
        elif version == 'B1':
            _sync(device)
            t0 = time.perf_counter()
            d, d_sq = prepare_query_static(futures, q_future)
            _sync(device)
            oracle_time += time.perf_counter() - t0
        else:
            d = None
            _sync(device)
            t0 = time.perf_counter()
            d_sq = prepare_query_static_chunked(futures, q_future,
                                                candidate_chunk_size=candidate_chunk_size)
            _sync(device)
            oracle_time += time.perf_counter() - t0

        selected = torch.zeros_like(cand_mask)
        picks = []
        for t in range(top_k):
            h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
            u_hat = arm_score(h_t, E, None)
            valid_now = cand_mask & ~selected
            prefix = torch.stack(picks, dim=1) if picks else torch.zeros(
                batch_x.size(0), 0, dtype=torch.long, device=device)

            _sync(device)
            t0 = time.perf_counter()
            if version == 'B0':
                a = dense_utility(prefix, w, futures, q_future, chunk_size=candidate_chunk_size)
            elif version == 'B1':
                a = dense_utility_optimized(prefix, w, d=d, d_sq=d_sq, chunk_size=candidate_chunk_size)
            else:
                a = dense_utility_optimized(prefix, w, futures=futures, query_future=q_future,
                                            d_sq=d_sq, candidate_chunk_size=candidate_chunk_size)
            _sync(device)
            oracle_time += time.perf_counter() - t0

            pick = a.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)
        n += 1
    total_time = time.perf_counter() - t_wall_start
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
    return oracle_time, total_time, n, peak_alloc, peak_reserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--n_iters', type=int, default=30)
    ap.add_argument('--n_warmup', type=int, default=3)
    ap.add_argument('--chunk_sizes', default='128,256,512,1024,2048')
    ap.add_argument('--out_csv', required=True)
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

    rows = []
    chunk_sizes = [int(x) for x in cli.chunk_sizes.split(',')]
    # B0 (reference) does not depend on candidate_chunk_size in the same way
    # (its own internal chunk_size is a no-op knob already present in
    # dense_utility) -- still benchmarked once per chunk_size value for a
    # fully apples-to-apples row-by-row comparison table.
    for cs in chunk_sizes:
        for version in ('B0', 'B1', 'B2'):
            oracle_t, total_t, n, peak_alloc, peak_res = run_version(
                version, cs, exp, args, host, model, sc, device, cli.n_iters, cli.n_warmup)
            row = {
                'cell': cli.cell, 'version': version, 'candidate_chunk_size': cs,
                'n_iters': n, 'oracle_sec_per_iter': oracle_t / max(n, 1),
                'total_sec_per_iter': total_t / max(n, 1),
                'sec_per_100_iter': 100 * total_t / max(n, 1),
                'peak_allocated_mb': peak_alloc / 1e6, 'peak_reserved_mb': peak_res / 1e6,
            }
            rows.append(row)
            print(f"[benchmark_opt02] {cli.cell} chunk={cs:>5} {version}: "
                 f"oracle={row['oracle_sec_per_iter']:.5f}s/iter "
                 f"total={row['total_sec_per_iter']:.5f}s/iter "
                 f"peak_alloc={row['peak_allocated_mb']:.1f}MB "
                 f"peak_reserved={row['peak_reserved_mb']:.1f}MB")

    Path(cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[benchmark_opt02] wrote {cli.out_csv} ({len(rows)} rows), checkpoint={ckpt_desc}')


if __name__ == '__main__':
    main()
