#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT03 -- B0 vs B1 (chunk 2048) vs B2 (chunk 4096)
benchmark for the paths OPT03 actually optimizes: Individual Oracle (A) and
Oracle-Choice CE (C). The Greedy Set Oracle (E) is NOT rebenchmarked here --
it is the exact, unmodified OPT02 code
(`utils.dense_utility_optimized.dense_utility_optimized`, re-exported by
`utils/oracle_compute_optimized.py`), already benchmarked in
`results/TRACK-A-WEATHER-OPT02/benchmark_H{96,720}*.csv`; this script cites
those numbers rather than duplicating the measurement.

`--path {individual,choice_ce}` selects which of A/C to benchmark.
Oracle-Choice CE has no chunk parameter (dead-code elimination, not
chunking) -- for `--path choice_ce` the `--chunk_sizes` argument is ignored
and only ONE optimized variant is measured, reported under the `B1` column
for table-format consistency with the other paths.

Real Weather pipeline (encoder -> host scoring -> K=10 greedy loop),
`torch.cuda.synchronize()` brackets ONLY the timed component per iteration,
same warm-up policy for every version -- identical convention to
`scripts/benchmark_weather_opt02.py`.
"""
import argparse
import csv
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
from utils.dense_utility import candidate_weights
from utils.oracle_compute_optimized import (individual_oracle_utility_optimized,
                                            oracle_choice_step_loss_optimized,
                                            prepare_individual_query_static)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def run_version(path, version, chunk_size, exp, args, host, model, sc, device,
                n_iters, n_warmup, top_k=10):
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

    def one_iter(record):
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

        cand_sq = None
        if path == 'individual' and version != 'B0':
            _sync(device)
            t0 = time.perf_counter()
            cand_sq = prepare_individual_query_static(futures, candidate_chunk_size=chunk_size)
            _sync(device)
            if record is not None:
                record['path_time'] += time.perf_counter() - t0

        selected = torch.zeros_like(cand_mask)
        picks = []
        for t in range(top_k):
            h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
            u_hat = arm_score(h_t, E, None)
            valid_now = cand_mask & ~selected

            if path == 'individual':
                _sync(device)
                t0 = time.perf_counter()
                if version == 'B0':
                    u_target = individual_utility(futures, q_future)
                else:
                    u_target = individual_oracle_utility_optimized(
                        futures, q_future, cand_sq=cand_sq, candidate_chunk_size=chunk_size)
                _sync(device)
                if record is not None:
                    record['path_time'] += time.perf_counter() - t0
            else:  # choice_ce -- needs a target; individual Oracle stands in
                    # as a realistic u_target source (path itself is what's timed)
                u_target = individual_utility(futures, q_future)

            if path == 'choice_ce':
                _sync(device)
                t0 = time.perf_counter()
                if version == 'B0':
                    _, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, host.tau_topk)
                else:
                    _, _ = oracle_choice_step_loss_optimized(u_hat, u_target, valid_now, host.tau_topk)
                _sync(device)
                if record is not None:
                    record['path_time'] += time.perf_counter() - t0

            pick = u_target.masked_fill(~valid_now, float('-inf')).argmax(dim=-1)
            picks.append(pick)
            selected = selected.scatter(1, pick.unsqueeze(-1), True)

    for _ in range(n_warmup):
        one_iter(None)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    record = {'path_time': 0.0}
    t_wall_start = time.perf_counter()
    for _ in range(n_iters):
        one_iter(record)
    total_time = time.perf_counter() - t_wall_start
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    peak_res = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
    return record['path_time'], total_time, n_iters, peak_alloc, peak_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--path', choices=['individual', 'choice_ce'], required=True)
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--n_iters', type=int, default=30)
    ap.add_argument('--n_warmup', type=int, default=3)
    ap.add_argument('--chunk_sizes', default='2048,4096')
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
    chunk_sizes = [int(x) for x in cli.chunk_sizes.split(',')] if cli.path == 'individual' else [None]
    versions = ['B0', 'B1', 'B2'] if cli.path == 'individual' else ['B0', 'B1']
    for cs in chunk_sizes:
        for version in versions:
            path_t, total_t, n, peak_alloc, peak_res = run_version(
                cli.path, version, cs, exp, args, host, model, sc, device, cli.n_iters, cli.n_warmup)
            row = {
                'cell': cli.cell, 'path': cli.path, 'version': version,
                'chunk_size': cs if cs is not None else 'n/a',
                'n_iters': n, 'path_sec_per_iter': path_t / max(n, 1),
                'total_sec_per_iter': total_t / max(n, 1),
                'peak_allocated_mb': peak_alloc / 1e6, 'peak_reserved_mb': peak_res / 1e6,
            }
            rows.append(row)
            print(f"[bench_opt03] {cli.cell} {cli.path} chunk={cs} {version}: "
                 f"path={row['path_sec_per_iter']:.5f}s/iter "
                 f"total={row['total_sec_per_iter']:.5f}s/iter "
                 f"peak_alloc={row['peak_allocated_mb']:.1f}MB")

    Path(cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[bench_opt03] wrote {cli.out_csv} ({len(rows)} rows), checkpoint={ckpt_desc}')


if __name__ == '__main__':
    main()
