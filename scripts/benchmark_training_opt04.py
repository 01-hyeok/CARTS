#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT04 -- REAL training-iteration benchmark: full
forward + backward + `optimizer.step()`, using the actual production
`run_sequence`/dispatcher wiring in `scripts/train_factorial_e2e01.py`
(not an isolated microbenchmark). B0 = `--oracle_compute_impl reference`,
B1 = `--oracle_compute_impl optimized`. Same scratch initial model state
(saved once, reloaded for both B0 and B1), same batch order (same
`torch.manual_seed` before building the loader iterator), same optimizer
hyperparameters, same K=10, same chunk_size, FP32.

Reports both wall-clock-with-dataloader and compute-only (post-batch-
transfer) timings, median and p95 over the measured window,
`torch.cuda.synchronize()` bracketing every timed boundary, plus peak
allocated/reserved VRAM (`reset_peak_memory_stats` before the measured
window).
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts import train_factorial_e2e01 as T
from scripts.train_margutil01 import build_experiment, memory_value
from utils.dense_utility import candidate_weights


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _p95(xs):
    xs = sorted(xs)
    idx = max(0, int(round(0.95 * (len(xs) - 1))))
    return xs[idx]


def build_fresh(reference_ckpt, pred_len, device, seed):
    torch.manual_seed(seed)
    exp, args = build_experiment(reference_ckpt, {'pred_len': pred_len, 'seq_len': pred_len,
                                                  'batch_size': 32, 'seed': seed, 'top_k': 10})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    return exp, args, model, sc


def run_training_iters(cell, target, prefix_policy, oracle_compute_impl, chunk_size,
                       reference_ckpt, stage2_host, pred_len, device,
                       n_warmup, n_iters, seed, init_state):
    exp, args, model, sc = build_fresh(reference_ckpt, pred_len, device, seed)
    model.load_state_dict(init_state['model'])
    sc.load_state_dict(init_state['sc'])
    host = T.HostScorer(stage2_host, device)
    tau_topk = host.tau_topk
    channels = list(range(int(args.enc_in)))
    torch.manual_seed(seed)  # same batch order for B0/B1
    _, loader = exp._get_data(flag='train')
    it = iter(loader)

    params = [p for p in model.parameters() if p.requires_grad] + list(sc.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)

    resolved = T.resolve_oracle_compute_impl(oracle_compute_impl, chunk_size)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    def one_iter():
        t_wall0 = time.perf_counter()
        batch_x, batch_y, batch_start_idx = next_batch()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        _sync(device)
        t_compute0 = time.perf_counter()

        optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            E = T.encode_raw(model, exp.memory_x, c)
            z_q = T.encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, tau_topk)
            losses, _, _, _ = T.run_sequence(
                z_q, E, cand_mask, sc, None, w_host, futures, query_future,
                target, prefix_policy, tau_topk, 10, chunk_size,
                choice_ce_impl=resolved['choice_ce_impl'],
                individual_impl=resolved['individual_impl'],
                greedy_set_impl=resolved['greedy_set_impl'])
            batch_loss = batch_loss + sum(losses) / 10
        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        optimizer.step()
        _sync(device)
        t1 = time.perf_counter()
        return (t1 - t_wall0), (t1 - t_compute0)

    for _ in range(n_warmup):
        one_iter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    wall_times, compute_times = [], []
    for _ in range(n_iters):
        w, c = one_iter()
        wall_times.append(w)
        compute_times.append(c)

    peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    peak_res = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
    return {
        'cell': cell, 'target': target, 'prefix_policy': prefix_policy,
        'oracle_compute_impl': oracle_compute_impl, 'resolved': resolved, 'chunk_size': chunk_size,
        'n_warmup': n_warmup, 'n_iters': n_iters,
        'wall_median_sec': statistics.median(wall_times), 'wall_p95_sec': _p95(wall_times),
        'compute_median_sec': statistics.median(compute_times), 'compute_p95_sec': _p95(compute_times),
        'wall_samples_per_sec': 32.0 / statistics.median(wall_times),
        'compute_samples_per_sec': 32.0 / statistics.median(compute_times),
        'peak_allocated_mb': peak_alloc / 1e6, 'peak_reserved_mb': peak_res / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--target', choices=['individual', 'greedy_set'], required=True)
    ap.add_argument('--prefix_policy', default='onpolicy')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_warmup', type=int, default=5)
    ap.add_argument('--n_iters', type=int, default=30)
    ap.add_argument('--n_repeats', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--contended', action='store_true',
                    help='mark this run as measured under GPU contention from other jobs')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # SAME initial model/set_conditioner state for B0 and B1
    exp0, args0, model0, sc0 = build_fresh(cli.reference_ckpt, cli.pred_len, device, cli.seed)
    init_state = {'model': {k: v.detach().clone() for k, v in model0.state_dict().items()},
                  'sc': {k: v.detach().clone() for k, v in sc0.state_dict().items()}}
    del exp0, args0, model0, sc0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    rows = []
    for impl in ('reference', 'optimized'):
        for rep in range(cli.n_repeats):
            r = run_training_iters(cli.cell, cli.target, cli.prefix_policy, impl, cli.chunk_size,
                                   cli.reference_ckpt, cli.stage2_host, cli.pred_len, device,
                                   cli.n_warmup, cli.n_iters, cli.seed, init_state)
            r['repeat'] = rep
            r['contended'] = bool(cli.contended)
            rows.append(r)
            print(f"[bench_training_opt04] {cli.cell}/{cli.target}/{cli.prefix_policy} "
                 f"impl={impl} rep={rep} wall_median={r['wall_median_sec']:.4f}s "
                 f"compute_median={r['compute_median_sec']:.4f}s "
                 f"peak_alloc={r['peak_allocated_mb']:.1f}MB contended={cli.contended}")

    Path(cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out_csv, 'w', newline='') as fh:
        flat_rows = []
        for r in rows:
            fr = {k: v for k, v in r.items() if k != 'resolved'}
            fr.update({f'resolved_{k}': v for k, v in r['resolved'].items() if k != 'fallback_reason'})
            flat_rows.append(fr)
        w = csv.DictWriter(fh, fieldnames=list(flat_rows[0]))
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out_json).write_text(json.dumps(rows, indent=2))
    print(f'[bench_training_opt04] wrote {cli.out_csv} / {cli.out_json}')


if __name__ == '__main__':
    main()
