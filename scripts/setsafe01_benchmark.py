#!/usr/bin/env python3
"""TRACK-A-WEATHER-SETSAFE01 -- real training-iteration benchmark for the
Set-Oracle SAFE path specifically: B0 = reference C + reference E, B1 =
optimized C + reference E (`--oracle_compute_impl safe`). Unlike OPT04's
benchmark (which measured the C+E bundle for Set arms), this isolates
Choice-CE's own contribution with E held fixed at reference throughout, so
the reported speedup cannot be attributed to the (rejected) Set-Oracle
algebraic optimization.

Full forward + backward + `optimizer.step()`, real Weather H720 batches,
`torch.cuda.synchronize()` at every timed boundary. Reports per-iteration
component splits (encoder time, Set-Oracle-utility time, choice_ce time)
in addition to total wall/compute time, so "Set utility time" and "encoder
time" can be reported separately as requested.
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

from scripts import train_factorial_e2e01 as T
from scripts.setsafe01_verify import build_fresh_full, clone_state
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _p95(xs):
    xs = sorted(xs)
    return xs[max(0, int(round(0.95 * (len(xs) - 1))))]


def run_iters(oracle_compute_impl, chunk_size, reference_ckpt, stage2_host, pred_len, device,
             n_warmup, n_iters, seed, init_state, scorer, prefix_policy):
    exp, args, model, sc, metric = build_fresh_full(reference_ckpt, pred_len, device, seed, scorer)
    model.load_state_dict(init_state['model'])
    sc.load_state_dict(init_state['sc'])
    if metric is not None and init_state.get('metric') is not None:
        metric.load_state_dict(init_state['metric'])
    host = T.HostScorer(stage2_host, device)
    tau_topk = host.tau_topk
    channels = list(range(int(args.enc_in)))
    torch.manual_seed(seed)
    _, loader = exp._get_data(flag='train')
    it = iter(loader)
    params = ([p for p in model.parameters() if p.requires_grad] + list(sc.parameters())
             + (list(metric.parameters()) if metric is not None else []))
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)
    resolved = T.resolve_oracle_compute_impl(oracle_compute_impl, chunk_size)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    def one_iter(record):
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
            t0 = time.perf_counter()
            E = T.encode_raw(model, exp.memory_x, c)
            z_q = T.encode_raw(model, batch_x, c)
            _sync(device)
            if record is not None:
                record['encoder_time'] += time.perf_counter() - t0

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, tau_topk)
            losses, _, _, _ = T.run_sequence(
                z_q, E, cand_mask, sc, metric, w_host, futures, query_future,
                'greedy_set', prefix_policy, tau_topk, 10, chunk_size,
                choice_ce_impl=resolved['choice_ce_impl'],
                individual_impl=resolved['individual_impl'],
                greedy_set_impl=resolved['greedy_set_impl'])
            batch_loss = batch_loss + sum(losses) / 10
        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        optimizer.step()
        _sync(device)
        t1 = time.perf_counter()
        if record is not None:
            record['wall'].append(t1 - t_wall0)
            record['compute'].append(t1 - t_compute0)
        return

    for _ in range(n_warmup):
        one_iter(None)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    record = {'wall': [], 'compute': [], 'encoder_time': 0.0}
    for _ in range(n_iters):
        one_iter(record)

    peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    peak_res = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
    return {
        'oracle_compute_impl': oracle_compute_impl, 'resolved': resolved,
        'n_warmup': n_warmup, 'n_iters': n_iters,
        'wall_median_sec': statistics.median(record['wall']), 'wall_p95_sec': _p95(record['wall']),
        'compute_median_sec': statistics.median(record['compute']), 'compute_p95_sec': _p95(record['compute']),
        'encoder_time_total_sec': record['encoder_time'],
        'encoder_time_per_iter_sec': record['encoder_time'] / max(n_iters, 1),
        'raw_wall_samples_sec': record['wall'], 'raw_compute_samples_sec': record['compute'],
        'peak_allocated_mb': peak_alloc / 1e6, 'peak_reserved_mb': peak_res / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix_policy', default='onpolicy')
    ap.add_argument('--scorer', default='cosine')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_warmup', type=int, default=20)
    ap.add_argument('--n_iters', type=int, default=100)
    ap.add_argument('--n_repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--contended', action='store_true')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_json', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp0, args0, model0, sc0, metric0 = build_fresh_full(cli.reference_ckpt, cli.pred_len, device,
                                                          cli.seed, cli.scorer)
    init_state = {'model': clone_state(model0), 'sc': clone_state(sc0),
                  'metric': clone_state(metric0) if metric0 is not None else None}
    del exp0, args0, model0, sc0, metric0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    rows = []
    for impl in ('reference', 'safe'):
        for rep in range(cli.n_repeats):
            r = run_iters(impl, cli.chunk_size, cli.reference_ckpt, cli.stage2_host, cli.pred_len,
                         device, cli.n_warmup, cli.n_iters, cli.seed, init_state, cli.scorer,
                         cli.prefix_policy)
            r['repeat'] = rep
            r['contended'] = bool(cli.contended)
            rows.append(r)
            print(f"[setsafe01_benchmark] impl={impl} rep={rep} "
                 f"wall_median={r['wall_median_sec']:.4f}s wall_p95={r['wall_p95_sec']:.4f}s "
                 f"iters_per_sec={1.0/r['wall_median_sec']:.3f} "
                 f"peak_alloc={r['peak_allocated_mb']:.1f}MB contended={cli.contended}")

    Path(cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out_csv, 'w', newline='') as fh:
        flat = []
        for r in rows:
            fr = {k: v for k, v in r.items() if k not in ('resolved', 'raw_wall_samples_sec',
                                                           'raw_compute_samples_sec')}
            fr.update({f'resolved_{k}': v for k, v in r['resolved'].items() if k != 'fallback_reason'})
            flat.append(fr)
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        for r in flat:
            w.writerow(r)
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out_json).write_text(json.dumps(rows, indent=2))
    print(f'[setsafe01_benchmark] wrote {cli.out_csv} / {cli.out_json}')


if __name__ == '__main__':
    main()
