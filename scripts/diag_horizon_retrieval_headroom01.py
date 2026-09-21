#!/usr/bin/env python3
"""TRACK-A-HORIZON-RETRIEVAL-HEADROOM01 -- Oracle diagnostic only: does one
retrieved Top-K set suffice for the entire H=720 forecast horizon, or would
block-specific (short/mid/long) Top-K sets do meaningfully better?

No retriever is trained and no checkpoint weights are loaded here -- this is
a pure Oracle-utility computation over the existing candidate memory bank,
future values reconstructed via the SAME production
`scripts.train_margutil01.memory_value()` convention every other Track-A
experiment uses. `--reference_ckpt` supplies dataset/config args only
(weights never loaded), exactly the `build_experiment` pattern used
throughout this project.

Per-candidate distance at horizon H is `individual_utility_memsafe` --
`u_i = -MSE(memory_c[i] + offset_c, query_future)`, mean over the LAST axis
of whatever is passed in. Passing a time-sliced `memory_c`/`query_future`
therefore computes exactly the BLOCK utility for that slice with zero new
math -- this script never reimplements the -MSE reduction, it only slices
inputs before an unmodified call, so it is automatically candidate-chunked
(memsafe) for Solar's large N the same way P0.2 already made it for
train_factorial_e2e01.py.

Block partition (fixed, spec section 3): B1=[0:96], B2=[96:336], B3=[336:720].
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

BLOCKS = {'block1': (0, 96), 'block2': (96, 336), 'block3': (336, 720)}
TOP_K = 10
TOP_K_UNION_CAP = 30  # |S_B1 u S_B2 u S_B3| <= 3*K = 30 always


def _query_id(batch_start_idx):
    return (batch_start_idx if torch.is_tensor(batch_start_idx)
           else torch.as_tensor(batch_start_idx)).cpu()


def _gather_mean(memory_c, offset_c, picks, h_lo, h_hi):
    """mean_{i in picks} (memory_c[i, h_lo:h_hi] + offset_c) -- [B, h_hi-h_lo]."""
    seg = memory_c[:, h_lo:h_hi]  # [N, H']
    gathered = seg[picks]  # [B, K, H']
    return gathered.mean(dim=1) + offset_c.view(-1, 1)


def _gather_mean_masked(memory_c, offset_c, picks, counts, h_lo, h_hi):
    """Same as `_gather_mean` but only the first `counts[b]` columns of
    `picks` (already sorted best-first) count toward the mean per row --
    used for the budget-matched control where the picked count varies
    per query/channel."""
    seg = memory_c[:, h_lo:h_hi]
    gathered = seg[picks]  # [B, K, H']
    col = torch.arange(picks.size(1), device=picks.device).unsqueeze(0)
    keep = (col < counts.unsqueeze(1)).float().unsqueeze(-1)  # [B, K, 1]
    summed = (gathered * keep).sum(dim=1)
    denom = counts.clamp_min(1).float().unsqueeze(-1)
    return summed / denom + offset_c.view(-1, 1)


def _mse(pred, target):
    return ((pred - target) ** 2).mean(dim=-1)


def _rank_of(logits, valid_mask, picks):
    """1-based rank of each index in `picks` under `logits` (higher=better),
    masked to `valid_mask`. [B, len(picks-per-row)]."""
    masked = logits.masked_fill(~valid_mask, float('-inf'))
    target_vals = masked.gather(1, picks)
    rank = (masked.unsqueeze(-1) > target_vals.unsqueeze(1)).sum(dim=1) + 1
    return rank


@torch.no_grad()
def run_split(exp, args, split, channels, top_k=TOP_K, chunk_size=4096, limit_batches=0):
    exp._ensure_memory()
    device = exp.device
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    _, loader = exp._get_data(flag=split, shuffle=False)

    rows = []
    channel_rows = []
    n_batches = 0
    mask_ref = None
    for batch_x, batch_y, batch_start_idx in loader:
        if limit_batches and n_batches >= limit_batches:
            break
        n_batches += 1
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        qid = _query_id(batch_start_idx)
        bsz = batch_x.size(0)
        neg_inf = float('-inf')

        per_ch = {'global10_mse': [], 'global30_mse': [], 'globalU_mse': [], 'block_mse': [],
                 'b1_global_mse': [], 'b1_oracle_mse': [], 'b2_global_mse': [], 'b2_oracle_mse': [],
                 'b3_global_mse': [], 'b3_oracle_mse': [],
                 'overlap_g_b1': [], 'overlap_g_b2': [], 'overlap_g_b3': [],
                 'overlap_b1_b2': [], 'overlap_b1_b3': [], 'overlap_b2_b3': [],
                 'unique_u': [],
                 'rank_b3_of_b1': [], 'rank_b1_of_b3': [], 'rank_g_of_b1': [], 'rank_g_of_b3': []}

        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            u_g = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            u_g = u_g.masked_fill(~cand_mask, neg_inf)
            u_blocks = {}
            for name, (lo, hi) in BLOCKS.items():
                u = individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi],
                                               chunk_size)
                u_blocks[name] = u.masked_fill(~cand_mask, neg_inf)

            s_g10 = stable_topk_indices(u_g, top_k, largest=True)
            s_g30 = stable_topk_indices(u_g, min(TOP_K_UNION_CAP, u_g.size(-1)), largest=True)
            s_b1 = stable_topk_indices(u_blocks['block1'], top_k, largest=True)
            s_b2 = stable_topk_indices(u_blocks['block2'], top_k, largest=True)
            s_b3 = stable_topk_indices(u_blocks['block3'], top_k, largest=True)

            union_ids = torch.cat([s_b1, s_b2, s_b3], dim=1)  # [B, 30]
            sorted_union, _ = union_ids.sort(dim=1)
            is_dup = torch.zeros_like(sorted_union, dtype=torch.bool)
            is_dup[:, 1:] = sorted_union[:, 1:] == sorted_union[:, :-1]
            unique_count = (~is_dup).sum(dim=1)  # U(q,c), 10..30

            def overlap(a, b):
                eq = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)
                return eq.float().sum(dim=-1) / top_k

            ov_g_b1 = overlap(s_g10, s_b1)
            ov_g_b2 = overlap(s_g10, s_b2)
            ov_g_b3 = overlap(s_g10, s_b3)
            ov_b1_b2 = overlap(s_b1, s_b2)
            ov_b1_b3 = overlap(s_b1, s_b3)
            ov_b2_b3 = overlap(s_b2, s_b3)

            rank_b3_of_b1 = _rank_of(u_blocks['block3'], cand_mask, s_b1).float().mean(dim=1)
            rank_b1_of_b3 = _rank_of(u_blocks['block1'], cand_mask, s_b3).float().mean(dim=1)
            rank_g_of_b1 = _rank_of(u_g, cand_mask, s_b1).float().mean(dim=1)
            rank_g_of_b3 = _rank_of(u_g, cand_mask, s_b3).float().mean(dim=1)

            yhat_g10 = _gather_mean(memory_c, offset_c, s_g10, 0, 720)
            yhat_g30 = _gather_mean(memory_c, offset_c, s_g30, 0, 720)
            yhat_gu = _gather_mean_masked(memory_c, offset_c, s_g30, unique_count, 0, 720)

            yb1 = _gather_mean(memory_c, offset_c, s_b1, *BLOCKS['block1'])
            yb2 = _gather_mean(memory_c, offset_c, s_b2, *BLOCKS['block2'])
            yb3 = _gather_mean(memory_c, offset_c, s_b3, *BLOCKS['block3'])
            yhat_block = torch.cat([yb1, yb2, yb3], dim=1)

            global10_mse = _mse(yhat_g10, query_future)
            global30_mse = _mse(yhat_g30, query_future)
            globalU_mse = _mse(yhat_gu, query_future)
            block_mse = _mse(yhat_block, query_future)

            yg_b1 = _gather_mean(memory_c, offset_c, s_g10, *BLOCKS['block1'])
            yg_b2 = _gather_mean(memory_c, offset_c, s_g10, *BLOCKS['block2'])
            yg_b3 = _gather_mean(memory_c, offset_c, s_g10, *BLOCKS['block3'])
            b1_global_mse = _mse(yg_b1, query_future[:, 0:96])
            b1_oracle_mse = _mse(yb1, query_future[:, 0:96])
            b2_global_mse = _mse(yg_b2, query_future[:, 96:336])
            b2_oracle_mse = _mse(yb2, query_future[:, 96:336])
            b3_global_mse = _mse(yg_b3, query_future[:, 336:720])
            b3_oracle_mse = _mse(yb3, query_future[:, 336:720])

            vals = dict(
                global10_mse=global10_mse, global30_mse=global30_mse, globalU_mse=globalU_mse,
                block_mse=block_mse, b1_global_mse=b1_global_mse, b1_oracle_mse=b1_oracle_mse,
                b2_global_mse=b2_global_mse, b2_oracle_mse=b2_oracle_mse,
                b3_global_mse=b3_global_mse, b3_oracle_mse=b3_oracle_mse,
                overlap_g_b1=ov_g_b1, overlap_g_b2=ov_g_b2, overlap_g_b3=ov_g_b3,
                overlap_b1_b2=ov_b1_b2, overlap_b1_b3=ov_b1_b3, overlap_b2_b3=ov_b2_b3,
                unique_u=unique_count.float(),
                rank_b3_of_b1=rank_b3_of_b1, rank_b1_of_b3=rank_b1_of_b3,
                rank_g_of_b1=rank_g_of_b1, rank_g_of_b3=rank_g_of_b3,
            )
            vals_cpu = {k: v.cpu() for k, v in vals.items()}
            for k, v in vals_cpu.items():
                per_ch[k].append(v)
            for b in range(bsz):
                crow = {'query_id': int(qid[b]), 'channel': c, 'split': split}
                crow.update({k: float(v[b]) for k, v in vals_cpu.items()})
                channel_rows.append(crow)

        for b in range(bsz):
            row = {'query_id': int(qid[b]), 'split': split}
            for k in per_ch:
                row[k] = float(torch.stack([per_ch[k][ci][b] for ci in range(len(channels))]).mean())
            row['delta_block_vs_global'] = row['global10_mse'] - row['block_mse']
            row['relative_gain_block_vs_global'] = (
                row['delta_block_vs_global'] / row['global10_mse'] if row['global10_mse'] > 1e-12 else float('nan'))
            row['block1_regret'] = row['b1_global_mse'] - row['b1_oracle_mse']
            row['block2_regret'] = row['b2_global_mse'] - row['b2_oracle_mse']
            row['block3_regret'] = row['b3_global_mse'] - row['b3_oracle_mse']
            rows.append(row)

    return rows, channel_rows


def query_order_hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(int(r['query_id']).to_bytes(8, 'big', signed=True))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-HEADROOM01')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--splits', default='train,val,test')
    cli = ap.parse_args()

    if cli.pred_len != 720:
        raise SystemExit('[ISSUE][ABORT] this diagnostic is defined only for pred_len=720 '
                         '(spec section 3 fixed block boundaries)')

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': 16,
        'seed': 0, 'top_k': cli.top_k, 'tau_topk': 0.1,
    })
    channels = list(range(int(args.enc_in)))

    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    import csv

    all_rows = {}
    for split in cli.splits.split(','):
        rows, channel_rows = run_split(exp, args, split, channels, top_k=cli.top_k,
                                       chunk_size=cli.chunk_size, limit_batches=cli.limit_batches)
        all_rows[split] = rows
        keys = sorted({k for r in rows for k in r})
        with open(cell_dir / f'query_level_{split}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        ckeys = sorted({k for r in channel_rows for k in r})
        with open(cell_dir / f'channel_level_{split}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=ckeys)
            w.writeheader()
            for r in channel_rows:
                w.writerow(r)
        (cell_dir / f'query_order_hash_{split}.json').write_text(json.dumps({
            'split': split, 'n_queries': len(rows), 'hash': query_order_hash(rows),
        }, indent=2))
        print(f'[horizon_retrieval_headroom01] {cli.cell}/{split}: {len(rows)} queries written')

    reference_hash = hashlib.sha256(Path(cli.reference_ckpt).read_bytes()).hexdigest()
    (cell_dir / 'config_fingerprint.json').write_text(json.dumps({
        'exp': 'TRACK-A-HORIZON-RETRIEVAL-HEADROOM01', 'cell': cli.cell, 'pred_len': cli.pred_len,
        'top_k': cli.top_k, 'channels': channels, 'blocks': BLOCKS,
        'reference_ckpt': cli.reference_ckpt, 'reference_ckpt_sha256': reference_hash,
        'value_reconstruction': 'scripts.train_margutil01.memory_value (unmodified)',
        'utility_fn': 'scripts.train_factorial_e2e01.individual_utility_memsafe (unmodified, '
                     'called with time-sliced inputs for block utilities)',
    }, indent=2))
    print(f'[horizon_retrieval_headroom01] {cli.cell} done.')


if __name__ == '__main__':
    main()
