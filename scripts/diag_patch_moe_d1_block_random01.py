#!/usr/bin/env python3
"""Patch-level MoE feasibility, Stage 3 D1: block-random in-train holdout.

Tests whether the train-timeline native_p16 dominance (73.2% raw train,
72.1% deployment-matched/strict-causal -- candidate-support mismatch
already ruled out by the latter) reflects a genuine feature-target
relationship that generalizes to HELD-OUT samples from the SAME train
regime, or is itself just an in-sample/non-generalizing artifact.

Design (user-specified, NOT naive per-window random split -- ETTh1's
heavily overlapping sliding windows would leak near-duplicates between
train/holdout otherwise):
  1. Partition the train timeline into contiguous blocks of `--block_size`
     windows.
  2. Randomly label each block 'holdout' (prob=`--holdout_frac`) or 'train'
     (seed=0, reproducible) -- blocks interleave train/holdout/train/train/
     holdout/... as in the user's diagram, NOT a single chronological cut
     (that would just be deployment_matched01 again).
  3. For every holdout-block window (a D1 query), a train-block window is a
     valid candidate iff BOTH (a) it is in a 'train'-labeled block, AND
     (b) |candidate_start - query_start| >= seq_len + pred_len (the L+H
     gap) -- enforced per-pair directly, not just at block edges, so no
     overlapping window can leak into the candidate pool regardless of
     which block boundary it sits near.

Same frozen checkpoints, same encode_raw/arm_score/individual_utility_
memsafe/stable_topk_indices as the other patch_moe diagnostics -- only the
query subset and candidate mask differ.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('native_p16', 'p24', 'p48', 'p120')
PATCH_LEN = {'native_p16': 16, 'p24': 24, 'p48': 48, 'p120': 120}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--checkpoints_root', default='checkpoints/track_a_patch_retrieval_expert01')
    ap.add_argument('--out_dir', default='reports/patch_moe_feasibility')
    ap.add_argument('--block_size', type=int, default=720)
    ap.add_argument('--holdout_frac', type=float, default=0.2)
    ap.add_argument('--block_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / 'train_d1_block_random'
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = cli.pred_len + cli.pred_len  # seq_len == pred_len for these cells (L+H)

    per_arm_rows = {}
    ref_starts, ref_block_info = None, None
    for arm in ARMS:
        ckpt_path = f'{cli.checkpoints_root}/{cli.cell}/{arm}/checkpoint.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        patch_len = PATCH_LEN[arm]
        exp, args = build_experiment(cli.reference_ckpt, {
            'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
            'patch_len': patch_len, 'stride': patch_len,
            'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
        })
        exp._ensure_memory()
        model = exp.model.module if hasattr(exp.model, 'module') else exp.model
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        channels = list(range(int(args.enc_in)))
        memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

        starts = exp.memory_sampler.starts  # numpy, chronologically sorted, contiguous
        n_total = len(starts)
        n_blocks = int(np.ceil(n_total / cli.block_size))
        block_id = np.arange(n_total) // cli.block_size
        rng = np.random.RandomState(cli.block_seed)
        # fixed COUNT (not independent Bernoulli per block -- with few blocks,
        # Bernoulli can trivially draw zero holdout blocks by chance)
        n_holdout_blocks = max(1, round(n_blocks * cli.holdout_frac))
        holdout_block_ids = rng.choice(n_blocks, size=n_holdout_blocks, replace=False)
        block_is_holdout = np.zeros(n_blocks, dtype=bool)
        block_is_holdout[holdout_block_ids] = True
        window_is_holdout = block_is_holdout[block_id]
        window_is_train_pool = ~window_is_holdout

        holdout_starts = starts[window_is_holdout]
        train_pool_starts = starts[window_is_train_pool]

        _, loader = exp._get_data(flag='train', shuffle=False)
        x_all, y_all, start_all = [], [], []
        for bx, by, bstart in loader:
            x_all.append(bx); y_all.append(by)
            start_all.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
        x_all, y_all, start_all = torch.cat(x_all), torch.cat(y_all), torch.cat(start_all)
        start_to_row = {int(s): i for i, s in enumerate(start_all.tolist())}
        rows = [start_to_row[int(s)] for s in holdout_starts]
        x_sub, y_sub, start_sub = x_all[rows], y_all[rows], start_all[rows]

        # per-candidate "is it in the train-labeled pool" boolean, aligned to `starts` order
        train_pool_bool = window_is_train_pool  # [n_total], same order as `starts` / exp.memory_x

        ind_mse_rows = []
        for s in range(0, x_sub.size(0), cli.batch_size):
            bx = x_sub[s:s + cli.batch_size].float().to(device)
            by = y_sub[s:s + cli.batch_size].float().to(device)
            bstart = start_sub[s:s + cli.batch_size].numpy()
            bsz = bx.size(0)
            gap_ok = np.abs(starts[None, :] - bstart[:, None]) >= gap  # [B, N]
            valid_np = gap_ok & train_pool_bool[None, :]
            cand_mask = torch.from_numpy(valid_np).to(device)
            ind_mse_c = torch.zeros(bsz, len(channels))
            for c in channels:
                z_q = encode_raw(model, bx, c)
                E = encode_raw(model, exp.memory_x, c)
                memory_c, offset_c = memory_value(args, bx, memory_y, memory_x_last, c)
                query_future = by[:, :, c]
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d = -u
                s_score = arm_score(z_q, E, None).masked_fill(~cand_mask, float('-inf'))
                picks = stable_topk_indices(s_score, cli.top_k, largest=True)
                ind_mse_c[:, c] = d.gather(1, picks).mean(-1).cpu()
            ind_mse_rows.append(ind_mse_c)
        per_arm_rows[arm] = torch.cat(ind_mse_rows)
        if ref_starts is None:
            ref_starts = start_sub
            ref_block_info = {'n_blocks': n_blocks, 'n_holdout_blocks': int(block_is_holdout.sum()),
                              'n_holdout_windows': int(window_is_holdout.sum()),
                              'n_train_pool_windows': int(window_is_train_pool.sum())}
        else:
            assert torch.equal(start_sub, ref_starts), f'{arm}: holdout query set mismatch across arms'
        avg_valid = valid_np.sum(axis=1).mean() if 'valid_np' in dir() else float('nan')
        print(f'[d1_block_random] {cli.cell}/{arm}: mean retMSE@10={float(per_arm_rows[arm].mean()):.6f} '
             f'(n_holdout_queries={x_sub.size(0)})')

    n_rows = ref_starts.size(0)
    n_ch = per_arm_rows[ARMS[0]].size(1)
    with open(out_dir / 'per_query_patch_metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_start_idx', 'channel'] + [f'retmse10_{a}' for a in ARMS] + ['winner_arm', 'winner_retmse10'])
        stacked = torch.stack([per_arm_rows[a] for a in ARMS], dim=0)
        winner_idx = stacked.argmin(dim=0)
        for qi in range(n_rows):
            start_val = int(ref_starts[qi])
            for c in range(n_ch):
                vals = [float(stacked[ai, qi, c]) for ai in range(len(ARMS))]
                w.writerow([start_val, c] + vals + [ARMS[int(winner_idx[qi, c])], min(vals)])

    means = {a: float(per_arm_rows[a].mean()) for a in ARMS}
    winner_share = {a: float((winner_idx == i).float().mean()) for i, a in enumerate(ARMS)}
    print(f'[d1_block_random] {cli.cell}: block_info={ref_block_info}')
    print(f'[d1_block_random] {cli.cell}: per_arm_means={means} winner_share={winner_share}')


if __name__ == '__main__':
    main()
