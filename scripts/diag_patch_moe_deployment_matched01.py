#!/usr/bin/env python3
"""Patch-level MoE feasibility, Stage 2 remedy: deployment-matched
pseudo-query construction.

Root cause confirmed (Stage 2 audit, `utils/relation_memory.py::
RelationMemorySampler.valid_indices`, mask_mode='raft'): raw TRAIN queries
get a ~2879-candidate self/overlap-exclusion window carved out of the
7,201-candidate bank (since a train query's own start IS in the candidate
bank), while VAL/TEST queries get ZERO exclusion (their start is never in
the train-only candidate bank, so all 7,201 candidates are always valid).
This is a genuine, code-level candidate-support mismatch between the split
used to build the router's teacher signal (train) and the split it is
evaluated on (val/test) -- not a bug, but a real distribution shift.

This script builds PSEUDO-queries entirely inside the train timeline that
mimic the val/test candidate-support pattern exactly: the LAST `--pseudo_frac`
fraction of train windows (chronologically) become pseudo-queries, and they
may only search pseudo-candidates from the EARLIER portion of train whose
own window (including its own future, `pred_len`) has fully completed
before the pseudo-query starts -- `strict_causal`: candidate_end <=
pseudo_query_start. No self-exclusion window needed (impossible for this
mask to ever include a query's own start), exactly matching how val/test
queries see the candidate bank.

Same frozen checkpoints, same encode_raw/arm_score/individual_utility_
memsafe/stable_topk_indices as `diag_patch_moe_feasibility01.py` -- only the
candidate mask and the query subset differ.
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
    ap.add_argument('--pseudo_frac', type=float, default=0.2,
                    help='fraction of chronologically-LAST train windows used as pseudo-queries')
    ap.add_argument('--batch_size', type=int, default=32)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / 'train_deploy'
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm_rows = {}
    ref_starts = None
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

        starts = exp.memory_sampler.starts  # numpy, chronologically sorted
        n_total = len(starts)
        n_pseudo = max(int(n_total * cli.pseudo_frac), 1)
        pseudo_idx = np.arange(n_total - n_pseudo, n_total)  # last fraction, chronological
        pseudo_starts = starts[pseudo_idx]
        candidate_end = starts + cli.pred_len + cli.pred_len  # seq_len==pred_len for this experiment's cells

        _, loader = exp._get_data(flag='train', shuffle=False)
        x_all, y_all, start_all = [], [], []
        for bx, by, bstart in loader:
            x_all.append(bx); y_all.append(by)
            start_all.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
        x_all, y_all, start_all = torch.cat(x_all), torch.cat(y_all), torch.cat(start_all)
        start_to_row = {int(s): i for i, s in enumerate(start_all.tolist())}
        rows = [start_to_row[int(s)] for s in pseudo_starts]
        x_sub, y_sub, start_sub = x_all[rows], y_all[rows], start_all[rows]

        ind_mse_rows = []
        for s in range(0, x_sub.size(0), cli.batch_size):
            bx = x_sub[s:s + cli.batch_size].float().to(device)
            by = y_sub[s:s + cli.batch_size].float().to(device)
            bstart = start_sub[s:s + cli.batch_size].numpy()
            bsz = bx.size(0)
            # strict_causal, deployment-matched mask: candidate fully in the past,
            # no self-exclusion window needed (impossible to self-match by construction)
            valid_np = candidate_end[None, :] <= bstart[:, None]
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
        else:
            assert torch.equal(start_sub, ref_starts), f'{arm}: pseudo-query start mismatch'
        print(f'[deploy_matched] {cli.cell}/{arm}: mean retMSE@10={float(per_arm_rows[arm].mean()):.6f} '
             f'(n_pseudo_queries={x_sub.size(0)}, avg_valid_candidates={valid_np.sum(axis=1).mean():.0f})')

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
    print(f'[deploy_matched] {cli.cell}: per_arm_means={means} winner_share={winner_share}')


if __name__ == '__main__':
    main()
