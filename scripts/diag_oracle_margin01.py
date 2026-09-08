#!/usr/bin/env python3
"""EXP-ORACLE-CHOICE01 pre-training diagnostic (interpretation-only, per
spec -- does NOT change D1's loss/training in response to what it finds):
how sharp is the Set Oracle's true top-1 vs top-2 (etc.) utility margin at
each teacher-forced step? If true top-1/top-2 are frequently near-tied,
that is recorded as a LIMITATION on one-hot Oracle-Choice CE in the final
report, not acted on here (no automatic switch to multi-positive/soft/KL
targets).

Reuses `run_sequence_dense`'s oracle-prefix dense-utility computation
(imported, not reimplemented) on a query subsample, frozen B0 encoder,
same ETTh1 H96 data/teacher cache EXP-ORACLE-CHOICE01 itself will use.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_margutil01 import build_experiment, encode, memory_value
from utils.dense_utility import candidate_weights, dense_utility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    overrides = {'is_training': 0, 'model_id': 'oracle_margin_diag', 'des': 'oracle_margin_diag',
                 'checkpoints': '/tmp/oracle_margin_diag', 'seed': 0, 'top_k': cli.top_k,
                 'stage1_residual_teacher': 0, 'stage1_query_base_conditioning': 0,
                 'stage1_candidate_residual_conditioning': 0, 'stage1_retrieval_metric': 'cosine',
                 'stage1_full_memory_gradient_mode': 'full_online'}
    exp, args = build_experiment(cli.base_ckpt, overrides)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    b0_ckpt = torch.load(cli.base_ckpt, map_location='cpu')
    model.load_state_dict(b0_ckpt['model_state_dict'], strict=True)
    model.eval()
    exp._ensure_memory()
    channels = list(model.target_channels())
    c = channels[0]
    k = int(cli.top_k)
    tau = float(args.tau_topk)

    teacher = torch.load(cli.teacher_cache, map_location='cpu')
    if teacher['meta']['top_k'] != k:
        raise ValueError(f"teacher cache top_k={teacher['meta']['top_k']} != --top_k={k}")

    _, loader = exp._get_data(flag=cli.split, shuffle=False)
    E = encode(model, exp.memory_x, c)

    by_step = {t: {'abs_margin': [], 'rel_margin': [], 'near_tie': [], 'top5_spread': []} for t in range(k)}
    seen = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in loader:
            if seen >= cli.n_queries:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, counts = exp._candidate_mask(batch_start_idx)
            q = encode(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)

            rows = [teacher['splits'][cli.split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
            teacher_idx = teacher['splits'][cli.split]['teacher_idx'][int(c)][rows].to(device)
            query_valid = teacher_idx[:, 0] != -1
            if not bool(query_valid.any()):
                continue
            take = min(int(query_valid.sum()), cli.n_queries - seen)
            idx_take = query_valid.nonzero(as_tuple=True)[0][:take]

            selected_mask = torch.zeros_like(cand_mask)
            for t in range(k):
                prefix = teacher_idx[:, :t].clamp_min(0)
                a_dense = dense_utility(prefix, w, futures, query_future, chunk_size=cli.chunk_size)
                u_target = -a_dense
                valid_now = cand_mask & ~selected_mask
                for b in idx_take.tolist():
                    valid_b = valid_now[b]
                    ut = u_target[b][valid_b]
                    if ut.numel() < 5:
                        continue
                    top5 = ut.topk(5).values
                    abs_m = float(top5[0] - top5[1])
                    denom = float(top5[0].abs().clamp_min(1e-8))
                    rel_m = abs_m / denom
                    spread = float(top5[0] - top5[-1])
                    by_step[t]['abs_margin'].append(abs_m)
                    by_step[t]['rel_margin'].append(rel_m)
                    by_step[t]['near_tie'].append(1.0 if abs_m < 0.01 * denom else 0.0)
                    by_step[t]['top5_spread'].append(spread)
                nxt = teacher_idx[:, t:t + 1].clamp_min(0)
                selected_mask = selected_mask.scatter(1, nxt, True)
            seen += take

    out = {}
    for t in range(k):
        d = by_step[t]
        n = len(d['abs_margin'])
        out[f't{t+1}'] = {
            'n': n,
            'abs_margin_mean': sum(d['abs_margin']) / max(n, 1),
            'rel_margin_mean': sum(d['rel_margin']) / max(n, 1),
            'near_tie_frac': sum(d['near_tie']) / max(n, 1),
            'top5_spread_mean': sum(d['top5_spread']) / max(n, 1),
        }
    out['n_queries'] = seen
    out['split'] = cli.split
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    for t in range(k):
        r = out[f't{t+1}']
        print(f"[oracle_margin_diag] t={t+1} n={r['n']} abs_margin_mean={r['abs_margin_mean']:.5f} "
              f"rel_margin_mean={r['rel_margin_mean']:.4f} near_tie_frac={r['near_tie_frac']:.3f} "
              f"top5_spread_mean={r['top5_spread_mean']:.5f}")
    print(f'[oracle_margin_diag] written to {cli.out}')


if __name__ == '__main__':
    main()
