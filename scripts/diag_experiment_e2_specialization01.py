#!/usr/bin/env python3
"""Experiment E, E2 post-training diagnostics (spec sections 9 & 10).

Loads a trained E2 checkpoint (frozen, no further training) and computes,
for a given split:
  1. Retrieval Specialization Matrix: each student subspace's hard Top-10
     picks, scored under EVERY component's own teacher distance (not just
     its own) -- off-diagonal entries should be worse than the diagonal if
     specialization is real.
  2. Top-10 candidate overlap (Jaccard) between every pair of subspaces.
  3. E3 Equal-Weight Mixture: s_equal = alpha*s_shared + (1-alpha)/3*(s_local
     +s_trend+s_seasonal), alpha=0.5, hard Top-10 under s_equal, scored by
     overall (shared) MSE -- compared against Shared-only and existing
     single-embedding baseline.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_experiment_e_teacher01 import COMPONENTS, component_distance_memsafe
from scripts.train_experiment_e2_multisubspace01 import cosine_score, split_normalize, subspace_sizes
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_margutil01 import build_experiment, memory_value

ALPHA = 0.5


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out_dir', default='results/EXPERIMENT-E-E2')
    cli = ap.parse_args()

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.checkpoint, map_location='cpu')

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    sizes = subspace_sizes(int(args.d_model))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    _, loader = exp._get_data(flag=cli.split, shuffle=False)

    # spec_matrix[subspace][teacher] = sum of retmse (picks under `subspace`'s
    # own score, scored by `teacher`'s distance)
    spec_sum = {sub: {t: 0.0 for t in COMPONENTS} for sub in COMPONENTS}
    overlap_sum = {f'{a}_vs_{b}': 0.0 for a, b in itertools.combinations(COMPONENTS, 2)}
    equal_mse_sum, shared_mse_sum = 0.0, 0.0
    n_rows = 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        for c in channels:
            z_q_raw = encode_raw(model, batch_x, c)
            z_k_raw = encode_raw(model, exp.memory_x, c)
            zq = split_normalize(z_q_raw, sizes)
            zk = split_normalize(z_k_raw, sizes)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            d_by_comp, s_by_comp, picks_by_comp = {}, {}, {}
            for comp in COMPONENTS:
                d_by_comp[comp] = component_distance_memsafe(memory_c, offset_c, query_future, comp, period,
                                                              cli.chunk_size).masked_fill(~cand_mask, float('inf'))
                s_e = cosine_score(zq[comp], zk[comp]).masked_fill(~cand_mask, float('-inf'))
                s_by_comp[comp] = s_e
                picks_by_comp[comp] = stable_topk_indices(s_e, cli.top_k, largest=True)

            for sub in COMPONENTS:
                for teacher in COMPONENTS:
                    val = d_by_comp[teacher].gather(1, picks_by_comp[sub]).mean(-1).sum()
                    spec_sum[sub][teacher] += float(val)

            for a, b in itertools.combinations(COMPONENTS, 2):
                inter = (picks_by_comp[a].unsqueeze(-1) == picks_by_comp[b].unsqueeze(-2)).any(-1).float().sum(-1)
                union = float(cli.top_k) * 2 - inter
                overlap_sum[f'{a}_vs_{b}'] += float((inter / union.clamp_min(1e-9)).sum())

            s_equal = ALPHA * s_by_comp['shared'] + (1 - ALPHA) / 3.0 * (
                s_by_comp['local'] + s_by_comp['trend'] + s_by_comp['seasonal'])
            s_equal = s_equal.masked_fill(~cand_mask, float('-inf'))
            picks_equal = stable_topk_indices(s_equal, cli.top_k, largest=True)
            y_sel_equal = memory_c[picks_equal] + offset_c.view(-1, 1, 1)
            equal_mse_sum += float(((y_sel_equal.mean(dim=1) - query_future) ** 2).mean(dim=-1).sum())

            y_sel_shared = memory_c[picks_by_comp['shared']] + offset_c.view(-1, 1, 1)
            shared_mse_sum += float(((y_sel_shared.mean(dim=1) - query_future) ** 2).mean(dim=-1).sum())

            n_rows += batch_x.size(0)

    spec_matrix = {sub: {t: spec_sum[sub][t] / n_rows for t in COMPONENTS} for sub in COMPONENTS}
    overlap = {k: v / n_rows for k, v in overlap_sum.items()}
    equal_mean = equal_mse_sum / n_rows
    shared_mean = shared_mse_sum / n_rows

    result = {
        'cell': cli.cell, 'split': cli.split, 'n_rows': n_rows, 'top_k': cli.top_k, 'alpha_equal': ALPHA,
        'retrieval_specialization_matrix_mean_retmse': spec_matrix,
        'top10_jaccard_overlap': overlap,
        'shared_only_overall_agg_mse': shared_mean,
        'equal_mixture_overall_agg_mse': equal_mean,
        'equal_vs_shared_improve_pct': (shared_mean - equal_mean) / shared_mean * 100.0,
    }
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'e2_specialization_{cli.split}.json').write_text(json.dumps(result, indent=2))
    print(f"[e2_specialization] {cli.cell}/{cli.split}: shared_only={shared_mean:.6f} "
         f"equal_mixture={equal_mean:.6f} ({result['equal_vs_shared_improve_pct']:.2f}%)")
    for sub in COMPONENTS:
        row = {t: round(spec_matrix[sub][t], 4) for t in COMPONENTS}
        best_teacher = min(row, key=row.get)
        print(f"  {sub:9s} embedding picks -> best-scoring teacher: {best_teacher:9s} row={row}")
    print(f"  top10_jaccard={ {k: round(v,3) for k,v in overlap.items()} }")


if __name__ == '__main__':
    main()
