#!/usr/bin/env python3
"""TRACK-I-PCA-FUTURE-TEACHER01 -- Phase A: teacher-only diagnostic.

No encoder is trained or even evaluated here -- this compares candidate
RANKINGS produced directly by different teacher-relevance geometries
(raw future MSE vs several fixed PCA-projected future spaces), each
teacher's own Top-10 picks scored against the metric we actually care
about (raw future MSE), never against the teacher's own latent distance.

Reused UNMODIFIED: `individual_utility_memsafe`/`arm_score` (not used here,
no encoder) -- specifically `individual_utility_memsafe` for the raw
teacher's `d`, `normalized_teacher_prob`/`kl_loss`'s temperature
convention (`normalized_teacher_prob` reused verbatim for every arm's
probability distribution), `memory_value`/`build_experiment`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`).
`fit_pca`/`pca_distance_memsafe` (`pca_future_teacher01`, new this track).

No leakage: PCA is fit once per channel on `exp.memory_y`-derived
candidate futures, which are TRAIN-only by construction (the candidate
memory bank is built exclusively from the train split -- see
`exp._ensure_memory` / `RelationMemorySampler`). Val/test futures are
used only as query targets, never for fitting.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.pca_future_teacher01 import fit_pca, pca_distance_memsafe
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8

ARMS = [
    ('A0_raw', dict(kind='raw')),
    ('A1_pca_full', dict(kind='pca', max_dim=None, metric='l2')),
    ('A2_pca128', dict(kind='pca', max_dim=128, metric='l2')),
    ('A3_pca64', dict(kind='pca', max_dim=64, metric='l2')),
    ('A4_pca32', dict(kind='pca', max_dim=32, metric='l2')),
    ('A5_pca16', dict(kind='pca', max_dim=16, metric='l2')),
    ('A6_pca64_cosine', dict(kind='pca', max_dim=64, metric='cosine')),
]


def _rank_of(x):
    """Vectorized per-row rank (1=best/smallest), ties broken by index
    (stable) -- used only for the Spearman proxy below."""
    return x.argsort(dim=-1).argsort(dim=-1).float()


def spearman_proxy(d_a, d_b, valid_mask):
    """Row-wise Spearman correlation between two [B,N] distance matrices,
    computed over the masked-as-tied full row (masked entries pushed to
    +inf in BOTH before ranking, so they cluster identically at the end
    for both and do not spuriously drive correlation up or down --
    documented approximation, see script docstring; exact only among the
    entries both consider valid, exact if candidate_mask excludes few
    candidates per query, which is the common case in this codebase)."""
    a = d_a.masked_fill(~valid_mask, float('inf'))
    b = d_b.masked_fill(~valid_mask, float('inf'))
    ra = _rank_of(a)
    rb = _rank_of(b)
    ra_c = ra - ra.mean(-1, keepdim=True)
    rb_c = rb - rb.mean(-1, keepdim=True)
    num = (ra_c * rb_c).sum(-1)
    den = (ra_c.norm(dim=-1) * rb_c.norm(dim=-1)).clamp_min(EPS)
    return num / den


def topk_jaccard(idx_a, idx_b, k):
    hit = (idx_a.unsqueeze(-1) == idx_b.unsqueeze(-2)).any(-1).float().sum(-1)  # |A ∩ B|
    union = 2 * k - hit
    return hit / union.clamp_min(1)


def evaluate_split(exp, args, loader, channels, device, pca_bases, tau_t, top_k, chunk_size):
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums = {name: {} for name, _ in ARMS}
    n = 0
    raw_top10_cache = {}  # per (channel) not needed across batches; recomputed per batch
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            d_raw = -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            raw_oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), top_k, largest=False)

            for name, cfg in ARMS:
                if cfg['kind'] == 'raw':
                    d_t = d_raw
                else:
                    mean, comp = pca_bases[(name, c)]
                    d_t = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp,
                                               metric=cfg['metric'], chunk_size=chunk_size)
                d_t_masked = d_t.masked_fill(~cand_mask, float('inf'))
                teacher_idx = stable_topk_indices(d_t_masked, top_k, largest=False)
                teacher_idx11 = stable_topk_indices(d_t_masked, top_k + 1, largest=False)

                actual_raw_retmse10 = d_raw.gather(1, teacher_idx).mean(-1)
                recall10 = recall_at_k(teacher_idx, raw_oracle_idx, top_k)
                ndcg10 = ndcg_at_k(teacher_idx, d_raw, cand_mask, top_k)
                spearman = spearman_proxy(d_t, d_raw, cand_mask)

                y_sel = memory_c[teacher_idx] + offset_c.view(-1, 1, 1)
                uniform_agg = ((y_sel.mean(dim=1) - query_future) ** 2).mean(-1)

                p_t = normalized_teacher_prob(d_t_masked, cand_mask, tau_t)
                n_valid = cand_mask.sum(-1).float()
                entropy = -(p_t.clamp_min(EPS) * p_t.clamp_min(EPS).log()).sum(-1)
                top1_mass = p_t.max(-1).values
                top10_mass = p_t.gather(1, teacher_idx).sum(-1)
                eff_support = 1.0 / (p_t ** 2).sum(-1).clamp_min(EPS)
                score_std = d_t.masked_fill(~cand_mask, 0.0).std(dim=-1)  # crude but cheap
                d10 = d_t_masked.gather(1, teacher_idx[:, -1:]).squeeze(-1)
                d11 = d_t_masked.gather(1, teacher_idx11[:, -1:]).squeeze(-1)
                margin_10_11 = (d11 - d10)
                jaccard_vs_raw = topk_jaccard(teacher_idx, raw_oracle_idx, top_k)

                row = dict(actual_raw_retmse10=actual_raw_retmse10, recall10=recall10, ndcg10=ndcg10,
                          spearman=spearman, uniform_agg=uniform_agg, entropy=entropy, top1_mass=top1_mass,
                          top10_mass=top10_mass, eff_support=eff_support, score_std=score_std,
                          margin_10_11=margin_10_11, jaccard_vs_raw=jaccard_vs_raw)
                for k_, v in row.items():
                    sums[name][k_] = sums[name].get(k_, 0.0) + float(v.sum())
        n += bsz * len(channels)
    return {name: {k_: v / max(n, 1) for k_, v in d.items()} for name, d in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--checkpoint', default='checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/TRACK-I-PCA-FUTURE-TEACHER01')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    exp, args = build_experiment(cli.checkpoint, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': 32, 'seed': 0,
        'patch_len': 120, 'stride': 120,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    channels = list(range(int(args.enc_in)))

    # Fit every PCA basis ONCE, per channel, on the train-only candidate
    # bank (memory_c is query-independent -- see script docstring).
    dummy_batch_x = exp.memory_x[:1].to(device)
    pca_bases = {}
    rank_report = {}
    for c in channels:
        memory_c, _ = memory_value(args, dummy_batch_x, exp.memory_y, exp.memory_x_last, c)
        for name, cfg in ARMS:
            if cfg['kind'] != 'pca':
                continue
            mean, comp = fit_pca(memory_c, max_dim=cfg['max_dim'])
            pca_bases[(name, c)] = (mean, comp)
            rank_report.setdefault(name, []).append(comp.shape[0])

    print('[diag_i_pca] PCA bases fit (train-only). Actual dim per channel:')
    for name, dims in rank_report.items():
        print(f'  {name}: {dims}')

    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name, loader in (('val', val_loader), ('test', test_loader)):
        print(f'[diag_i_pca] evaluating split={split_name} ...')
        res = evaluate_split(exp, args, loader, channels, device, pca_bases, cli.tau_t, cli.top_k, cli.chunk_size)
        results[split_name] = res
        for name, metrics in res.items():
            print(f'  [{split_name}] {name}: actual_raw_retmse10={metrics["actual_raw_retmse10"]:.6f} '
                 f'recall10={metrics["recall10"]:.4f} ndcg10={metrics["ndcg10"]:.4f} '
                 f'spearman={metrics["spearman"]:.4f} uniform_agg={metrics["uniform_agg"]:.6f} '
                 f'entropy={metrics["entropy"]:.3f} eff_support={metrics["eff_support"]:.1f} '
                 f'jaccard_vs_raw={metrics["jaccard_vs_raw"]:.4f}')

    (out_dir / 'phase_a_diagnostic.json').write_text(json.dumps(
        {'pca_actual_dims': rank_report, 'results': results}, indent=2))
    print(f'[diag_i_pca] wrote {out_dir / "phase_a_diagnostic.json"}')


if __name__ == '__main__':
    main()
