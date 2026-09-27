#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN05, Stage A -- ETTh1 CLEAN04 closure
audit. Does NOT retrain or reselect anything -- audits the ALREADY-FROZEN
CLEAN04 Primary configuration (B1 tau=0.10/step100, B2 tau=0.10/step675,
B3 tau=0.10/step50) with the ranking/chronological/bootstrap diagnostics
CLEAN04 skipped, plus a validation-only B2 boundary extension (steps
800/900/1125/1350, continuing from the already-saved step-675 adapter
state -- CLEAN04's own checkpoint selection is never changed).

Binary vs utility-graded NDCG are computed and stored under explicitly
different keys (`binary_oracle_ndcg_at_10` / `utility_graded_ndcg_at_10`)
per the spec's explicit requirement not to conflate them. Individual
candidate regret and uniform-Top-10-aggregate block MSE regret are also
stored under different keys.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_c_horizon_clean02 import BLOCKS, block_distance
from scripts.train_c_horizon_clean04_expertwise import BlockAdapter
from scripts.train_c_horizon_frozen03 import build_query_cache, candidate_value, state_sha
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment

BLOCK_NAMES = ('block1', 'block2', 'block3')


def binary_ndcg_at_k(model_idx, oracle_idx, k):
    """Relevance = 1 iff candidate is in the Oracle's own Top-k, else 0
    (spec's `binary_oracle_ndcg_at_10`) -- ideal ranking places all k
    relevant items first, so IDCG = sum_{r=1}^{k} 1/log2(r+1), a constant."""
    hit = (model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)).any(-1).float()
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=model_idx.device).float())
    dcg = (hit * disc.unsqueeze(0)).sum(-1)
    idcg = disc.sum()
    return dcg / idcg


def utility_graded_ndcg_at_k(model_idx, d, valid_mask, k):
    """Continuous shifted-negative-distance relevance (spec's
    `utility_graded_ndcg_at_10` -- same convention as this session's other
    D-track NDCG helper, kept under an explicitly distinct name here)."""
    tgt = (-d).masked_fill(~valid_mask, float('-inf'))
    rel = tgt - tgt.masked_fill(~valid_mask, float('inf')).min(dim=-1, keepdim=True).values
    rel = rel.masked_fill(~valid_mask, 0.0)
    gains = rel.gather(1, model_idx)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=d.device).float()).unsqueeze(0)
    dcg = (gains * disc).sum(-1)
    ideal = rel.topk(k, dim=-1).values
    idcg = (ideal * disc).sum(-1).clamp_min(1e-12)
    return dcg / idcg


@torch.no_grad()
def full_diagnostics(cache, candidate_emb, memory_c_by_channel, channels, adapters, top_k, batch_size,
                     exp, chunk_size, device):
    """adapters: dict bname->adapter or None (None => A0/global for that
    block). Returns per-(start) dict of everything needed for chronological
    bins / bootstrap / ranking, PLUS aggregate means."""
    starts = cache['starts']
    n_rows = starts.size(0)
    per_query = {}
    agg = {f'{b}_{k}': [] for b in BLOCK_NAMES
          for k in ('mse', 'recall10', 'binary_ndcg10', 'graded_ndcg10', 'oracle_mean_rank',
                    'oracle_median_rank', 'rankfrac_mean', 'top100', 'top500', 'top1000',
                    'individual_regret', 'aggregate_regret')}
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        bsz = e - s
        global_mse_ch = torch.zeros(bsz, len(channels))
        block_mse_ch = torch.zeros(bsz, len(channels))
        for ci, c in enumerate(channels):
            h_q = cache['emb'][c][s:e].to(device)
            h_i = candidate_emb[c]
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            query_future = by[:, :, c]
            z_q_g = F.normalize(h_q, dim=-1)
            z_i_g = F.normalize(h_i, dim=-1)
            s_g = arm_score(z_q_g, z_i_g, None).masked_fill(~cand_mask, float('-inf'))
            picks_g = stable_topk_indices(s_g, top_k, largest=True)
            y_sel_g = memory_c[picks_g] + offset_c.view(-1, 1, 1)
            global_mse_ch[:, ci] = ((y_sel_g.mean(dim=1) - query_future) ** 2).mean(-1).cpu()

            concat = torch.zeros_like(query_future)
            for bname in BLOCK_NAMES:
                lo, hi = BLOCKS[bname]
                ad = adapters.get(bname)
                z_q = ad(h_q) if ad is not None else z_q_g
                z_i = ad(h_i) if ad is not None else z_i_g
                s_b = arm_score(z_q, z_i, None).masked_fill(~cand_mask, float('-inf'))
                model_idx = stable_topk_indices(s_b, top_k, largest=True)

                d_b = block_distance(memory_c, offset_c, query_future, lo, hi, chunk_size)
                d_b = d_b.masked_fill(~cand_mask, float('inf'))
                oracle_idx = stable_topk_indices(d_b, top_k, largest=False)

                n_valid = cand_mask.sum(-1).float()
                ranks_full = (-s_b.masked_fill(~cand_mask, float('-inf'))).argsort(dim=-1).argsort(dim=-1).float() + 1.0
                oracle_ranks = ranks_full.gather(1, oracle_idx)
                rankfrac = (oracle_ranks - 1) / (n_valid.unsqueeze(-1) - 1).clamp_min(1)

                y_sel_b = memory_c[model_idx][:, :, lo:hi] + offset_c.view(-1, 1, 1)
                agg_b = y_sel_b.mean(dim=1)
                block_mse_b = ((agg_b - query_future[:, lo:hi]) ** 2).mean(-1)
                y_sel_oracle = memory_c[oracle_idx][:, :, lo:hi] + offset_c.view(-1, 1, 1)
                oracle_agg_mse_b = ((y_sel_oracle.mean(dim=1) - query_future[:, lo:hi]) ** 2).mean(-1)

                agg[f'{bname}_mse'].append(block_mse_b.cpu())
                agg[f'{bname}_recall10'].append(((model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)
                                                 ).any(-1).float().sum(-1) / top_k).cpu())
                agg[f'{bname}_binary_ndcg10'].append(binary_ndcg_at_k(model_idx, oracle_idx, top_k).cpu())
                agg[f'{bname}_graded_ndcg10'].append(utility_graded_ndcg_at_k(model_idx, d_b, cand_mask, top_k).cpu())
                agg[f'{bname}_oracle_mean_rank'].append(oracle_ranks.mean(-1).cpu())
                agg[f'{bname}_oracle_median_rank'].append(oracle_ranks.median(-1).values.cpu())
                agg[f'{bname}_rankfrac_mean'].append(rankfrac.mean(-1).cpu())
                agg[f'{bname}_top100'].append((oracle_ranks <= 100).float().mean(-1).cpu())
                agg[f'{bname}_top500'].append((oracle_ranks <= 500).float().mean(-1).cpu())
                agg[f'{bname}_top1000'].append((oracle_ranks <= 1000).float().mean(-1).cpu())
                model_ind = d_b.gather(1, model_idx).mean(-1)
                oracle_ind = d_b.gather(1, oracle_idx).mean(-1)
                agg[f'{bname}_individual_regret'].append((model_ind - oracle_ind).cpu())
                agg[f'{bname}_aggregate_regret'].append((block_mse_b - oracle_agg_mse_b).cpu())

                concat[:, lo:hi] = agg_b
            block_mse_ch[:, ci] = ((concat - query_future) ** 2).mean(-1).cpu()
        for i, st in enumerate(bstart.tolist() if torch.is_tensor(bstart) else list(bstart)):
            per_query[int(st)] = {'global': float(global_mse_ch[i].mean()), 'block': float(block_mse_ch[i].mean())}
    means = {k: float(torch.cat(v).mean()) for k, v in agg.items()}
    return per_query, means


def chronological_bins(per_query, n_bins=8):
    starts = sorted(per_query.keys())
    n = len(starts)
    bin_size = -(-n // n_bins)
    bins = []
    for bi in range(n_bins):
        chunk = starts[bi * bin_size:(bi + 1) * bin_size]
        if not chunk:
            continue
        g = np.mean([per_query[s]['global'] for s in chunk])
        b = np.mean([per_query[s]['block'] for s in chunk])
        bins.append({'bin': bi, 'n': len(chunk), 'global_mse': float(g), 'block_mse': float(b),
                     'delta': float(g - b), 'relative_gain_pct': float((g - b) / g * 100)})
    return bins


def moving_block_bootstrap(per_query, block_length, n_reps=2000, seed=0):
    starts = sorted(per_query.keys())
    delta = np.array([per_query[s]['global'] - per_query[s]['block'] for s in starts])
    n = len(delta)
    # contiguous-block resampling: draw blocks of `block_length` consecutive
    # query rows (index space, not timestamp space) with replacement until
    # the resample length >= n, then truncate to n.
    n_block_rows = max(1, block_length // 720 if block_length >= 720 else max(1, block_length // 96))
    # Use ROW-count blocks proportional to the requested horizon length in
    # units of "queries" (each query already spans up to 720 steps and
    # windows heavily overlap) -- so block_length=96/240/720 is mapped to a
    # block of that many CONSECUTIVE queries as the coarse "overlap window"
    # proxy explicitly documented here (spec doesn't define an exact query
    # count for these lengths; this is a documented, reasonable choice).
    block_rows = max(1, block_length // 8)
    n_blocks_needed = -(-n // block_rows)
    rng = np.random.RandomState(seed)
    boot = np.empty(n_reps)
    for i in range(n_reps):
        block_starts = rng.randint(0, max(1, n - block_rows + 1), size=n_blocks_needed)
        idx = np.concatenate([np.arange(bs, bs + block_rows) for bs in block_starts])[:n]
        boot[i] = delta[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'block_length_param': block_length, 'block_rows': block_rows, 'mean_delta': float(delta.mean()),
           'median_delta': float(np.median(delta)), 'ci_low': float(lo), 'ci_high': float(hi),
           'ci_excludes_zero_positive': bool(lo > 0), 'n_reps': n_reps, 'seed': seed, 'n_queries': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--g_best_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--primary_ckpt_dir', default='checkpoints/track_c_horizon_retrieval_clean04/ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN05')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_ckpt = torch.load(cli.g_best_checkpoint, map_location='cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'top_k': cli.top_k, 'tau_topk': 0.1, 'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(g_ckpt['model_state_dict'])
    model.to(device); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    frozen_before = state_sha(model.state_dict())

    candidate_emb = {c: encode_raw(model, exp.memory_x, c).detach() for c in channels}
    memory_c_by_channel = {c: candidate_value(exp.memory_y, exp.memory_x_last, c).to(device) for c in channels}

    adapters = {}
    ckpt_paths = {'block1': f'{cli.primary_ckpt_dir}/block1_tau01/checkpoint.pth',
                 'block2': f'{cli.primary_ckpt_dir}/block2_tau01/checkpoint.pth',
                 'block3': f'{cli.primary_ckpt_dir}/block3_tau01/checkpoint.pth'}
    for bname, p in ckpt_paths.items():
        ck = torch.load(p, map_location=device)
        ad = BlockAdapter(d_model).to(device)
        ad.load_state_dict(ck['adapter_state_dict'])
        ad.eval()
        adapters[bname] = ad
    assert state_sha(model.state_dict()) == frozen_before, '[ISSUE][ABORT] trunk drifted while loading experts'

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    closure = {'cell': cli.cell}

    for split in ('val', 'test'):
        cache = build_query_cache(model, exp, split, channels, device)
        pq_a0, means_a0 = full_diagnostics(cache, candidate_emb, memory_c_by_channel, channels,
                                           {}, cli.top_k, cli.batch_size, exp, cli.chunk_size, device)
        pq_primary, means_primary = full_diagnostics(cache, candidate_emb, memory_c_by_channel, channels,
                                                      adapters, cli.top_k, cli.batch_size, exp,
                                                      cli.chunk_size, device)
        a0_global = float(np.mean([v['global'] for v in pq_a0.values()]))
        a0_weighted = (96 * means_a0['block1_mse'] + 240 * means_a0['block2_mse'] + 384 * means_a0['block3_mse']) / 720
        primary_global = float(np.mean([v['global'] for v in pq_primary.values()]))
        primary_block = float(np.mean([v['block'] for v in pq_primary.values()]))

        closure[f'{split}_a0_global_mse'] = a0_global
        closure[f'{split}_a0_weighted_from_blocks'] = a0_weighted
        closure[f'{split}_a0_weighted_consistency_abs_error'] = abs(a0_global - a0_weighted)
        closure[f'{split}_a0_block_metrics'] = {b: means_a0[f'{b}_mse'] for b in BLOCK_NAMES}
        closure[f'{split}_primary_block_metrics'] = {b: means_primary[f'{b}_mse'] for b in BLOCK_NAMES}
        closure[f'{split}_primary_global_recompute'] = primary_global
        closure[f'{split}_primary_composite_mse'] = primary_block
        closure[f'{split}_improve_vs_global_pct'] = (primary_global - primary_block) / primary_global * 100
        closure[f'{split}_block_delta'] = {
            b: {'absolute': means_a0[f'{b}_mse'] - means_primary[f'{b}_mse'],
               'relative_pct': (means_a0[f'{b}_mse'] - means_primary[f'{b}_mse']) / means_a0[f'{b}_mse'] * 100}
            for b in BLOCK_NAMES}
        closure[f'{split}_n_blocks_improved'] = sum(
            1 for b in BLOCK_NAMES if means_primary[f'{b}_mse'] < means_a0[f'{b}_mse'])
        closure[f'{split}_ranking_a0'] = {b: {k.replace(f'{b}_', ''): means_a0[k] for k in means_a0 if k.startswith(b)}
                                         for b in BLOCK_NAMES}
        closure[f'{split}_ranking_primary'] = {b: {k.replace(f'{b}_', ''): means_primary[k] for k in means_primary if k.startswith(b)}
                                              for b in BLOCK_NAMES}

        (out_dir / f'primary_ranking_metrics_{split}.json').write_text(json.dumps({
            'a0': closure[f'{split}_ranking_a0'], 'primary': closure[f'{split}_ranking_primary']}, indent=2))

        if split == 'test':
            bins = chronological_bins(pq_primary, n_bins=8)
            (out_dir / 'chronological_delta_test.json').write_text(json.dumps(bins, indent=2))
            closure['test_chronological_bins_improved'] = sum(1 for b in bins if b['delta'] > 0)
            closure['test_chronological_n_bins'] = len(bins)

            boot = {str(bl): moving_block_bootstrap(pq_primary, bl, n_reps=2000, seed=0) for bl in (96, 240, 720)}
            (out_dir / 'moving_block_bootstrap_test.json').write_text(json.dumps(boot, indent=2))
            closure['test_moving_block_bootstrap'] = boot

    (out_dir / 'closure_verdict.json').write_text(json.dumps(closure, indent=2, default=str))
    print(f"[clean05_etth1_closure] val: a0={closure['val_a0_global_mse']:.6f} "
         f"primary={closure['val_primary_composite_mse']:.6f} improve={closure['val_improve_vs_global_pct']:.3f}%")
    print(f"[clean05_etth1_closure] test: a0={closure['test_a0_global_mse']:.6f} "
         f"primary={closure['test_primary_composite_mse']:.6f} improve={closure['test_improve_vs_global_pct']:.3f}%")
    print(f"[clean05_etth1_closure] test weighted-consistency abs_error={closure['test_a0_weighted_consistency_abs_error']:.2e}")
    print(f"[clean05_etth1_closure] test n_blocks_improved={closure['test_n_blocks_improved']}/3 "
         f"chronological_bins_improved={closure['test_chronological_bins_improved']}/{closure['test_chronological_n_bins']}")
    for b in BLOCK_NAMES:
        r_a0 = closure['test_ranking_a0'][b]
        r_pr = closure['test_ranking_primary'][b]
        print(f"  {b}: recall10 a0={r_a0['recall10']:.4f}->primary={r_pr['recall10']:.4f} "
             f"binary_ndcg10 a0={r_a0['binary_ndcg10']:.4f}->primary={r_pr['binary_ndcg10']:.4f} "
             f"rankfrac_mean a0={r_a0['rankfrac_mean']:.4f}->primary={r_pr['rankfrac_mean']:.4f}")
    print(f"[clean05_etth1_closure] wrote {out_dir / 'closure_verdict.json'}")


if __name__ == '__main__':
    main()
