#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN04 -- composite evaluation.

Loads the REAL saved per-block adapter checkpoints (not a hand-computed
weighted average of already-reported numbers) and evaluates the composite
Global-vs-Block comparison in one pass, plus the alpha residual sweep
(spec section 15) and chronological bin analysis (section 19). This is the
actual re-verification the spec explicitly demands instead of trusting
arithmetic on already-reported per-block numbers.
"""
import argparse
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
from scripts.train_c_horizon_clean02 import BLOCKS
from scripts.train_c_horizon_clean04_expertwise import BlockAdapter
from scripts.train_c_horizon_frozen03 import build_query_cache, candidate_value, state_sha
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment

BLOCK_NAMES = ('block1', 'block2', 'block3')
ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)


def load_block_adapter(ckpt_path, d_model, device):
    ck = torch.load(ckpt_path, map_location=device)
    ad = BlockAdapter(d_model).to(device)
    ad.load_state_dict(ck['adapter_state_dict'])
    ad.eval()
    return ad, ck


@torch.no_grad()
def per_query_composite(cache, candidate_emb, memory_c_by_channel, channels, block_score_fn, top_k,
                        batch_size, exp, device):
    """block_score_fn(bname, h_q, h_i) -> [B,N] score. Returns per-(start)
    row -> {'global': mse, 'block': mse, per-block mse, per-block picks
    reused for jaccard/replacement-rate}."""
    starts = cache['starts']
    n_rows = starts.size(0)
    rows_out = []
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        bsz = e - s
        global_mse_ch = torch.zeros(bsz, len(channels))
        block_mse_ch = torch.zeros(bsz, len(channels))
        per_block_mse_ch = {b: torch.zeros(bsz, len(channels)) for b in BLOCK_NAMES}
        jacc_ch = {b: torch.zeros(bsz, len(channels)) for b in BLOCK_NAMES}
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
                s_b = block_score_fn(bname, h_q, h_i).masked_fill(~cand_mask, float('-inf'))
                picks_b = stable_topk_indices(s_b, top_k, largest=True)
                y_sel_b = memory_c[picks_b][:, :, lo:hi] + offset_c.view(-1, 1, 1)
                agg_b = y_sel_b.mean(dim=1)
                per_block_mse_ch[bname][:, ci] = ((agg_b - query_future[:, lo:hi]) ** 2).mean(-1).cpu()
                concat[:, lo:hi] = agg_b
                inter = (picks_b.unsqueeze(-1) == picks_g.unsqueeze(-2)).any(-1).float().sum(-1)
                jacc_ch[bname][:, ci] = (inter / (2 * top_k - inter).clamp_min(1e-9)).cpu()
            block_mse_ch[:, ci] = ((concat - query_future) ** 2).mean(-1).cpu()
        rows_out.append({'start': bstart.tolist() if torch.is_tensor(bstart) else list(bstart),
                         'global': global_mse_ch.mean(-1).tolist(), 'block': block_mse_ch.mean(-1).tolist(),
                         **{f'{b}_mse': per_block_mse_ch[b].mean(-1).tolist() for b in BLOCK_NAMES},
                         **{f'{b}_jaccard': jacc_ch[b].mean(-1).tolist() for b in BLOCK_NAMES}})
    out = {}
    for r in rows_out:
        for i, st in enumerate(r['start']):
            out[int(st)] = {k: r[k][i] for k in r if k != 'start'}
    return out


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
        delta = g - b
        bins.append({'bin': bi, 'n': len(chunk), 'global_mse': float(g), 'block_mse': float(b),
                     'delta': float(delta), 'relative_gain_pct': float(delta / g * 100),
                     'frac_improved': float(np.mean([1.0 if per_query[s]['global'] > per_query[s]['block'] else 0.0
                                                     for s in chunk]))})
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--g_best_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--config', required=True, help='JSON: {block1:{tau,step,ckpt},...}')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--split', default='val', choices=('val', 'test'))
    ap.add_argument('--alpha_sweep', action='store_true')
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN04')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = json.loads(Path(cli.config).read_text())

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
    cache = build_query_cache(model, exp, cli.split, channels, device)
    assert state_sha(model.state_dict()) == frozen_before

    adapters = {}
    for bname in BLOCK_NAMES:
        ad, ck = load_block_adapter(config[bname]['ckpt'], d_model, device)
        adapters[bname] = ad
        assert abs(ck['step'] - config[bname]['step']) < 1e-9 or ck['step'] == config[bname]['step']

    def score_fn_alpha1(bname, h_q, h_i):
        return arm_score(adapters[bname](h_q), adapters[bname](h_i), None)

    per_query = per_query_composite(cache, candidate_emb, memory_c_by_channel, channels, score_fn_alpha1,
                                    cli.top_k, cli.batch_size, exp, device)
    global_mean = float(np.mean([v['global'] for v in per_query.values()]))
    block_mean = float(np.mean([v['block'] for v in per_query.values()]))
    per_block_means = {b: float(np.mean([v[f'{b}_mse'] for v in per_query.values()])) for b in BLOCK_NAMES}
    jaccards = {b: float(np.mean([v[f'{b}_jaccard'] for v in per_query.values()])) for b in BLOCK_NAMES}

    result = {'cell': cli.cell, 'split': cli.split, 'config': cli.out_tag,
             'global_h720_mse': global_mean, 'block_h720_mse': block_mean,
             'improve_vs_global_pct': (global_mean - block_mean) / global_mean * 100,
             'per_block_mse': per_block_means, 'global_vs_block_jaccard': jaccards,
             'config_detail': {b: {'tau': config[b]['tau'], 'step': config[b]['step']} for b in BLOCK_NAMES}}

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'composite_{cli.out_tag}_{cli.split}.json').write_text(json.dumps(result, indent=2))
    print(f'[eval_composite] {cli.out_tag}/{cli.split}: global={global_mean:.6f} block={block_mean:.6f} '
         f'improve={result["improve_vs_global_pct"]:.3f}%')

    if cli.split == 'val':
        bins = chronological_bins(per_query, n_bins=8)
        (out_dir / f'chronological_delta_val_{cli.out_tag}.json').write_text(json.dumps(bins, indent=2))
        n_improved_bins = sum(1 for b in bins if b['delta'] > 0)
        print(f'[eval_composite] chronological bins improved: {n_improved_bins}/{len(bins)}')
        for b in bins:
            print(f"  bin{b['bin']}: global={b['global_mse']:.4f} block={b['block_mse']:.4f} "
                 f"delta={b['delta']:.4f} rel_gain={b['relative_gain_pct']:.2f}%")

    if cli.alpha_sweep and cli.split == 'val':
        alpha_results = {}
        for bname in BLOCK_NAMES:
            alpha_results[bname] = {}
            for alpha in ALPHAS:
                def score_fn_alpha(bn, h_q, h_i, _bname=bname, _alpha=alpha):
                    z_q_g = F.normalize(h_q, dim=-1); z_i_g = F.normalize(h_i, dim=-1)
                    s_g = arm_score(z_q_g, z_i_g, None)
                    if bn != _bname:
                        return s_g
                    s_a = arm_score(adapters[bn](h_q), adapters[bn](h_i), None)
                    return (1 - _alpha) * s_g + _alpha * s_a
                pq = per_query_composite(cache, candidate_emb, memory_c_by_channel, channels, score_fn_alpha,
                                         cli.top_k, cli.batch_size, exp, device)
                bm = float(np.mean([v[f'{bname}_mse'] for v in pq.values()]))
                alpha_results[bname][alpha] = bm
            best_alpha = min(ALPHAS, key=lambda a: alpha_results[bname][a])
            print(f'[eval_composite] alpha sweep {bname}: {alpha_results[bname]} best_alpha={best_alpha}')
        (out_dir / 'alpha_sweep_metrics.json').write_text(json.dumps(alpha_results, indent=2))


if __name__ == '__main__':
    main()
