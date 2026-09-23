#!/usr/bin/env python3
"""Patch-level MoE feasibility, Stage 8: fixed-budget fusion (no future
information, no retraining -- loads the 4 already-trained
TRACK-A-PATCH-RETRIEVAL-EXPERT01 checkpoints simultaneously and combines
their FULL candidate-bank scores per query/channel).

All methods output exactly K=10 candidates (spec requirement). Temperature
is each arm's OWN training tau_s=0.1 (identical across all 4 arms per their
config_fingerprint -- reused as-is, never tuned on val/test).

Methods:
  - individual arms (reused from diag_patch_moe_feasibility01.py's cached
    summary, not recomputed here)
  - best_fixed (reused)
  - scale_oracle (reused)
  - uniform_prob_fusion: p_uniform(i) = mean_P softmax(s_P(q,i)/tau), top-10
    of p_uniform
  - reciprocal_rank_fusion: RRF(i) = sum_P 1/(60 + rank_P(i)), standard
    k=60 constant, top-10 of RRF score
  - mean_rank_fusion: top-10 of lowest mean_P(rank_P(i))

retMSE@10 for a fused Top-10 uses the SAME individual_utility_memsafe-based
distance already used everywhere else in this session -- no new math.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_argsort, stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('native_p16', 'p24', 'p48', 'p120')
PATCH_LEN = {'native_p16': 16, 'p24': 24, 'p48': 48, 'p120': 120}
RRF_K = 60
TAU = 0.1  # each arm's own training tau_s, identical across arms (config_fingerprint), not tuned here


def _load_arm(arm_name, cell, checkpoints_root, reference_ckpt, pred_len, device):
    ckpt_path = f'{checkpoints_root}/{cell}/{arm_name}/checkpoint.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    patch_len = PATCH_LEN[arm_name]
    exp, args = build_experiment(reference_ckpt, {
        'pred_len': pred_len, 'seq_len': pred_len, 'batch_size': 32, 'seed': 0,
        'patch_len': patch_len, 'stride': patch_len,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return exp, args, model


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
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.split
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {a: _load_arm(a, cli.cell, cli.checkpoints_root, cli.reference_ckpt, cli.pred_len, device)
           for a in ARMS}
    ref_exp = arms[ARMS[0]][0]
    _, loader = ref_exp._get_data(flag=cli.split, shuffle=False)
    channels = list(range(int(arms[ARMS[0]][1].enc_in)))

    sums = {m: 0.0 for m in ('uniform_prob_fusion', 'reciprocal_rank_fusion', 'mean_rank_fusion')}
    n_rows = 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            per_arm_s, per_arm_d, cand_mask_c = {}, None, None
            for a in ARMS:
                exp, args, model = arms[a]
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                z_q = encode_raw(model, batch_x, c)
                E = encode_raw(model, exp.memory_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                s = arm_score(z_q, E, None).masked_fill(~cand_mask, float('-inf'))
                per_arm_s[a] = s
                if per_arm_d is None:
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    per_arm_d = -u  # distance, same for every arm (same candidate futures/query future)
                    cand_mask_c = cand_mask

            # ---- uniform probability fusion ----
            p_sum = None
            for a in ARMS:
                logits = (per_arm_s[a] / TAU).masked_fill(~cand_mask_c, float('-inf'))
                p = torch.softmax(logits, dim=-1)
                p_sum = p if p_sum is None else p_sum + p
            p_uniform = p_sum / len(ARMS)
            picks_uf = stable_topk_indices(p_uniform, cli.top_k, largest=True)
            sums['uniform_prob_fusion'] += float(per_arm_d.gather(1, picks_uf).mean(-1).sum())

            # ---- reciprocal rank fusion + mean-rank fusion ----
            rrf_sum, rank_sum = None, None
            for a in ARMS:
                order = stable_argsort(-per_arm_s[a].masked_fill(~cand_mask_c, float('inf')), dim=-1)
                rank = torch.empty_like(order)
                ar = torch.arange(order.size(-1), device=device).unsqueeze(0).expand_as(order)
                rank.scatter_(1, order, ar)
                rrf_term = 1.0 / (RRF_K + rank.float() + 1.0)
                rrf_term = rrf_term.masked_fill(~cand_mask_c, float('-inf'))
                rank_masked = rank.float().masked_fill(~cand_mask_c, float('inf'))
                rrf_sum = rrf_term if rrf_sum is None else rrf_sum + rrf_term
                rank_sum = rank_masked if rank_sum is None else rank_sum + rank_masked
            picks_rrf = stable_topk_indices(rrf_sum, cli.top_k, largest=True)
            sums['reciprocal_rank_fusion'] += float(per_arm_d.gather(1, picks_rrf).mean(-1).sum())
            mean_rank = rank_sum / len(ARMS)
            picks_mr = stable_topk_indices(mean_rank, cli.top_k, largest=False)
            sums['mean_rank_fusion'] += float(per_arm_d.gather(1, picks_mr).mean(-1).sum())

        n_rows += bsz

    denom = n_rows * len(channels)
    result = {m: sums[m] / denom for m in sums}
    result['n_queries'] = n_rows
    result['n_channels'] = len(channels)
    result['tau_used'] = TAU
    result['rrf_k'] = RRF_K

    # merge with existing per-arm/oracle summary for a single comparison file
    summary_path = Path(cli.out_dir) / cli.cell / cli.split / 'summary.json'
    if summary_path.exists():
        base = json.loads(summary_path.read_text())
        result['best_fixed_arm'] = base['best_fixed_arm']
        result['best_fixed_mean'] = base['best_fixed_mean']
        result['scale_oracle_mean'] = base['scale_oracle_mean']
        for m in ('uniform_prob_fusion', 'reciprocal_rank_fusion', 'mean_rank_fusion'):
            result[f'{m}_vs_best_fixed_improve_pct'] = (
                (base['best_fixed_mean'] - result[m]) / base['best_fixed_mean'] * 100.0)

    (out_dir / 'fusion_summary.json').write_text(json.dumps(result, indent=2))
    print(f"[patch_moe_fusion] {cli.cell}/{cli.split}: " +
         ' '.join(f'{m}={result[m]:.6f}' for m in ('uniform_prob_fusion', 'reciprocal_rank_fusion', 'mean_rank_fusion')))
    if 'best_fixed_mean' in result:
        print(f"  best_fixed={result['best_fixed_mean']:.6f} scale_oracle={result['scale_oracle_mean']:.6f} " +
             ' '.join(f"{m}_improve={result[f'{m}_vs_best_fixed_improve_pct']:.2f}%"
                      for m in ('uniform_prob_fusion', 'reciprocal_rank_fusion', 'mean_rank_fusion')))


if __name__ == '__main__':
    main()
