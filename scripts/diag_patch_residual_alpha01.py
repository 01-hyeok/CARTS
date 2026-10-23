#!/usr/bin/env python3
"""TRACK-PATCH-REPRESENTATION-RESIDUAL01 -- PHASE B: Residual-V5,
evaluation-only, fixed alpha sweep. No training. Reads the EXISTING
T-V5 selected checkpoint (read-only) and computes, for several fixed
alpha values, on the VALIDATION split only:

    r_q = normalize(mean_p h_p)                 -- PHASE A's winning representation
    q_m(alpha) = normalize(g_m + alpha * r_q)    -- g_m = T-V5's own SlotHeads(z_cls)

alpha=0 must reproduce T-V5's own scores exactly (asserted, not just
reported). Candidate branch (pooled P100) is untouched.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import slot_mechanism_diagnostics
from scripts.train_temporal_query_heads01 import union_oracle_recall_curve
from scripts.train_transformer_patch_query_heads01 import build_transformer_model, encode_raw_with_tokens
from utils.candidate_pool import (
    encode_pooled_candidates, gather_candidate_histories, gather_candidate_values, pooled_future_mse,
)
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)


@torch.no_grad()
def run_cell(cli, device):
    exp, args, model = build_transformer_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    assert model.encoder.patch_embed.num_patches == cli.expected_num_patches

    slot_heads = SlotHeads(int(args.d_model), n_slots=5).to(device)
    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt', expected_meta)

    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag='val', shuffle=False)

    acc = {a: {'ret': [], 'agg': [], 'recall': [], 'ndcg': [], 'D': [], 'C': [],
              'top1': [], 'top10': [], 'union_curves': [], 'union_size': []}
          for a in ALPHAS}
    parity_max_abs_diff = 0.0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        for c in channels:
            z_q, h_patches = encode_raw_with_tokens(model, batch_x, c)
            g = slot_heads(z_q)  # [B,5,D]
            r_q = F.normalize(h_patches.mean(dim=1), dim=-1)  # [B,D]

            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)
            pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
            z_k = encode_pooled_candidates(lambda x, cc: encode_raw(model, x, cc), pooled_x, c)
            z_k = F.normalize(z_k, dim=-1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)
            valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)

            # reference T-V5 scores (alpha=0, for parity check only)
            g_norm = F.normalize(g, dim=-1)
            ref_scores = torch.einsum('bsd,bmd->bsm', g_norm, z_k)

            for alpha in ALPHAS:
                q_m = F.normalize(g + alpha * r_q.unsqueeze(1), dim=-1)
                scores = torch.einsum('bsd,bmd->bsm', q_m, z_k)
                if alpha == 0.0:
                    diff = float((scores - ref_scores).abs().max())
                    parity_max_abs_diff = max(parity_max_abs_diff, diff)

                picks, p_heads, p_mix = mean_mixture_topk_selection(scores, valid_mask, cli.tau_s, k=TOP_K)
                ind_mse_i = d_pool.gather(1, picks)
                y_sel = pooled_memory_c.gather(
                    1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
                r_c = y_sel.mean(dim=1)
                agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
                D_ = ind_mse_i.mean(dim=-1) / TOP_K
                C_ = agg_mse - D_
                recall = recall_at_k(picks, oracle_idx, TOP_K)
                ndcg = ndcg_at_k(picks, d_pool, valid_mask, TOP_K)

                mech = slot_mechanism_diagnostics(scores, valid_mask, p_heads, cli.tau_s)
                curve, usize = union_oracle_recall_curve(scores, valid_mask, oracle_idx, TOP_K)

                a = acc[alpha]
                a['ret'].append(ind_mse_i.mean(dim=-1).cpu())
                a['agg'].append(agg_mse.cpu())
                a['recall'].append(recall.cpu())
                a['ndcg'].append(ndcg.cpu())
                a['D'].append(D_.cpu())
                a['C'].append(C_.cpu())
                a['top1'].append(mech['slot_top1_overlap'])
                a['top10'].append(mech['slot_top10_overlap'])
                a['union_curves'].append(curve)
                a['union_size'].append(usize)

    out = {'parity_max_abs_diff_alpha0': parity_max_abs_diff}
    for alpha in ALPHAS:
        a = acc[alpha]
        out[f'alpha_{alpha}'] = {
            'retmse10': float(torch.cat(a['ret']).mean()),
            'agg_mse10': float(torch.cat(a['agg']).mean()),
            'recall10': float(torch.cat(a['recall']).mean()),
            'ndcg10': float(torch.cat(a['ndcg']).mean()),
            'D': float(torch.cat(a['D']).mean()),
            'C': float(torch.cat(a['C']).mean()),
            'slot_top1_overlap': sum(a['top1']) / len(a['top1']),
            'slot_top10_overlap': sum(a['top10']) / len(a['top10']),
            'union_size_mean': sum(a['union_size']) / len(a['union_size']),
            'union_oracle_recall_by_n_heads': torch.tensor(a['union_curves']).mean(dim=0).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, required=True)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--expected_num_patches', type=int, default=6)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    if cli.stride is None:
        cli.stride = cli.patch_len
    cli.init_seed = cli.seed
    cli.top_k = TOP_K
    cli.batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[audit] cell={cli.cell} T-V5 checkpoint = {cli.v5_checkpoint}')
    out = run_cell(cli, device)
    assert out['parity_max_abs_diff_alpha0'] < 1e-5, (
        f'[ISSUE][ABORT] alpha=0 parity check failed: max abs diff = {out["parity_max_abs_diff_alpha0"]}')
    print(f'[phaseB parity] alpha=0 max abs score diff = {out["parity_max_abs_diff_alpha0"]:.2e} (PASS)')

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseB.json').write_text(json.dumps(out, indent=2))
    for alpha in ALPHAS:
        r = out[f'alpha_{alpha}']
        print(f'[phaseB] {cli.cell} alpha={alpha}: agg_mse10={r["agg_mse10"]:.6f} retmse10={r["retmse10"]:.6f} '
             f'top10_overlap={r["slot_top10_overlap"]:.4f} union_recall5={r["union_oracle_recall_by_n_heads"][-1]:.4f}')


if __name__ == '__main__':
    main()
