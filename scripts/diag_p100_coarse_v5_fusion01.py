#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE A: does fusing the
coarse delta-last-cosine score (that built Shared-P100, then was
historically discarded) with canonical V5's own learned Mean-Mixture
score, inside the SAME fixed Shared-P100 candidate support, improve
P100-internal ranking over either score alone? No training -- both
scores come from EXISTING, frozen, read-only artifacts:

  - coarse score c_i: `utils.candidate_pool.compute_coarse_delta_last_cosine_scores`
    (UNMODIFIED, the exact function that built the pool), restricted
    post-hoc to the 100 already-selected pool candidates via gather
    (mathematically identical to computing it directly on just those
    100 -- cosine of one candidate's history does not depend on any
    other candidate).
  - V5 score: the canonical sign-fixed `TRACK-V-SHARED-TOP100-SIGNFIX01`
    checkpoint's own Mean-Mixture p_mix, computed WITHIN Shared-P100
    exactly as production inference already does.

Fusion (spec section 7): log p_fused = (1-beta)*log(p_c+eps) +
beta*log(p_v5+eps), renormalized via softmax (mathematically
equivalent to plain renormalization since p_c/p_v5 already sum to 1).
beta=0 recovers coarse-only (A0, asserted to match within 1e-4); beta=1
recovers V5-only (A1, asserted to match within 1e-4); 0<beta<1 is the
fusion arm (A2).

tau_c = 0.05, a ONE-TIME TRAIN-only calibration already performed and
recorded in `results/TRACK-P100-COARSE-TO-FINE-RETRIEVAL01/tau_c_calibration.json`
(`scripts/calibrate_p100_coarse_tau01.py`) -- never re-swept here.
tau_s = 0.1, the existing canonical V5 production temperature
(`selected_checkpoint.json` for every cell confirms tau_s=0.1).

Reuses UNMODIFIED: `encode_raw` (`train_factorial_e2e01`), `SlotHeads`
(`train_k_multislot_predictive_retrieval01`), `memory_value`
(`train_margutil01`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01`), `spearman_kendall`
(`diag_set_difficulty01`), `CandidatePoolCache` (`train_retriever_pool01`),
`stable_topk_indices` (`RelationStage1`), `build_model`
(`train_j_shared_encoder_drift01`), `compute_coarse_delta_last_cosine_scores`
(`utils.candidate_pool`).
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
from scripts.diag_set_difficulty01 import spearman_kendall
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TOP_K = 10
N_SLOTS = 5
TAU_C = 0.05  # one-time TRAIN-only calibration, see tau_c_calibration.json -- fixed, never swept here
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-12


def spearman_batched(a, b):
    """Fully vectorized Spearman rank correlation, a/b: [B,N] (all N
    candidates valid by construction -- Shared-P100 is never masked).
    Avoids the per-row Python loop + per-call `torch.Generator()`
    overhead of `diag_set_difficulty01.spearman_kendall` (which this
    script also uses, but ONLY once per batch for the one Kendall-tau
    figure requested by spec section 13 -- not for every Spearman
    correlation, which would make the full beta sweep prohibitively
    slow at P100/Weather-21-channel scale)."""
    ra = a.argsort(dim=-1).argsort(dim=-1).float()
    rb = b.argsort(dim=-1).argsort(dim=-1).float()
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    denom = (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)
    return (ra * rb).sum(dim=-1) / denom  # [B]


def _eval_arm(p_final, d_pool, pooled_future, query_future, oracle_idx, valid_mask):
    picks = stable_topk_indices(p_final, TOP_K, largest=True)
    ind_mse = d_pool.gather(1, picks)
    retmse = ind_mse.mean(dim=-1)
    y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
    r_c = y_sel.mean(dim=1)
    agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
    D_ = ind_mse.mean(dim=-1) / TOP_K
    C_ = agg_mse - D_
    recall = recall_at_k(picks, oracle_idx, TOP_K)
    ndcg = ndcg_at_k(picks, d_pool, valid_mask, TOP_K)
    return {'picks': picks, 'retmse': retmse, 'agg_mse': agg_mse, 'D': D_, 'C': C_,
           'recall': recall, 'ndcg': ndcg}


@torch.no_grad()
def run_cell_split(cli, device, split):
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    channels = list(range(int(args.enc_in)))
    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{split}.pt', expected_meta)
    _, loader = exp._get_data(flag=split, shuffle=False)

    agg = {beta: {'retmse': [], 'agg_mse': [], 'recall': [], 'ndcg': [], 'D': [], 'C': [],
                 'oracle_corr_spearman': []} for beta in BETAS}
    oracle_stats = {'oracle_retmse': [], 'oracle_agg_mse': []}
    disagreement = {'top1_agree': [], 'top10_overlap': [], 'spearman_c_v5': [], 'kendall_c_v5': [],
                    'spearman_c_oracle': [], 'spearman_v5_oracle': []}
    endpoint_check = {'beta0_vs_coarse_maxdiff': [], 'beta1_vs_v5_maxdiff': []}

    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)  # [N,D]
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            bsz = batch_x.size(0)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)  # [B,100]

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)  # [B,N]
            c_i = coarse_full.gather(1, pool_idx_global)  # [B,100]
            p_c = F.softmax(c_i / TAU_C, dim=-1)

            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)  # [B,5,D]
            k_pool = k_full[pool_idx_global]  # [B,100,D]
            scores = torch.einsum('bsd,bmd->bsm', q, k_pool)  # [B,5,100]
            p_heads = F.softmax(scores / cli.tau_s, dim=-1)
            p_v5 = p_heads.mean(dim=1)  # [B,100]

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)  # [B,100,H]
            d_pool = ((pooled_future - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,100]
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)
            valid_mask = torch.ones_like(c_i, dtype=torch.bool)

            oracle_y = pooled_future.gather(1, oracle_idx.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
            oracle_agg_mse = ((oracle_y.mean(dim=1) - query_future) ** 2).mean(dim=-1)
            oracle_retmse = d_pool.gather(1, oracle_idx).mean(dim=-1)
            oracle_stats['oracle_retmse'].append(oracle_retmse.cpu())
            oracle_stats['oracle_agg_mse'].append(oracle_agg_mse.cpu())

            # ---- disagreement metrics (beta-independent) ----
            top10_c = stable_topk_indices(p_c, TOP_K, largest=True)
            top10_v5 = stable_topk_indices(p_v5, TOP_K, largest=True)
            overlap10 = (top10_c.unsqueeze(-1) == top10_v5.unsqueeze(-2)).any(-1).float().sum(-1) / TOP_K
            top1_agree = (top10_c[:, 0] == top10_v5[:, 0]).float()
            disagreement['top10_overlap'].append(overlap10.cpu())
            disagreement['top1_agree'].append(top1_agree.cpu())
            disagreement['spearman_c_v5'].append(spearman_batched(p_c, p_v5).cpu())
            disagreement['spearman_c_oracle'].append(spearman_batched(p_c, -d_pool).cpu())
            disagreement['spearman_v5_oracle'].append(spearman_batched(p_v5, -d_pool).cpu())
            # Kendall tau (coarse vs V5) ONLY -- the one per-row, sampled-pair
            # helper call per batch (not replicated across the beta sweep)
            kt_list = [spearman_kendall(p_c[b], p_v5[b], valid_mask[b])[1] for b in range(bsz)]
            disagreement['kendall_c_v5'].append(torch.tensor(kt_list))

            # ---- fusion sweep ----
            for beta in BETAS:
                raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
                p_fused = F.softmax(raw, dim=-1)
                if beta == 0.0:
                    endpoint_check['beta0_vs_coarse_maxdiff'].append(float((p_fused - p_c).abs().max()))
                if beta == 1.0:
                    endpoint_check['beta1_vs_v5_maxdiff'].append(float((p_fused - p_v5).abs().max()))
                r = _eval_arm(p_fused, d_pool, pooled_future, query_future, oracle_idx, valid_mask)
                a = agg[beta]
                a['retmse'].append(r['retmse'].cpu()); a['agg_mse'].append(r['agg_mse'].cpu())
                a['recall'].append(r['recall'].cpu()); a['ndcg'].append(r['ndcg'].cpu())
                a['D'].append(r['D'].cpu()); a['C'].append(r['C'].cpu())
                a['oracle_corr_spearman'].append(spearman_batched(p_fused, -d_pool).cpu())

    out = {'cell': cli.cell, 'split': split, 'tau_c': TAU_C, 'tau_s': cli.tau_s}
    out['oracle_retmse10'] = float(torch.cat(oracle_stats['oracle_retmse']).mean())
    out['oracle_agg_mse10'] = float(torch.cat(oracle_stats['oracle_agg_mse']).mean())
    out['disagreement'] = {
        'top1_agree': float(torch.cat(disagreement['top1_agree']).mean()),
        'top10_overlap': float(torch.cat(disagreement['top10_overlap']).mean()),
        'spearman_coarse_vs_v5': float(torch.cat(disagreement['spearman_c_v5']).nanmean()),
        'kendall_coarse_vs_v5': float(torch.cat(disagreement['kendall_c_v5']).nanmean()),
        'spearman_coarse_vs_oracle': float(torch.cat(disagreement['spearman_c_oracle']).nanmean()),
        'spearman_v5_vs_oracle': float(torch.cat(disagreement['spearman_v5_oracle']).nanmean()),
    }
    out['endpoint_sanity'] = {
        'beta0_vs_coarse_maxdiff': max(endpoint_check['beta0_vs_coarse_maxdiff']),
        'beta1_vs_v5_maxdiff': max(endpoint_check['beta1_vs_v5_maxdiff']),
    }
    out['by_beta'] = {}
    for beta in BETAS:
        a = agg[beta]
        out['by_beta'][str(beta)] = {
            'retmse10': float(torch.cat(a['retmse']).mean()), 'agg_mse10': float(torch.cat(a['agg_mse']).mean()),
            'recall10': float(torch.cat(a['recall']).mean()), 'ndcg10': float(torch.cat(a['ndcg']).mean()),
            'D': float(torch.cat(a['D']).mean()), 'C': float(torch.cat(a['C']).mean()),
            'spearman_vs_oracle': float(torch.cat(a['oracle_corr_spearman']).nanmean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--splits', nargs='+', default=['val'])
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = TOP_K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[audit] cell={cli.cell} V5 checkpoint = {cli.v5_checkpoint} tau_c={TAU_C} tau_s={cli.tau_s}')

    for split in cli.splits:
        out = run_cell_split(cli, device, split)
        assert out['endpoint_sanity']['beta0_vs_coarse_maxdiff'] < 1e-4, (
            f'[ISSUE][ABORT] beta=0 does not recover coarse-only: {out["endpoint_sanity"]}')
        assert out['endpoint_sanity']['beta1_vs_v5_maxdiff'] < 1e-4, (
            f'[ISSUE][ABORT] beta=1 does not recover V5-only: {out["endpoint_sanity"]}')
        print(f'[sanity] endpoint check PASS: {out["endpoint_sanity"]}')
        (out_dir / f'{cli.cell}_{split}_phaseA_fusion.json').write_text(json.dumps(out, indent=2))
        for beta in BETAS:
            r = out['by_beta'][str(beta)]
            print(f'[diag_p100_fusion] {cli.cell}/{split}/beta={beta}: agg_mse10={r["agg_mse10"]:.6f} '
                 f'retmse10={r["retmse10"]:.6f} spearman_vs_oracle={r["spearman_vs_oracle"]:.4f}')
        print(f'[diag_p100_fusion] {cli.cell}/{split}: disagreement={out["disagreement"]} '
             f'oracle_agg_mse10={out["oracle_agg_mse10"]:.6f}')


if __name__ == '__main__':
    main()
