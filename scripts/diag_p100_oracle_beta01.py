#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE B step 1: Oracle Beta
diagnostic (evaluation-only, cheating oracle, no training). For each
query/channel, picks the beta in {0,.25,.5,.75,1} whose FUSED
selection (log-domain fusion of coarse + canonical V5 Mean-Mixture
scores, identical math to Phase A's
`scripts.diag_p100_coarse_v5_fusion01`) gives the lowest ACTUAL future
AggMSE@10 -- i.e. this reads the future to pick beta per query, which
is why it is diagnostic-only and never deployable. Records: global
fixed-beta AggMSE (recomputed here, must match Phase A's own numbers
-- asserted), per-query oracle-beta AggMSE, oracle gain %, oracle beta
distribution, oracle beta entropy, beta winner share.

Reuses UNMODIFIED everything Phase A reuses, plus Phase A's own
`_eval_arm` / fusion math (same TAU_C=0.05, TAU_S=0.1, BETAS) imported
directly rather than re-derived, so the oracle and the deployable
Phase A numbers can never silently drift apart.
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
from scripts.diag_p100_coarse_v5_fusion01 import BETAS, EPS, TAU_C, TOP_K, N_SLOTS
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores


def _entropy(p, eps=1e-12):
    return -(p * (p + eps).log()).sum(dim=-1)


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

    agg_mse_by_beta = {beta: [] for beta in BETAS}
    per_query_best_beta = []
    per_query_best_agg = []
    per_query_global_beta_agg = {beta: [] for beta in BETAS}  # for cross-check vs Phase A

    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            bsz = batch_x.size(0)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
            c_i = coarse_full.gather(1, pool_idx_global)
            p_c = F.softmax(c_i / TAU_C, dim=-1)

            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)
            k_pool = k_full[pool_idx_global]
            scores = torch.einsum('bsd,bmd->bsm', q, k_pool)
            p_heads = F.softmax(scores / cli.tau_s, dim=-1)
            p_v5 = p_heads.mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)

            agg_mse_this_row = {}  # beta -> [B]
            for beta in BETAS:
                raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
                p_fused = F.softmax(raw, dim=-1)
                picks = stable_topk_indices(p_fused, TOP_K, largest=True)
                y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
                r_c = y_sel.mean(dim=1)
                agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)  # [B]
                agg_mse_this_row[beta] = agg_mse
                agg_mse_by_beta[beta].append(agg_mse.cpu())

            stacked = torch.stack([agg_mse_this_row[b] for b in BETAS], dim=1)  # [B, n_beta]
            best_j = stacked.argmin(dim=1)  # [B]
            best_val = stacked.gather(1, best_j.unsqueeze(1)).squeeze(1)
            per_query_best_beta.extend([BETAS[j] for j in best_j.tolist()])
            per_query_best_agg.append(best_val.cpu())

    out = {'cell': cli.cell, 'split': split}
    global_fixed_agg = {str(b): float(torch.cat(v).mean()) for b, v in agg_mse_by_beta.items()}
    out['global_fixed_beta_agg_mse10'] = global_fixed_agg
    best_global_beta = min(global_fixed_agg, key=global_fixed_agg.get)
    out['best_global_fixed_beta'] = best_global_beta
    out['best_global_fixed_agg_mse10'] = global_fixed_agg[best_global_beta]

    oracle_mean = float(torch.cat(per_query_best_agg).mean())
    out['oracle_beta_agg_mse10'] = oracle_mean
    out['oracle_gain_pct_vs_best_global'] = (
        (global_fixed_agg[best_global_beta] - oracle_mean) / global_fixed_agg[best_global_beta] * 100.0)

    n = len(per_query_best_beta)
    winner_share = {str(b): per_query_best_beta.count(b) / n for b in BETAS}
    out['oracle_beta_winner_share'] = winner_share
    probs = torch.tensor([winner_share[str(b)] for b in BETAS])
    out['oracle_beta_entropy'] = float(_entropy(probs.unsqueeze(0)).squeeze(0))
    out['oracle_beta_entropy_max'] = float(torch.log(torch.tensor(float(len(BETAS)))))
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
    print(f'[audit] cell={cli.cell} V5 checkpoint = {cli.v5_checkpoint} oracle-beta diagnostic (evaluation-only)')

    for split in cli.splits:
        out = run_cell_split(cli, device, split)
        (out_dir / f'{cli.cell}_{split}_oracle_beta.json').write_text(json.dumps(out, indent=2))
        print(f'[diag_oracle_beta] {cli.cell}/{split}: best_global_beta={out["best_global_fixed_beta"]} '
             f'best_global_agg={out["best_global_fixed_agg_mse10"]:.6f} '
             f'oracle_agg={out["oracle_beta_agg_mse10"]:.6f} '
             f'oracle_gain%={out["oracle_gain_pct_vs_best_global"]:.3f} '
             f'winner_share={out["oracle_beta_winner_share"]} '
             f'entropy={out["oracle_beta_entropy"]:.4f}/{out["oracle_beta_entropy_max"]:.4f}')


if __name__ == '__main__':
    main()
