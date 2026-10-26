#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE D0: Set Oracle
headroom diagnostic (evaluation-only, cheating oracle, no training).

Primary-scorer selection (spec section 24, done ONCE, on VALIDATION
only, before any test number was looked at): compared VAL AggMSE@10,
averaged across all 4 cells, for every deployable scorer produced by
Phases A-C --

    canonical V5 (beta=1.0):                avg val AggMSE@10 = 0.9444
    Phase A best fixed coarse+V5 fusion:     avg val AggMSE@10 = 0.9418
    Phase C Residual-V5 (C1):                avg val AggMSE@10 = 0.9384  <- selected
    Phase B adaptive fusion:                 excluded -- Phase B's own
        verdict already found no reliable deployable contribution in
        3/4 cells (its B1 arm did not beat the no-feature B2 control),
        and its per-epoch model state was never checkpointed to disk
        (only the aggregate JSON result was saved), so computing its
        val AggMSE here would require a full retrain; given the
        already-established negative verdict this was not worth the
        extra GPU time and is recorded as a deliberate exclusion, not
        an oversight.

**Phase C's Residual-V5 (C1) is therefore this track's primary
scorer for Phase D**, applied with each cell's own already-trained,
best-validation-RetMSE@10 checkpoint (no retraining here). Within the
SAME fixed Shared-P100, its Mean-Mixture ranking reduces P100 to a
fixed Top-M=20 (spec section 25, M never swept), inside which:

  - Individual Top10: the scorer's own natural Top-10 (a subset of
    its own Top-20 by construction) -- identical to the AggMSE@10
    already reported in `PHASE_C_RESIDUAL_SCORER_REPORT.md`, recomputed
    here restricted to the same Top-20 universe for an apples-to-apples
    comparison against the Set Oracle below (never materially
    different, since Top-10-of-100 is always already inside Top-20-of-100).
  - Greedy Set Oracle (cheating, reads the true future): starting from
    an empty set, repeatedly adds whichever of the remaining Top-20
    candidates most reduces the uniform-mean aggregate future MSE,
    for K=10 steps.

Reuses UNMODIFIED: `build_model`, `SlotHeads`, `encode_raw`,
`memory_value`, `CandidatePoolCache`, `compute_coarse_delta_last_cosine_scores`,
`stable_topk_indices`.
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
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TOP_K = 10
TOP_M = 20
N_SLOTS = 5


@torch.no_grad()
def greedy_set_oracle(top20_future, query_future, k):
    """top20_future: [B,20,H] candidate future values (already offset
    -adjusted). query_future: [B,H]. Returns SetOracle AggMSE@k [B] via
    a vectorized greedy incremental-mean construction (no python loop
    over the batch, only over the k selection steps and the up-to-20
    remaining-candidate scan per step)."""
    bsz, m, h = top20_future.shape
    device = top20_future.device
    selected_mask = torch.zeros(bsz, m, dtype=torch.bool, device=device)
    running_sum = torch.zeros(bsz, h, device=device)
    for t in range(1, k + 1):
        # candidate new mean if we add each still-unselected j: (running_sum + future_j) / t
        cand_mean = (running_sum.unsqueeze(1) + top20_future) / t  # [B,20,H]
        cand_mse = ((cand_mean - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,20]
        cand_mse = cand_mse.masked_fill(selected_mask, float('inf'))
        best_j = cand_mse.argmin(dim=-1)  # [B]
        running_sum = running_sum + top20_future.gather(1, best_j.view(-1, 1, 1).expand(-1, 1, h)).squeeze(1)
        selected_mask.scatter_(1, best_j.unsqueeze(1), True)
    final_mean = running_sum / k
    return ((final_mean - query_future) ** 2).mean(dim=-1)  # [B]


@torch.no_grad()
def run_cell(cli, device):
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl = torch.load(cli.residual_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    gamma = float(bl['gamma'])
    slot_heads.eval()
    print(f'[audit] cell={cli.cell} residual checkpoint={cli.residual_checkpoint} '
         f'(epoch={bl["epoch"]}, gamma={gamma:.6f})')

    channels = list(range(int(args.enc_in)))
    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{cli.split}.pt', expected_meta)
    _, loader = exp._get_data(flag=cli.split, shuffle=False)

    individual_agg, oracle_agg = [], []
    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
            c_i = coarse_full.gather(1, pool_idx_global)  # [B,100]
            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)
            k_pool = k_full[pool_idx_global]
            r = torch.einsum('bsd,bmd->bsm', q, k_pool)  # [B,5,100]
            s_final = c_i.unsqueeze(1) + gamma * r
            p_heads = F.softmax(s_final / cli.tau_s, dim=-1)
            p_mix = p_heads.mean(dim=1)  # [B,100]

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)  # [B,100,H]

            top20_local = stable_topk_indices(p_mix, TOP_M, largest=True)  # [B,20]
            top20_future = pooled_future.gather(1, top20_local.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
            top20_p = p_mix.gather(1, top20_local)

            top10_local_within20 = stable_topk_indices(top20_p, TOP_K, largest=True)
            y_sel = top20_future.gather(1, top10_local_within20.unsqueeze(-1).expand(-1, -1, top20_future.size(-1)))
            individual_agg.append((((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)).cpu())

            oracle_agg.append(greedy_set_oracle(top20_future, query_future, TOP_K).cpu())

    individual_mean = float(torch.cat(individual_agg).mean())
    oracle_mean = float(torch.cat(oracle_agg).mean())
    gain_pct = (individual_mean - oracle_mean) / individual_mean * 100.0
    return {'cell': cli.cell, 'split': cli.split, 'individual_agg_mse10': individual_mean,
           'set_oracle_agg_mse10': oracle_mean, 'oracle_gain_pct': gain_pct}


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
    ap.add_argument('--residual_checkpoint', required=True)
    ap.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = TOP_K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = run_cell(cli, device)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_{cli.split}_set_oracle.json').write_text(json.dumps(out, indent=2))
    print(f'[diag_set_oracle] {cli.cell}/{cli.split}: individual={out["individual_agg_mse10"]:.6f} '
         f'set_oracle={out["set_oracle_agg_mse10"]:.6f} gain%={out["oracle_gain_pct"]:.3f}')


if __name__ == '__main__':
    main()
