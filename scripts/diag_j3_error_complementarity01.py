#!/usr/bin/env python3
"""TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01 -- NO TRAINING, pure post-hoc
analysis of J0 (TRACK-J-SHARED-ENCODER-DRIFT01, best_epoch=10) and J1
(TRACK-J2-KEY-UPDATE-DECOMPOSITION01/J1_stopgrad_key, best_epoch=1)
best checkpoints, on the full ETTh1_720 test split.

Decomposes Uniform Aggregate MSE@10 into a diagonal/individual term D and
a cross-error-interaction term C:
    Agg = D + C
    D = (1/K^2) sum_i (1/H)||e_i||^2 = retMSE@K / K
    C = (1/K^2) sum_{i!=j} (1/H) e_i . e_j
where e_i = Y_i - Y_q is the candidate-i prediction error, computed via
the SAME `memory_value`/`individual_utility_memsafe`-based delta_last
reconstruction every other script this session has used (reused
unmodified). Top-10 picks are the SAME future-blind
`arm_score`/`stable_topk_indices` selection the checkpoint's own
production evaluation uses.

Reused UNMODIFIED: `build_model`, `state_hash`
(`train_j_shared_encoder_drift01`); `encode_raw`/`arm_score`/
`individual_utility_memsafe` (`train_factorial_e2e01`); `memory_value`
(`train_margutil01`); `stable_topk_indices` (`RelationStage1`);
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_j_shared_encoder_drift01 import build_model, build_probe_set, state_hash
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
TOP_K = 10
OUT_DIR = REPO_ROOT / 'results/TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01/ETTh1_720'

J0_CKPT = REPO_ROOT / 'checkpoints/track_j_shared_encoder_drift01/ETTh1_720/checkpoint.pth'
J1_CKPT = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'
J2_CKPT = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J2_true_frozen_key/checkpoint.pth'
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'

EXPECTED = {
    'J0': {'retmse10': 0.9535986185073853, 'agg10': 0.6417305831398282, 'recall10': 0.020256697067192624},
    'J1': {'retmse10': 1.0083508065768652, 'agg10': 0.5643467860562461, 'recall10': 0.02198660683019885},
}


class _Cli:
    reference_ckpt = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
                      'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_'
                      'nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
    cell = 'ETTh1_720'
    pred_len = 720
    seq_len = 720
    patch_len = 16
    top_k = 10
    tau_t = 0.1
    tau_s = 0.1
    batch_size = 32
    learning_rate = 1e-3
    chunk_size = 4096
    init_seed = 0
    loader_seed = 0


def load_arm_model(exp, args, base_model, arm):
    """Returns (score_model_q, score_model_k) -- for J0/J1 both are the
    same single loaded model; for J2, E_q is the checkpointed model and
    E_k is a frozen deepcopy of the (shared) init, matching the original
    run exactly."""
    device = next(base_model.parameters()).device
    m = copy.deepcopy(base_model).to(device)
    if arm in ('J0', 'J1'):
        ckpt_path = J0_CKPT if arm == 'J0' else J1_CKPT
        bl = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(bl['model_state_dict'])
        m.eval()
        return m, m, bl.get('epoch')
    elif arm == 'J2':
        bl = torch.load(J2_CKPT, map_location=device)
        m.load_state_dict(bl['model_state_dict'])
        m.eval()
        e_k = copy.deepcopy(base_model).to(device)
        for p in e_k.parameters():
            p.requires_grad_(False)
        e_k.eval()
        assert state_hash(e_k) == EXPECTED_INIT_HASH
        return m, e_k, bl.get('epoch')
    raise ValueError(arm)


@torch.no_grad()
def run_probe_decomposition(exp, args, model_q, model_k, probe, channels, device, chunk_size, k=TOP_K):
    """Returns a list of per-query-channel dict rows plus a list of
    per-candidate leave-one-out rows, computed on the FIXED probe set
    (same 256-query, deterministic, evenly-spaced construction TRACK-J's
    own `final_test_metrics.json` was computed on -- matching that
    exactly is required for the reproduction gate to pass, see script
    docstring)."""
    rows = []
    loo_rows = []
    n_total = 0
    batch_x, batch_y, batch_start_idx = probe['x'], probe['y'], probe['start_idx']
    cand_mask = probe['cand_mask']
    bsz = batch_x.size(0)
    if True:
        for c in channels:
            z_q = encode_raw(model_q, batch_x, c)
            z_k = encode_raw(model_k, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]  # [B, H]
            H = query_future.size(-1)

            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u
            s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
            model_idx = stable_topk_indices(s, k, largest=True)  # future-blind
            oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), k, largest=False)
            recall10 = recall_at_k(model_idx, oracle_idx, k)
            ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, k)

            y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)  # [B, K, H]
            e = y_sel - query_future.unsqueeze(1)  # [B, K, H]

            individual_mse_i = (e ** 2).mean(-1)  # [B, K]
            retmse_query = individual_mse_i.mean(-1)  # [B]
            D_query = retmse_query / k

            agg_pred = y_sel.mean(dim=1)  # [B, H]
            agg_err = agg_pred - query_future
            Agg_query = (agg_err ** 2).mean(-1)  # [B]
            C_query_via_subtraction = Agg_query - D_query

            # independent direct cross-term computation
            cross_mat = torch.einsum('bih,bjh->bij', e, e) / H  # [B, K, K]
            diag_sum = torch.diagonal(cross_mat, dim1=-2, dim2=-1).sum(-1)  # [B]
            total_sum = cross_mat.sum(dim=(-1, -2))  # [B]
            off_diag_sum = total_sum - diag_sum
            D_direct = diag_sum / (k * k)
            C_direct = off_diag_sum / (k * k)
            Agg_direct = D_direct + C_direct

            # exact identity check (assert later at macro level too)
            assert torch.allclose(D_query, D_direct, atol=1e-4), 'D mismatch'
            assert torch.allclose(Agg_query, Agg_direct, atol=1e-4), 'Agg identity mismatch'

            # pairwise error dot / cosine, off-diagonal (i<j) only
            norms = e.norm(dim=-1)  # [B, K]
            denom = norms.unsqueeze(-1) * norms.unsqueeze(-2) + EPS
            cos_mat = torch.einsum('bih,bjh->bij', e, e) / denom  # e_i.e_j / (|e_i||e_j|)
            iu = torch.triu_indices(k, k, offset=1)
            pair_dot = cross_mat[:, iu[0], iu[1]]  # [B, 45], already /H
            pair_cos = cos_mat[:, iu[0], iu[1]]  # [B, 45]

            # future diversity (pairwise future MSE, i<j)
            fut_diff = y_sel.unsqueeze(2) - y_sel.unsqueeze(1)  # [B,K,K,H]
            fut_sq = (fut_diff ** 2).mean(-1)  # [B,K,K]
            fut_pair = fut_sq[:, iu[0], iu[1]]  # [B,45]

            # bias/variance decomposition over horizon steps
            mu_h = e.mean(dim=1)  # [B, H]
            var_h = e.var(dim=1, unbiased=False)  # [B, H]

            # sign cancellation per horizon step
            pos_count = (e > 0).float().sum(dim=1)  # [B, H]
            neg_count = (e < 0).float().sum(dim=1)  # [B, H]
            mixed = ((pos_count > 0) & (neg_count > 0)).float().mean(-1)  # [B]

            # leave-one-out benefit per candidate
            k_agg = y_sel.sum(dim=1)  # [B, H]
            for kk in range(k):
                agg_wo = (k_agg - y_sel[:, kk]) / (k - 1)
                mse_wo = ((agg_wo - query_future) ** 2).mean(-1)  # [B]
                benefit = mse_wo - Agg_query  # [B]
                for b in range(bsz):
                    loo_rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c, 'rank': kk,
                                     'individual_mse_i': float(individual_mse_i[b, kk]),
                                     'benefit_i': float(benefit[b])})

            for b in range(bsz):
                rows.append({
                    'query_start_idx': int(batch_start_idx[b]), 'channel': c,
                    'retmse10': float(retmse_query[b]), 'D': float(D_query[b]), 'C': float(C_query_via_subtraction[b]),
                    'agg_mse10': float(Agg_query[b]), 'recall10': float(recall10[b]), 'ndcg10': float(ndcg10[b]),
                    'mean_pairwise_error_dot': float(pair_dot[b].mean()),
                    'mean_pairwise_error_cosine': float(pair_cos[b].mean()),
                    'frac_error_cosine_neg': float((pair_cos[b] < 0).float().mean()),
                    'frac_error_dot_neg': float((pair_dot[b] < 0).float().mean()),
                    'mean_future_pairwise_mse': float(fut_pair[b].mean()),
                    'mean_abs_mu_h': float(mu_h[b].abs().mean()), 'mean_mu_h_sq': float((mu_h[b] ** 2).mean()),
                    'mean_variance_h': float(var_h[b].mean()), 'mixed_sign_fraction': float(mixed[b]),
                })
        n_total += bsz
    return rows, loo_rows, n_total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _Cli()
    exp, args, base_model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    assert state_hash(base_model) == EXPECTED_INIT_HASH, '[ISSUE][ABORT] init hash mismatch'
    n_candidates = int(exp.memory_x.size(0))
    assert n_candidates == 7201, f'[ISSUE][ABORT] unexpected candidate count {n_candidates}'

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe = build_probe_set(exp, test_loader, 256, device)
    print(f'[diag_j3] probe set: n={probe["n"]} (deterministic, evenly-spaced -- same construction '
         f'TRACK-J/TRACK-J2 used for their own final_test_metrics.json)')

    all_rows = {}
    loo_all = {}
    reproduction = {}
    for arm in ('J0', 'J1', 'J2'):
        print(f'[diag_j3] running probe-set decomposition for {arm} ...')
        model_q, model_k, epoch = load_arm_model(exp, args, base_model, arm)
        rows, loo_rows, n = run_probe_decomposition(exp, args, model_q, model_k, probe, channels, device, cli.chunk_size)
        all_rows[arm] = rows
        loo_all[arm] = loo_rows
        df = pd.DataFrame(rows)
        macro = {'retmse10': float(df['retmse10'].mean()), 'agg_mse10': float(df['agg_mse10'].mean()),
                 'recall10': float(df['recall10'].mean()), 'ndcg10': float(df['ndcg10'].mean()),
                 'D': float(df['D'].mean()), 'C': float(df['C'].mean()), 'n_queries_seen': n, 'best_epoch': epoch}
        reproduction[arm] = macro
        print(f'[diag_j3] {arm} best_epoch={epoch} n={n} macro={json.dumps(macro, indent=2)}')
        if arm in EXPECTED:
            exp_v = EXPECTED[arm]
            diffs = {'retmse10_diff': abs(macro['retmse10'] - exp_v['retmse10']),
                    'agg_diff': abs(macro['agg_mse10'] - exp_v['agg10']),
                    'recall_diff': abs(macro['recall10'] - exp_v['recall10'])}
            reproduction[arm]['diffs_vs_saved'] = diffs
            print(f'[diag_j3] {arm} reproduction diffs vs saved: {diffs}')
            for k_, v in diffs.items():
                if v > 1e-5:
                    print(f'[diag_j3][ISSUE] {arm} {k_} = {v} exceeds 1e-5 tolerance')

        df.to_parquet(OUT_DIR / f'per_query_decomposition_{arm}.parquet', index=False)
        pd.DataFrame(loo_rows).to_csv(OUT_DIR / f'leave_one_out_contribution_{arm}.csv', index=False)

    (OUT_DIR / 'reproduction.json').write_text(json.dumps(reproduction, indent=2))
    print('[diag_j3] done. reproduction.json written.')


if __name__ == '__main__':
    main()
