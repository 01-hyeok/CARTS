#!/usr/bin/env python3
"""TRACK-L-EVAL-ALIGNMENT-FULLTEST01 -- unified evaluator.

NO TRAINING. Evaluates J0/J1/K1/K2's EXISTING checkpoints (never
re-selected, never re-trained) on TWO aligned populations:
  P256  -- the same fixed 256-query probe TRACK-J/J2/J3 used
           (`build_probe_set`, reproduced deterministically and checked
           against the saved `probe_query_ids.json`)
  FULL2161 -- the complete ETTh1_720 test split, all 2161 queries.

All four arms share the SAME candidate mask / memory reconstruction /
oracle computation / retMSE / aggregate MSE / D-C decomposition /
Recall@10 / NDCG@10 / regret code path -- only "how scores are produced"
and "how Top-10 is hard-selected" differ per arm (single-score-vector
Top-10 for J0/J1; 10-slot greedy-unique selection for K1/K2), matching
each arm's own original evaluation protocol exactly.

Reused UNMODIFIED: `build_model`/`build_probe_set`/`state_hash`
(`train_j_shared_encoder_drift01`), `encode_raw`/`arm_score`/
`individual_utility_memsafe` (`train_factorial_e2e01`), `memory_value`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`SlotHeads`/`compute_scores`/`hard_unique_selection`
(`train_k_multislot_predictive_retrieval01`).
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_j_shared_encoder_drift01 import build_model, build_probe_set, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import (
    N_SLOTS, SlotHeads, compute_scores, hard_unique_selection,
)
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
TOP_K = 10
OUT_DIR = REPO_ROOT / 'results/TRACK-L-EVAL-ALIGNMENT-FULLTEST01/ETTh1_720'
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'
# NOTE: `probe_query_ids.json` (saved by TRACK-J) is that track's VALIDATION
# probe, not its test probe -- TRACK-J/J2 never saved test-probe indices to
# a file; they (and TRACK-J3) relied solely on `build_probe_set`'s
# determinism (already unit-tested, TRACK-J item 12) on the TEST loader.
# We do the same here and verify correctness via the P256 reproduction gate
# (Section 5) reproducing J0/J1's saved test numbers to <1e-5, rather than
# comparing against the (wrongly-named-for-this-purpose) val-probe file.

CKPTS = {
    'J0': REPO_ROOT / 'checkpoints/track_j_shared_encoder_drift01/ETTh1_720/checkpoint.pth',
    'J1': REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth',
    'K1': REPO_ROOT / 'checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K1_multislot_relevance/checkpoint.pth',
    'K2': REPO_ROOT / 'checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth',
}
IS_MULTISLOT = {'J0': False, 'J1': False, 'K1': True, 'K2': True}

EXPECTED_P256 = {
    'J0': {'retmse10': 0.9535986185073853, 'agg10': 0.6417305831398282, 'recall10': 0.020256697067192624},
    'J1': {'retmse10': 1.0083508065768652, 'agg10': 0.5643467860562461, 'recall10': 0.02198660683019885},
}
EXPECTED_FULL = {
    'K1': {'retmse10': 1.1396289975989746, 'agg10': 0.5737499419385272, 'recall10': 0.02001718850809434},
    'K2': {'retmse10': 1.120914531099258, 'agg10': 0.5469431672556793, 'recall10': 0.02746083227808086},
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


def load_arm(arm, base_model, d_model, device):
    ckpt_path = CKPTS[arm]
    bl = torch.load(ckpt_path, map_location=device)
    m = copy.deepcopy(base_model).to(device)
    m.load_state_dict(bl['model_state_dict'])
    m.eval()
    slot_heads = None
    if IS_MULTISLOT[arm]:
        slot_heads = SlotHeads(d_model, N_SLOTS).to(device)
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        slot_heads.eval()
    ckpt_hash = state_hash(m) if not IS_MULTISLOT[arm] else None
    return m, slot_heads, bl.get('epoch'), ckpt_path


@torch.no_grad()
def score_and_decompose(arm, m, slot_heads, exp, args, batch_x, batch_y, batch_start_idx, channels, device,
                        chunk_size, top_k=TOP_K):
    """One shared decomposition path for every arm. Returns per-channel
    dict rows (one dict per channel, each holding [B]-length tensors)."""
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    rows_per_channel = {}
    for c in channels:
        if IS_MULTISLOT[arm]:
            scores = compute_scores(m, slot_heads, batch_x, exp.memory_x, c)  # [B, S, N]
        else:
            z_q = encode_raw(m, batch_x, c)
            z_k = encode_raw(m, exp.memory_x, c)
            scores = arm_score(z_q, z_k, None).unsqueeze(1)  # [B, 1, N] -- uniform shape w/ multislot path

        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), top_k, largest=False)

        if IS_MULTISLOT[arm]:
            model_idx = hard_unique_selection(scores, cand_mask, s=top_k)
        else:
            s_masked = scores[:, 0, :].masked_fill(~cand_mask, float('-inf'))
            model_idx = stable_topk_indices(s_masked, top_k, largest=True)

        model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
        oracle_ind_mse = d_raw.gather(1, oracle_idx).mean(-1)
        recall10 = recall_at_k(model_idx, oracle_idx, top_k)
        ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, top_k)

        y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)  # [B, K, H]
        e = y_sel - query_future.unsqueeze(1)
        individual_mse_i = (e ** 2).mean(-1)
        retmse = individual_mse_i.mean(-1)
        D_ = retmse / top_k
        agg_pred = y_sel.mean(dim=1)
        agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
        C_ = agg_mse - D_

        # exact identity check, every query-channel, every call
        assert torch.allclose(D_ + C_, agg_mse, atol=1e-4), f'[ISSUE] Agg=D+C failed for {arm} channel {c}'
        assert torch.allclose(D_, retmse / top_k, atol=1e-6)

        # complementarity diagnostics (pairwise error dot/cosine, future diversity, sign cancellation)
        norms = e.norm(dim=-1)
        denom = norms.unsqueeze(-1) * norms.unsqueeze(-2) + EPS
        cross = torch.einsum('bih,bjh->bij', e, e)
        cos = cross / denom
        iu = torch.triu_indices(top_k, top_k, offset=1)
        pair_dot = (cross / e.size(-1))[:, iu[0], iu[1]]
        pair_cos = cos[:, iu[0], iu[1]]
        fut_diff = y_sel.unsqueeze(2) - y_sel.unsqueeze(1)
        fut_pair = (fut_diff ** 2).mean(-1)[:, iu[0], iu[1]]
        mu_h = e.mean(dim=1)
        pos_count = (e > 0).float().sum(dim=1)
        neg_count = (e < 0).float().sum(dim=1)
        mixed = ((pos_count > 0) & (neg_count > 0)).float().mean(-1)

        rows_per_channel[c] = dict(
            query_start_idx=batch_start_idx if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx),
            retmse10=model_ind_mse.cpu(), D=D_.cpu(), C=C_.cpu(), agg_mse10=agg_mse.cpu(),
            recall10=recall10.cpu(), ndcg10=ndcg10.cpu(), oracle_regret=(model_ind_mse - oracle_ind_mse).cpu(),
            mean_pairwise_error_dot=pair_dot.mean(-1).cpu(), mean_pairwise_error_cosine=pair_cos.mean(-1).cpu(),
            frac_error_cosine_neg=(pair_cos < 0).float().mean(-1).cpu(),
            mean_future_pairwise_mse=fut_pair.mean(-1).cpu(), mixed_sign_fraction=mixed.cpu(),
            mean_abs_mu_h=mu_h.abs().mean(-1).cpu(),
        )
    return rows_per_channel


def run_population(arm, m, slot_heads, exp, args, loader, channels, device, chunk_size):
    all_rows = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        per_ch = score_and_decompose(arm, m, slot_heads, exp, args, batch_x, batch_y, batch_start_idx,
                                     channels, device, chunk_size)
        bsz = batch_x.size(0)
        for c in channels:
            r = per_ch[c]
            for b in range(bsz):
                all_rows.append({'query_start_idx': int(r['query_start_idx'][b]), 'channel': c,
                                 **{k: float(v[b]) for k, v in r.items() if k != 'query_start_idx'}})
    return pd.DataFrame(all_rows)


def macro_summary(df):
    keys = ['retmse10', 'D', 'C', 'agg_mse10', 'recall10', 'ndcg10', 'oracle_regret',
           'mean_pairwise_error_dot', 'mean_pairwise_error_cosine', 'frac_error_cosine_neg',
           'mean_future_pairwise_mse', 'mixed_sign_fraction', 'mean_abs_mu_h']
    return {k: float(df[k].mean()) for k in keys}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _Cli()
    exp, args, base_model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    n_candidates = int(exp.memory_x.size(0))
    assert state_hash(base_model) == EXPECTED_INIT_HASH
    assert n_candidates == 7201

    _, val_loader_unused = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    probe = build_probe_set(exp, test_loader, 256, device)
    probe_b = build_probe_set(exp, test_loader, 256, device)
    assert probe['start_idx'].tolist() == probe_b['start_idx'].tolist(), '[ISSUE][ABORT] P256 probe not deterministic'
    (OUT_DIR / 'p256_probe_query_ids.json').write_text(json.dumps(
        {'start_idx': probe['start_idx'].tolist(), 'n_probe': probe['n']}, indent=2))
    print(f'[eval_l] P256 test-probe construction verified self-deterministic (n={probe["n"]}); '
         f'correctness vs TRACK-J/J2 confirmed via the reproduction gate below, not a saved-file comparison '
         f'(TRACK-J\'s own probe_query_ids.json is its VALIDATION probe, a different population)')

    n_full = len(exp._get_data(flag='test', shuffle=False)[1].dataset)
    print(f'[eval_l] FULL test split n={n_full}')

    checkpoint_audit = {}
    dfs_p256, dfs_full = {}, {}
    for arm in ('J0', 'J1', 'K1', 'K2'):
        m, slot_heads, epoch, ckpt_path = load_arm(arm, base_model, d_model, device)
        ckpt_hash_full = None
        with open(ckpt_path, 'rb') as f:
            import hashlib
            ckpt_hash_full = hashlib.sha256(f.read()).hexdigest()
        checkpoint_audit[arm] = {'path': str(ckpt_path.relative_to(REPO_ROOT)), 'file_sha256': ckpt_hash_full,
                                 'best_epoch': epoch, 'n_candidates': n_candidates, 'top_k': TOP_K,
                                 'channels': len(channels), 'is_multislot': IS_MULTISLOT[arm]}
        print(f'[eval_l] {arm}: best_epoch={epoch} ckpt={ckpt_path.name}')

        # P256
        _, test_loader_p = exp._get_data(flag='test', shuffle=False)
        probe_p = build_probe_set(exp, test_loader_p, 256, device)
        df_p256_rows = []
        per_ch = score_and_decompose(arm, m, slot_heads, exp, args, probe_p['x'], probe_p['y'], probe_p['start_idx'],
                                     channels, device, cli.chunk_size)
        bsz = probe_p['x'].size(0)
        for c in channels:
            r = per_ch[c]
            for b in range(bsz):
                df_p256_rows.append({'query_start_idx': int(r['query_start_idx'][b]), 'channel': c,
                                     **{k: float(v[b]) for k, v in r.items() if k != 'query_start_idx'}})
        df_p256 = pd.DataFrame(df_p256_rows)
        dfs_p256[arm] = df_p256

        # FULL2161
        _, test_loader_full = exp._get_data(flag='test', shuffle=False)
        df_full = run_population(arm, m, slot_heads, exp, args, test_loader_full, channels, device, cli.chunk_size)
        dfs_full[arm] = df_full

        macro_p = macro_summary(df_p256)
        macro_f = macro_summary(df_full)
        print(f'[eval_l] {arm} P256: retmse={macro_p["retmse10"]:.6f} agg={macro_p["agg_mse10"]:.6f} '
             f'recall={macro_p["recall10"]:.6f}')
        print(f'[eval_l] {arm} FULL: retmse={macro_f["retmse10"]:.6f} agg={macro_f["agg_mse10"]:.6f} '
             f'recall={macro_f["recall10"]:.6f} n={len(df_full)/len(channels):.0f}')

        (OUT_DIR / 'P256' / f'{arm}.json').write_text(json.dumps(
            {**macro_p, 'n_queries': int(df_p256.query_start_idx.nunique())}, indent=2))
        (OUT_DIR / 'FULL2161' / f'{arm}.json').write_text(json.dumps(
            {**macro_f, 'n_queries': int(df_full.query_start_idx.nunique())}, indent=2))
        df_p256.to_parquet(OUT_DIR / 'P256' / f'{arm}_per_query.parquet', index=False)
        df_full.to_parquet(OUT_DIR / 'FULL2161' / f'{arm}_per_query.parquet', index=False)

    (OUT_DIR / 'checkpoint_audit.json').write_text(json.dumps(checkpoint_audit, indent=2))

    # reproduction gates
    repro = {}
    for arm, exp_v in EXPECTED_P256.items():
        got = json.loads((OUT_DIR / 'P256' / f'{arm}.json').read_text())
        diffs = {'retmse10_diff': abs(got['retmse10'] - exp_v['retmse10']),
                'agg_diff': abs(got['agg_mse10'] - exp_v['agg10']),
                'recall_diff': abs(got['recall10'] - exp_v['recall10'])}
        repro[f'{arm}_P256'] = diffs
        print(f'[eval_l] {arm} P256 repro diffs: {diffs}')
    for arm, exp_v in EXPECTED_FULL.items():
        got = json.loads((OUT_DIR / 'FULL2161' / f'{arm}.json').read_text())
        diffs = {'retmse10_diff': abs(got['retmse10'] - exp_v['retmse10']),
                'agg_diff': abs(got['agg_mse10'] - exp_v['agg10']),
                'recall_diff': abs(got['recall10'] - exp_v['recall10'])}
        repro[f'{arm}_FULL'] = diffs
        print(f'[eval_l] {arm} FULL repro diffs: {diffs}')
    (OUT_DIR / 'reproduction_gates.json').write_text(json.dumps(repro, indent=2))
    max_diff = max(v for d in repro.values() for v in d.values())
    print(f'[eval_l] max reproduction diff across all gates: {max_diff:.2e} '
         f'({"PASS" if max_diff < 1e-5 else "[ISSUE] EXCEEDS 1e-5"})')

    # comparison CSVs
    for pop_name, dfs in (('P256', dfs_p256), ('FULL2161', dfs_full)):
        rows = []
        for arm in ('J0', 'J1', 'K1', 'K2'):
            rows.append({'arm': arm, **macro_summary(dfs[arm])})
        pd.DataFrame(rows).to_csv(OUT_DIR / pop_name / 'comparison.csv', index=False)

    # per-channel FULL2161
    ch_rows = []
    for arm in ('J0', 'J1', 'K1', 'K2'):
        g = dfs_full[arm].groupby('channel').agg(
            retmse10=('retmse10', 'mean'), D=('D', 'mean'), C=('C', 'mean'),
            agg_mse10=('agg_mse10', 'mean'), recall10=('recall10', 'mean')).reset_index()
        g['arm'] = arm
        ch_rows.append(g)
    pd.concat(ch_rows, ignore_index=True).to_csv(OUT_DIR / 'FULL2161' / 'per_channel.csv', index=False)

    # J3 decomposition on FULL2161 (J0 vs J1)
    j3dir = OUT_DIR / 'j3_full_decomposition'
    j0f, j1f = dfs_full['J0'], dfs_full['J1']
    macro_j0, macro_j1 = macro_summary(j0f), macro_summary(j1f)
    delta = {k: macro_j1[k] - macro_j0[k] for k in ('D', 'C', 'agg_mse10', 'retmse10')}
    gain_ind = macro_j0['D'] - macro_j1['D']
    gain_cross = macro_j0['C'] - macro_j1['C']
    r_cross = gain_cross / (macro_j0['agg_mse10'] - macro_j1['agg_mse10'])
    macro_out = {'J0': macro_j0, 'J1': macro_j1, 'delta_J1_minus_J0': delta,
                'Gain_individual': gain_ind, 'Gain_cross': gain_cross, 'R_cross': r_cross}
    (j3dir / 'macro.json').write_text(json.dumps(macro_out, indent=2))
    print(f'[eval_l] FULL2161 J3 macro: D_J0={macro_j0["D"]:.6f} D_J1={macro_j1["D"]:.6f} '
         f'C_J0={macro_j0["C"]:.6f} C_J1={macro_j1["C"]:.6f} R_cross={r_cross:.4f}')

    paired = j0f.merge(j1f, on=['query_start_idx', 'channel'], suffixes=('_j0', '_j1'))
    paired['delta_agg'] = paired['agg_mse10_j1'] - paired['agg_mse10_j0']
    paired['delta_D'] = paired['D_j1'] - paired['D_j0']
    paired['delta_C'] = paired['C_j1'] - paired['C_j0']
    paired.to_csv(j3dir / 'paired_query.csv', index=False)
    frac_j1_better = float((paired['delta_agg'] < 0).mean())
    frac_ind_worse_set_better = float(((paired['delta_D'] > 0) & (paired['delta_agg'] < 0)).mean())
    print(f'[eval_l] FULL2161 paired: frac J1 agg better={frac_j1_better:.4f} '
         f'individual-worse-but-set-better frac={frac_ind_worse_set_better:.4f}')

    ch_j3 = paired.groupby('channel').agg(J0_D=('D_j0', 'mean'), J1_D=('D_j1', 'mean'),
                                          J0_C=('C_j0', 'mean'), J1_C=('C_j1', 'mean'),
                                          J0_Agg=('agg_mse10_j0', 'mean'), J1_Agg=('agg_mse10_j1', 'mean'))
    ch_j3.to_csv(j3dir / 'channel.csv')

    pd.DataFrame([{'arm': 'J0', **{k: macro_j0[k] for k in ('mean_pairwise_error_dot', 'mean_pairwise_error_cosine',
                                                             'frac_error_cosine_neg')}},
                 {'arm': 'J1', **{k: macro_j1[k] for k in ('mean_pairwise_error_dot', 'mean_pairwise_error_cosine',
                                                            'frac_error_cosine_neg')}}]
                ).to_csv(j3dir / 'pairwise_error.csv', index=False)
    pd.DataFrame([{'arm': 'J0', 'mixed_sign_fraction': macro_j0['mixed_sign_fraction'],
                  'mean_abs_mu_h': macro_j0['mean_abs_mu_h']},
                 {'arm': 'J1', 'mixed_sign_fraction': macro_j1['mixed_sign_fraction'],
                  'mean_abs_mu_h': macro_j1['mean_abs_mu_h']}]
                ).to_csv(j3dir / 'sign_cancellation.csv', index=False)

    # bootstrap (query-clustered, all 7 channels jointly)
    import numpy as np
    rng = np.random.default_rng(20260930)
    n_reps = 10000
    g = paired.groupby('query_start_idx')
    per_q = {
        'j0_D': g['D_j0'].mean(), 'j1_D': g['D_j1'].mean(),
        'j0_C': g['C_j0'].mean(), 'j1_C': g['C_j1'].mean(),
        'j0_Agg': g['agg_mse10_j0'].mean(), 'j1_Agg': g['agg_mse10_j1'].mean(),
        'j0_cos': g['mean_pairwise_error_cosine_j0'].mean(), 'j1_cos': g['mean_pairwise_error_cosine_j1'].mean(),
    }

    def boot(a1, a0):
        n = len(a1)
        diffs = np.empty(n_reps)
        for i in range(n_reps):
            idx = rng.integers(0, n, size=n)
            diffs[i] = (a1.values[idx] - a0.values[idx]).mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {'mean_diff': float(diffs.mean()), 'ci_2.5': float(lo), 'ci_97.5': float(hi),
                'ci_excludes_zero': bool(lo > 0 or hi < 0)}

    bootstrap_results = {
        'agg_mse10': boot(per_q['j1_Agg'], per_q['j0_Agg']), 'D': boot(per_q['j1_D'], per_q['j0_D']),
        'C': boot(per_q['j1_C'], per_q['j0_C']), 'mean_pairwise_error_cosine': boot(per_q['j1_cos'], per_q['j0_cos']),
        'n_queries': int(len(per_q['j0_D'])), 'n_reps': n_reps,
    }
    (j3dir / 'bootstrap.json').write_text(json.dumps(bootstrap_results, indent=2))
    print(f'[eval_l] FULL2161 bootstrap: {json.dumps(bootstrap_results, indent=2)}')

    # population comparison
    pop_rows = []
    for arm in ('J0', 'J1', 'K1', 'K2'):
        mp = macro_summary(dfs_p256[arm])
        mf = macro_summary(dfs_full[arm])
        for k_ in ('retmse10', 'agg_mse10', 'recall10', 'C'):
            pop_rows.append({'arm': arm, 'metric': k_, 'P256': mp[k_], 'FULL2161': mf[k_],
                             'rel_diff_pct': 100.0 * (mp[k_] - mf[k_]) / mf[k_]})
    pd.DataFrame(pop_rows).to_csv(OUT_DIR / 'population_comparison.csv', index=False)

    # revised verdict
    macro_k1_f, macro_k2_f = macro_summary(dfs_full['K1']), macro_summary(dfs_full['K2'])
    macro_k1_p, macro_k2_p = macro_summary(dfs_p256['K1']), macro_summary(dfs_p256['K2'])
    delta_ret_full = 100.0 * (macro_k2_f['retmse10'] - macro_j1['retmse10']) / macro_j1['retmse10']
    delta_agg_full = 100.0 * (macro_k2_f['agg_mse10'] - macro_j1['agg_mse10']) / macro_j1['agg_mse10']
    macro_j1_p = macro_summary(dfs_p256['J1'])
    delta_agg_p256 = 100.0 * (macro_k2_p['agg_mse10'] - macro_j1_p['agg_mse10']) / macro_j1_p['agg_mse10']

    j3_status = 'FULL-TEST CONFIRMED' if (macro_j1['D'] > macro_j0['D'] and macro_j1['C'] < macro_j0['C']
                                          and macro_j1['agg_mse10'] < macro_j0['agg_mse10'] and r_cross > 0.75
                                          and bootstrap_results['C']['ci_excludes_zero']) else 'PROBE-DEPENDENT'
    track_k_status = 'CONFIRMED' if delta_ret_full > 10.0 else 'OVERTURNED'

    q1_diffs = list(repro.get('J0_P256', {}).values()) + list(repro.get('J1_P256', {}).values())
    q2_diffs = list(repro.get('K1_FULL', {}).values()) + list(repro.get('K2_FULL', {}).values())
    revised = {
        'Q1_J0J1_P256_reproduction_pass': bool(max(q1_diffs, default=1) < 1e-5),
        'Q2_K1K2_FULL_reproduction_pass': bool(max(q2_diffs, default=1) < 1e-5),
        'Q3_D_J1_gt_D_J0_full': bool(macro_j1['D'] > macro_j0['D']),
        'Q4_C_J1_lt_C_J0_full': bool(macro_j1['C'] < macro_j0['C']),
        'Q5_Agg_J1_lt_Agg_J0_full': bool(macro_j1['agg_mse10'] < macro_j0['agg_mse10']),
        'Q6_R_cross_full': r_cross,
        'Q7_J3_verdict': j3_status,
        'Q8_K2_vs_J1_retmse_pct_change_full': delta_ret_full,
        'Q9_exceeds_10pct_threshold': bool(delta_ret_full > 10.0),
        'Q10_track_k_verdict': track_k_status,
        'Q11_K2_improves_agg_vs_J1_full': bool(macro_k2_f['agg_mse10'] < macro_j1['agg_mse10']),
        'Q12_K2_improves_agg_vs_J1_p256': bool(macro_k2_p['agg_mse10'] < macro_j1_p['agg_mse10']),
        'delta_agg_K2_vs_J1_full_pct': delta_agg_full,
        'delta_agg_K2_vs_J1_p256_pct': delta_agg_p256,
    }
    (OUT_DIR / 'revised_verdict.json').write_text(json.dumps(revised, indent=2))
    print(f'[eval_l] REVISED VERDICT: {json.dumps(revised, indent=2)}')
    print('[eval_l] done.')


if __name__ == '__main__':
    main()
