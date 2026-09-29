#!/usr/bin/env python3
"""TRACK-J2-KEY-UPDATE-DECOMPOSITION01 -- post-hoc Orthogonal Procrustes
diagnostic (spec sections 11-12). NOT a training arm -- runs only on
TRACK-J's existing A0 best checkpoint (epoch 10).

Question: how much of St0/S0t's degradation (reported in TRACK-J) is a
simple global rotation/reflection mismatch between the initial (E0) and
current (Et) embedding coordinate systems, vs a genuine nonlinear
geometry change?

R* = argmin_{R^T R = I} ||Zt R - Z0||_F^2, fit via SVD of Zt^T Z0
(U, S, Vh = svd(Zt^T Z0); R* = U @ Vh), using ONLY train-memory candidate
embeddings (Z0 = E0(memory), Zt = Et(memory)) -- val/test futures never
touch the fit.

P1 aligned-St0 = cos(Et(Xq) @ R*, E0(Xk))
P2 aligned-S0t = cos(E0(Xq), Et(Xk) @ R*)

Evaluated on the SAME deterministic probe set TRACK-J used
(`results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/probe_query_ids.json`,
reproduced here and checked for exact match) and on the test split.
"""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_j_shared_encoder_drift01 import (
    build_model, build_probe_set, precompute_probe_fixed, state_hash, variant_metrics,
)

EXPECTED_J0_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'
J0_CKPT = 'checkpoints/track_j_shared_encoder_drift01/ETTh1_720/checkpoint.pth'
J0_PROBE_IDS = 'results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/probe_query_ids.json'


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


def fit_procrustes(z0, zt):
    u, s, vh = torch.linalg.svd(zt.T @ z0)
    r_star = u @ vh
    return r_star, s


def evaluate(exp, args, model_q, model_k, r_star, probe, fixed, channels, top_k, align_side, device):
    rows = []
    with torch.no_grad():
        for c in channels:
            zq = encode_raw(model_q, probe['x'], c)
            zk = encode_raw(model_k, exp.memory_x, c)
            if align_side == 'query':  # P1: aligned-St0 -- rotate the Et query into E0 space
                zq = zq @ r_star
            elif align_side == 'key':  # P2: aligned-S0t -- rotate the Et key into E0 space
                zk = zk @ r_star
            s = arm_score(zq, zk, None)
            m, _ = variant_metrics(s, probe['cand_mask'], fixed[c], top_k)
            rows.append({'channel': c, **m})
    return rows


def main():
    cli = _Cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))

    assert state_hash(model) == EXPECTED_J0_INIT_HASH, '[ISSUE][ABORT] init hash mismatch'
    import copy
    E0 = copy.deepcopy(model).to(device)
    E0.eval()

    ckpt_path = REPO_ROOT / J0_CKPT
    bl = torch.load(ckpt_path, map_location=device)
    Et = copy.deepcopy(model).to(device)
    Et.load_state_dict(bl['model_state_dict'])
    Et.eval()
    print(f'[procrustes] loaded J0 best checkpoint epoch={bl.get("epoch")}')

    # ---- fit R* per channel on TRAIN-MEMORY candidate embeddings only ----
    r_stars = {}
    fit_report = {}
    with torch.no_grad():
        for c in channels:
            z0 = encode_raw(E0, exp.memory_x, c)
            zt = encode_raw(Et, exp.memory_x, c)
            r_star, sv = fit_procrustes(z0, zt)
            r_stars[c] = r_star
            residual_before = float((zt - z0).norm())
            residual_after = float((zt @ r_star - z0).norm())
            fit_report[c] = {'residual_before': residual_before, 'residual_after': residual_after,
                             'residual_reduction_pct': 100.0 * (1 - residual_after / max(residual_before, 1e-9)),
                             'singular_values_sum': float(sv.sum()), 'is_orthogonal_check':
                             float((r_star.T @ r_star - torch.eye(r_star.size(0), device=device)).abs().max())}
            print(f'[procrustes] channel={c} residual {residual_before:.3f} -> {residual_after:.3f} '
                 f'({fit_report[c]["residual_reduction_pct"]:.1f}% reduction) '
                 f'orthogonality_check={fit_report[c]["is_orthogonal_check"]:.2e}')

    out_dir = Path('results/TRACK-J2-KEY-UPDATE-DECOMPOSITION01/ETTh1_720/procrustes')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'alignment_metrics.json').write_text(json.dumps(fit_report, indent=2))

    # ---- probe set: reproduce TRACK-J's own deterministic probe, verify exact match ----
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe = build_probe_set(exp, val_loader, 256, device)
    j0_probe_ids = json.loads((REPO_ROOT / J0_PROBE_IDS).read_text())
    assert probe['start_idx'].tolist() == j0_probe_ids['start_idx'], (
        '[ISSUE][ABORT] reconstructed probe does not match TRACK-J\'s own probe_query_ids.json')
    print('[procrustes] probe set matches TRACK-J\'s own probe_query_ids.json exactly')
    fixed = precompute_probe_fixed(exp, args, probe, channels, device, cli.chunk_size)

    def eval_multi_r(model_q, model_k, probe_, fixed_, side):
        rows = []
        with torch.no_grad():
            for c in channels:
                zq = encode_raw(model_q, probe_['x'], c)
                zk = encode_raw(model_k, exp.memory_x, c)
                if side == 'query':
                    zq = zq @ r_stars[c]
                elif side == 'key':
                    zk = zk @ r_stars[c]
                s = arm_score(zq, zk, None)
                m, _ = variant_metrics(s, probe_['cand_mask'], fixed_[c], cli.top_k)
                rows.append({'channel': c, **m})
        return rows

    # baselines (unaligned), for direct comparison
    st0_raw = eval_multi_r(Et, E0, probe, fixed, side='none')
    s0t_raw = eval_multi_r(E0, Et, probe, fixed, side='none')
    stt_raw = eval_multi_r(Et, Et, probe, fixed, side='none')
    s00_raw = eval_multi_r(E0, E0, probe, fixed, side='none')
    p1_aligned = eval_multi_r(Et, E0, probe, fixed, side='query')   # aligned St0
    p2_aligned = eval_multi_r(E0, Et, probe, fixed, side='key')     # aligned S0t

    def avg(rows, k='retmse10'):
        return sum(r[k] for r in rows) / len(rows)

    val_summary = {
        'S00': avg(s00_raw), 'St0_raw': avg(st0_raw), 'S0t_raw': avg(s0t_raw), 'Stt_raw': avg(stt_raw),
        'St0_aligned_P1': avg(p1_aligned), 'S0t_aligned_P2': avg(p2_aligned),
    }
    print('[procrustes] VAL retMSE@10:', json.dumps(val_summary, indent=2))
    (out_dir / 'aligned_val_metrics.json').write_text(json.dumps(
        {'summary_retmse10': val_summary, 'per_channel': {
            'St0_raw': st0_raw, 'S0t_raw': s0t_raw, 'Stt_raw': stt_raw, 'S00': s00_raw,
            'St0_aligned_P1': p1_aligned, 'S0t_aligned_P2': p2_aligned}}, indent=2))

    test_probe = build_probe_set(exp, test_loader, 256, device)
    test_fixed = precompute_probe_fixed(exp, args, test_probe, channels, device, cli.chunk_size)
    st0_raw_t = eval_multi_r(Et, E0, test_probe, test_fixed, side='none')
    s0t_raw_t = eval_multi_r(E0, Et, test_probe, test_fixed, side='none')
    stt_raw_t = eval_multi_r(Et, Et, test_probe, test_fixed, side='none')
    s00_raw_t = eval_multi_r(E0, E0, test_probe, test_fixed, side='none')
    p1_aligned_t = eval_multi_r(Et, E0, test_probe, test_fixed, side='query')
    p2_aligned_t = eval_multi_r(E0, Et, test_probe, test_fixed, side='key')
    test_summary = {
        'S00': avg(s00_raw_t), 'St0_raw': avg(st0_raw_t), 'S0t_raw': avg(s0t_raw_t), 'Stt_raw': avg(stt_raw_t),
        'St0_aligned_P1': avg(p1_aligned_t), 'S0t_aligned_P2': avg(p2_aligned_t),
    }
    print('[procrustes] TEST retMSE@10:', json.dumps(test_summary, indent=2))
    (out_dir / 'aligned_test_metrics.json').write_text(json.dumps(
        {'summary_retmse10': test_summary, 'per_channel': {
            'St0_raw': st0_raw_t, 'S0t_raw': s0t_raw_t, 'Stt_raw': stt_raw_t, 'S00': s00_raw_t,
            'St0_aligned_P1': p1_aligned_t, 'S0t_aligned_P2': p2_aligned_t}}, indent=2))
    print('[procrustes] done.')


if __name__ == '__main__':
    main()
