#!/usr/bin/env python3
"""EXP-RETRIEVAL-QUALITY-DIAG01 -- diagnostic only, NO training.

Question: why does adding retrieval fail to improve forecasting at ETTh1
H720 while it helps at H96? Decomposes the failure into (i) the quality of
the retrieved future `Y_ret` itself, (ii) whether `Y_ret` corrects the base
prediction in the RIGHT DIRECTION, (iii) whether an oracle mixing weight
would rescue it (gate/fusion bottleneck vs signal bottleneck).

Design rules honoured:
  * No Stage-1 training, no loss/scorer/hyperparameter/Oracle-definition
    changes, no full rerun. Existing checkpoints and retrieval caches are
    reused; only evaluation is computed.
  * ONE canonical Base-only checkpoint per horizon supplies `Y_base` for
    EVERY arm, so arms are never compared using their own jointly-trained
    base predictors.
  * The Set Oracle's aggregation weight uses ONE FIXED reference scorer
    (the canonical S0_wce Stage-1 encoder's cosine score), so the Oracle
    definition is checkpoint-independent -- this avoids the endogenous
    target problem recorded in `research/AUDIT_ORACLE_RANK_GAIN01.md`.
  * Query futures are used ONLY to define Oracle arms and to compute
    diagnostic upper bounds; they never enter a learned arm's selection.

Fusion convention analysed: `Y_final = (1-lam) Y_base + lam Y_ret`
(equivalently `Y_base + lam (Y_ret - Y_base)`), i.e. `mixture`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_scratch01 import encode_raw
from utils.dense_utility import candidate_weights
from utils.oracle_intervention import select_greedy_weighted_set

REF_S1 = {
    96: 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth',
    720: 'checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth',
}
KL_ASYM = {
    (96, 's2ls'): 'checkpoints/stage1/ETTh1/seq96_pred96/stage1_carts_s2ls_asym_kl_ETTh1_96_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_s2ls_asym_kl_ETTh1_sl96_pl96_0/checkpoint.pth',
    (96, 'e2'): 'checkpoints/stage1/ETTh1/seq96_pred96/stage1_carts_e2_asym_kl_ETTh1_96_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_e2_asym_kl_ETTh1_sl96_pl96_0/checkpoint.pth',
    (720, 's2ls'): 'checkpoints/stage1/ETTh1/seq720_pred720/stage1_carts_s2ls_asym_kl_ETTh1_720_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_s2ls_asym_kl_ETTh1_sl720_pl720_0/checkpoint.pth',
    (720, 'e2'): 'checkpoints/stage1/ETTh1/seq720_pred720/stage1_carts_e2_asym_kl_ETTh1_720_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_e2_asym_kl_ETTh1_sl720_pl720_0/checkpoint.pth',
}
CACHE_ARMS = {
    'Individual-TF-Asym': 'individual_tf_asymmetric',
    'Set-TF-Asym': 'set_tf_asymmetric',
    'Individual-OP-Asym': 'individual_onpolicy_asymmetric',
    'Set-OP-Asym': 'set_onpolicy_asymmetric',
}


def load_stage1(ckpt_path, pred_len, device, tag):
    """Rebuild a Stage-1 model from an EXISTING checkpoint (weights loaded).
    Uses the checkpoint's own saved args so the model's own retrieval
    metric (e.g. the trained asymmetric projection) is reconstructed."""
    ck = torch.load(ckpt_path, map_location='cpu')
    overrides = {'is_training': 0, 'model_id': f'diag_rq01_{tag}', 'des': 'diag',
                 'checkpoints': '/tmp/diag_rq01', 'pred_len': pred_len, 'seq_len': pred_len}
    exp, args = build_experiment(ckpt_path, overrides)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return exp, args, model


def arm_score(model, z_q, E):
    """The model's OWN production retrieval score: its trained
    `retrieval_metric` if it has one (asymmetric arms), else cosine."""
    m = getattr(model, 'retrieval_metric', None)
    if m is not None:
        return m.score(z_q, E)
    zq = torch.nn.functional.normalize(z_q, dim=-1)
    ek = torch.nn.functional.normalize(E, dim=-1)
    return torch.matmul(zq, ek.transpose(0, 1))


def weighted_aggregate(picks, score, futures, cand_mask, tau, uniform=False):
    """Production convention: alpha = softmax(score/tau) renormalised over
    the selected Top-K, then a weighted mean of the offset-adjusted futures."""
    if uniform:
        alpha = torch.full_like(picks, 1.0 / picks.size(1), dtype=futures.dtype)
    else:
        w = candidate_weights(score, cand_mask, tau)
        wp = w.gather(1, picks)
        alpha = wp / wp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    gathered = futures.gather(1, picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
    return (alpha.unsqueeze(-1) * gathered).sum(dim=1)


@torch.no_grad()
def compute_yret(kind, exp, args, model, ref_model, split, channels, k, tau, device, chunk):
    """Y_ret [N, H, C] for one arm, in loader order (shuffle=False).

    kind: 'model_topk'  -> the arm model's own score, static Top-K (no future)
          'oracle_ind'  -> Top-K by ascending true future MSE (diagnostic)
          'oracle_ind_uniform' -> same picks, uniform weights
          'oracle_set'  -> greedy weighted Set Oracle, FIXED reference score
    """
    _, loader = exp._get_data(flag=split, shuffle=False)
    rows = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        per_c = []
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            qf = batch_y[:, :, c]
            neg = torch.finfo(futures.dtype).min / 4

            if kind == 'model_topk':
                E = encode_raw(model, exp.memory_x, c)
                z_q = encode_raw(model, batch_x, c)
                score = arm_score(model, z_q, E)
                picks = score.masked_fill(~cand_mask, neg).topk(k, dim=-1).indices
                per_c.append(weighted_aggregate(picks, score, futures, cand_mask, tau))
            else:
                # fixed reference score -> Oracle definitions never depend on
                # any experiment's own (changing) encoder
                Eref = encode_raw(ref_model, exp.memory_x, c)
                zq_ref = encode_raw(ref_model, batch_x, c)
                ref_score = arm_score(ref_model, zq_ref, Eref)
                if kind == 'oracle_set':
                    picks = select_greedy_weighted_set(futures, qf, ref_score, cand_mask, k, tau)
                    per_c.append(weighted_aggregate(picks, ref_score, futures, cand_mask, tau))
                else:
                    d_i = ((futures - qf.unsqueeze(1)) ** 2).mean(dim=-1)
                    picks = (-d_i).masked_fill(~cand_mask, neg).topk(k, dim=-1).indices
                    per_c.append(weighted_aggregate(picks, ref_score, futures, cand_mask, tau,
                                                     uniform=(kind == 'oracle_ind_uniform')))
        rows.append(torch.stack(per_c, dim=-1).cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def compute_ybase(base_ckpt, exp, split, device, batch=64):
    """Canonical Y_base [N, H, C] from the ONE Base-only checkpoint."""
    ck = torch.load(base_ckpt, map_location='cpu')
    args_ = exp.args
    bh = BaseForecastHead(int(args_.seq_len), int(args_.pred_len), int(args_.enc_in),
                          mode='shared_target_linear').to(device)
    bh.load_state_dict(ck['base_head_state_dict'])
    bh.eval()
    _, loader = exp._get_data(flag=split, shuffle=False)
    rows, ys = [], []
    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.float().to(device)
        rows.append((bh(batch_x) + batch_x[:, -1:, :]).cpu())
        ys.append(batch_y.float().cpu())
    return torch.cat(rows, 0), torch.cat(ys, 0)


def flat_qc(t):
    """[N,H,C] -> [N*C, H] (one row per query x channel)."""
    N, H, C = t.shape
    return t.permute(0, 2, 1).reshape(N * C, H)


def metrics_for_arm(y_base, y_ret, y_true, y_base_val, y_ret_val, y_true_val, n_ch):
    """Metrics A-E for one arm (test), plus the val-fitted global lambda."""
    B, R, T = flat_qc(y_base), flat_qc(y_ret), flat_qc(y_true)
    e_base = ((B - T) ** 2).mean(-1)
    e_ret = ((R - T) ** 2).mean(-1)

    c_true, c_ret = T - B, R - B
    nt, nr = c_true.norm(dim=-1), c_ret.norm(dim=-1)
    ok = (nt > 1e-12) & (nr > 1e-12)
    cos = torch.zeros_like(nt)
    cos[ok] = (c_true[ok] * c_ret[ok]).sum(-1) / (nt[ok] * nr[ok])

    d = c_ret
    dd = (d * d).sum(-1)
    lam_raw = torch.where(dd > 1e-12, (c_true * d).sum(-1) / dd.clamp_min(1e-12), torch.zeros_like(dd))
    lam_star = lam_raw.clamp(0, 1)
    y_ol = B + lam_star.unsqueeze(-1) * d
    oracle_lambda_mse = ((y_ol - T) ** 2).mean().item()

    # val-fitted global lambda (no test leakage): closed form, then clipped
    Bv, Rv, Tv = flat_qc(y_base_val), flat_qc(y_ret_val), flat_qc(y_true_val)
    dv, ev = Rv - Bv, Tv - Bv
    lam_val = float(((ev * dv).sum() / (dv * dv).sum().clamp_min(1e-12)).clamp(0, 1))
    y_gl = B + lam_val * d
    global_mse = ((y_gl - T) ** 2).mean().item()

    # channel-wise val-fitted lambda
    lam_val_c, global_c_mse = [], 0.0
    NC = B.size(0)
    for c in range(n_ch):
        idx = torch.arange(c, NC, n_ch)
        dvc, evc = dv[torch.arange(c, Bv.size(0), n_ch)], ev[torch.arange(c, Bv.size(0), n_ch)]
        lc = float(((evc * dvc).sum() / (dvc * dvc).sum().clamp_min(1e-12)).clamp(0, 1))
        lam_val_c.append(lc)
        global_c_mse += ((B[idx] + lc * d[idx] - T[idx]) ** 2).mean().item()
    global_c_mse /= n_ch

    gain_full = e_base - e_ret                       # lambda = 1
    gain_gl = e_base - ((y_gl - T) ** 2).mean(-1)    # val-global lambda

    def pct(x, q):
        return float(np.percentile(x.numpy(), q))

    return {
        'base_mse': e_base.mean().item(), 'ret_mse': e_ret.mean().item(),
        'ret_gain_mean': gain_full.mean().item(), 'ret_gain_median': gain_full.median().item(),
        'ret_better_frac': (e_ret < e_base).float().mean().item(),
        'ret_worse_frac': (e_ret > e_base).float().mean().item(),
        'corr_cos_mean': cos[ok].mean().item(), 'corr_cos_median': cos[ok].median().item(),
        'pos_align_frac': (cos[ok] > 0).float().mean().item(),
        'neg_align_frac': (cos[ok] < 0).float().mean().item(),
        'strong_neg_frac': (cos[ok] < -0.5).float().mean().item(),
        'zero_norm_frac': (~ok).float().mean().item(),
        'oracle_lambda_mse': oracle_lambda_mse,
        'lambda_star_mean': lam_star.mean().item(), 'lambda_star_median': lam_star.median().item(),
        'lambda_eq0_frac': (lam_star <= 1e-8).float().mean().item(),
        'lambda_eq1_frac': (lam_star >= 1 - 1e-8).float().mean().item(),
        'lambda_interior_frac': ((lam_star > 1e-8) & (lam_star < 1 - 1e-8)).float().mean().item(),
        'lambda_val_global': lam_val, 'global_lambda_mse': global_mse,
        'lambda_val_channelwise': lam_val_c, 'global_lambda_channelwise_mse': global_c_mse,
        'helpful_frac_lam1': (gain_full > 0).float().mean().item(),
        'harmful_frac_lam1': (gain_full < 0).float().mean().item(),
        'mean_gain_helpful_lam1': gain_full[gain_full > 0].mean().item() if (gain_full > 0).any() else float('nan'),
        'mean_damage_harmful_lam1': gain_full[gain_full < 0].mean().item() if (gain_full < 0).any() else float('nan'),
        'gain_p5_lam1': pct(gain_full, 5), 'gain_p95_lam1': pct(gain_full, 95),
        'helpful_frac_glob': (gain_gl > 0).float().mean().item(),
        'harmful_frac_glob': (gain_gl < 0).float().mean().item(),
        'gain_p5_glob': pct(gain_gl, 5), 'gain_p95_glob': pct(gain_gl, 95),
        '_per_qc': {'e_base': e_base, 'e_ret': e_ret, 'cos': cos, 'ok': ok,
                    'lam_star': lam_star, 'gain_full': gain_full, 'gain_gl': gain_gl},
    }


def horizon_bins(y_base, y_ret, y_true, lam_val, bins):
    out = []
    for lo, hi in bins:
        b, r, t = y_base[:, lo:hi, :], y_ret[:, lo:hi, :], y_true[:, lo:hi, :]
        B, R, T = flat_qc(b), flat_qc(r), flat_qc(t)
        eb, er = ((B - T) ** 2).mean(-1), ((R - T) ** 2).mean(-1)
        ct, cr = T - B, R - B
        nt, nr = ct.norm(dim=-1), cr.norm(dim=-1)
        ok = (nt > 1e-12) & (nr > 1e-12)
        cos = ((ct[ok] * cr[ok]).sum(-1) / (nt[ok] * nr[ok])).mean().item() if ok.any() else float('nan')
        mix = ((B + lam_val * (R - B) - T) ** 2).mean().item()
        out.append({'bin': f'{lo+1}-{hi}', 'base_mse': eb.mean().item(), 'ret_mse': er.mean().item(),
                    'mixture_mse': mix, 'corr_cos_mean': cos,
                    'ret_better_frac': (er < eb).float().mean().item()})
    return out


def channelwise(y_base, y_ret, y_true, lam_val_c, n_ch):
    rows = []
    for c in range(n_ch):
        b, r, t = y_base[:, :, c], y_ret[:, :, c], y_true[:, :, c]
        eb, er = ((b - t) ** 2).mean(-1), ((r - t) ** 2).mean(-1)
        ct, cr = t - b, r - b
        nt, nr = ct.norm(dim=-1), cr.norm(dim=-1)
        ok = (nt > 1e-12) & (nr > 1e-12)
        cos = ((ct[ok] * cr[ok]).sum(-1) / (nt[ok] * nr[ok])) if ok.any() else torch.zeros(1)
        dd = (cr * cr).sum(-1)
        lam = torch.where(dd > 1e-12, (ct * cr).sum(-1) / dd.clamp_min(1e-12), torch.zeros_like(dd)).clamp(0, 1)
        ol = ((b + lam.unsqueeze(-1) * cr - t) ** 2).mean().item()
        lc = lam_val_c[c]
        rows.append({'channel': c, 'base_mse': eb.mean().item(), 'ret_mse': er.mean().item(),
                     'ret_better_frac': (er < eb).float().mean().item(),
                     'corr_cos_mean': cos.mean().item(), 'oracle_lambda_mse': ol,
                     'lambda_eq0_frac': (lam <= 1e-8).float().mean().item(),
                     'lambda_val_channel': lc,
                     'global_lambda_mse': ((b + lc * cr - t) ** 2).mean().item()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_len', type=int, required=True, choices=[96, 720])
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--out_dir', default='results/EXP-RETRIEVAL-QUALITY-DIAG01')
    cli = ap.parse_args()
    H = cli.pred_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(cli.out_dir); out.mkdir(parents=True, exist_ok=True)

    ref_path = REF_S1[H]
    exp, args, ref_model = load_stage1(ref_path, H, device, 'ref')
    exp._ensure_memory()
    channels = list(ref_model.target_channels())
    n_ch = len(channels)
    tau = float(args.tau_topk)
    k = cli.top_k

    base_ckpt = f'checkpoints/exp_oracle_rank_gain01/stage2/carts_rank_gain01_stage2_base_only_ETTh1_H{H}/checkpoint.pth'
    y_base_te, y_true_te = compute_ybase(base_ckpt, exp, 'test', device)
    y_base_va, y_true_va = compute_ybase(base_ckpt, exp, 'val', device)
    base_mse_te = ((y_base_te - y_true_te) ** 2).mean().item()
    print(f'[diag] canonical Base-only test MSE = {base_mse_te:.6f} '
          f'(expected {0.39298 if H==96 else 0.48279:.5f})')

    manifest = {'pred_len': H, 'top_k': k, 'tau_topk': tau, 'device': str(device),
                'canonical_base_ckpt': base_ckpt, 'canonical_base_test_mse': base_mse_te,
                'fixed_reference_scorer_for_oracles': ref_path,
                'fusion_convention': 'Y_final = (1-lam)*Y_base + lam*Y_ret  [mixture]',
                'n_test_queries': int(y_true_te.size(0)), 'n_val_queries': int(y_true_va.size(0)),
                'shape_convention': '[N, H, C]', 'arms': {}}

    arms = {}
    # --- arms from existing caches (no recomputation) ---
    for label, cdir in CACHE_ARMS.items():
        p = Path(f'checkpoints/exp_oracle_rank_gain01/retrieval_cache/{cdir}_ETTh1_H{H}')
        if not (p / 'test.pt').exists():
            print(f'[diag] SKIP {label}: cache missing at {p}'); continue
        te = torch.load(p / 'test.pt', map_location='cpu')
        va = torch.load(p / 'val.pt', map_location='cpu')
        assert torch.equal(te['Y_q'], y_true_te), f'{label}: cache Y_q != canonical Y_true'
        arms[label] = (te['Y_ret'], va['Y_ret'])
        manifest['arms'][label] = {'source': 'cache', 'path': str(p)}

    # --- KL + Asymmetric (regenerated in the CURRENT harness) ---
    for variant in ('s2ls', 'e2'):
        cp = KL_ASYM.get((H, variant))
        if cp is None or not Path(cp).exists():
            print(f'[diag] SKIP KL+Asym[{variant}]: checkpoint missing'); continue
        e2exp, e2args, klm = load_stage1(cp, H, device, f'kl_{variant}')
        e2exp._ensure_memory()
        yr_te = compute_yret('model_topk', e2exp, e2args, klm, ref_model, 'test', channels, k, tau, device, cli.chunk_size)
        yr_va = compute_yret('model_topk', e2exp, e2args, klm, ref_model, 'val', channels, k, tau, device, cli.chunk_size)
        arms[f'KL+Asym[{variant}]'] = (yr_te, yr_va)
        manifest['arms'][f'KL+Asym[{variant}]'] = {'source': 'regenerated', 'checkpoint': cp,
                                                    'metric': str(getattr(klm, 'retrieval_metric_kind', 'cosine'))}
        print(f'[diag] KL+Asym[{variant}] Y_ret regenerated')

    # --- true Oracles (fixed reference scorer; future used for definition only) ---
    for label, kind in [('Oracle-Individual', 'oracle_ind'),
                        ('Oracle-Individual(uniform)', 'oracle_ind_uniform'),
                        ('Oracle-Set', 'oracle_set')]:
        yr_te = compute_yret(kind, exp, args, ref_model, ref_model, 'test', channels, k, tau, device, cli.chunk_size)
        yr_va = compute_yret(kind, exp, args, ref_model, ref_model, 'val', channels, k, tau, device, cli.chunk_size)
        arms[label] = (yr_te, yr_va)
        manifest['arms'][label] = {'source': 'computed', 'kind': kind,
                                   'weighting': 'uniform' if 'uniform' in label else 'score-weighted(fixed ref)'}
        print(f'[diag] {label} Y_ret computed')

    # --- metrics ---
    summary, chan_rows, hbin_rows, qrows = [], [], [], []
    bins = ([(0, 24), (24, 48), (48, 72), (72, 96)] if H == 96
            else [(0, 96), (96, 192), (192, 336), (336, 720)])
    for label, (yr_te, yr_va) in arms.items():
        m = metrics_for_arm(y_base_te, yr_te, y_true_te, y_base_va, yr_va, y_true_va, n_ch)
        per = m.pop('_per_qc')
        m['arm'] = label
        summary.append(m)
        for r in channelwise(y_base_te, yr_te, y_true_te, m['lambda_val_channelwise'], n_ch):
            r['arm'] = label; chan_rows.append(r)
        for r in horizon_bins(y_base_te, yr_te, y_true_te, m['lambda_val_global'], bins):
            r['arm'] = label; hbin_rows.append(r)
        for i in range(per['e_base'].numel()):
            qrows.append({'arm': label, 'qc_index': i,
                          'e_base': per['e_base'][i].item(), 'e_ret': per['e_ret'][i].item(),
                          'corr_cos': per['cos'][i].item(), 'lambda_star': per['lam_star'][i].item(),
                          'gain_lam1': per['gain_full'][i].item(), 'gain_global': per['gain_gl'][i].item()})
        print(f"  {label:30s} ret_mse={m['ret_mse']:.5f} cos={m['corr_cos_mean']:+.4f} "
              f"oracle_lam_mse={m['oracle_lambda_mse']:.5f} lam0%={m['lambda_eq0_frac']:.1%} "
              f"glob_lam={m['lambda_val_global']:.3f} glob_mse={m['global_lambda_mse']:.5f}")

    # --- stepwise delta (H720 primarily) ---
    step_rows = []
    for label, (yr_te, _) in arms.items():
        eb = ((y_base_te - y_true_te) ** 2).mean(dim=(0, 2))
        er = ((yr_te - y_true_te) ** 2).mean(dim=(0, 2))
        for h in range(H):
            step_rows.append({'arm': label, 'step': h + 1, 'base_mse': eb[h].item(),
                              'ret_mse': er[h].item(), 'delta_mse': (er[h] - eb[h]).item()})

    import csv
    def dump(rows, path, drop=()):
        if not rows: return
        keys = [k for k in rows[0].keys() if k not in drop]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)

    dump(summary, out / f'H{H}_summary.csv', drop=('lambda_val_channelwise',))
    dump(chan_rows, out / f'H{H}_channelwise.csv')
    dump(qrows, out / f'H{H}_querywise.csv')
    dump(hbin_rows, out / f'H{H}_horizon_bins.csv')
    dump(step_rows, out / f'H{H}_stepwise.csv')
    with open(out / f'H{H}_artifact_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f'[diag] written to {out}')


if __name__ == '__main__':
    main()
