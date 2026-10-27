#!/usr/bin/env python3
"""TRACK-P100-RETRIEVAL-UTILIZATION-AUDIT01 -- PHASE A: Oracle Lambda
diagnostic. No new Stage1/Stage2 training anywhere in this script --
reuses the EXACT existing B(base)/R(retrieval)/Y(future) tensors
already cached by TRACK-P100-COARSE-TO-FINE-RETRIEVAL01's own
`scripts/build_p100_coarse_to_fine_cache01.py` + Base Forecaster, via
`scripts/train_r_stage2_lambda01.py::load_tensors` (imported,
unmodified -- the same function `scripts/eval_p100_coarse_to_fine_stage2_01.py`
already uses, so B/R/Y here are byte-identical to what produced the
published Stage2 numbers).

Stage2 semantics audited directly from that code before writing this
(spec section 5): `fused = B + lambda*(R-B)`, `lambda=sigmoid(a)` (one
scalar for the WHOLE cell -- "global"), `mse = ((fused-Y)**2).mean()`
over ALL (query, horizon-step, channel) elements jointly. B/R/Y are
all already in the same absolute-future-value space (no further
inverse-transform needed) -- confirmed by this session's own repeated
cross-checks (R's own retrieval-only MSE here always matches the
published `retrieval_only_agg_mse_test` and B's matches
`base_only_mse_test` exactly, reprinted below as part of each run's
own sanity output, not assumed).

Arms (section 7): A0 Base-only (lambda=0). A1 Current Global Lambda
(read from the EXISTING Stage2 `metrics.json`, never retrained here).
A2 Validation-Optimal Global Lambda -- closed form (MSE as a function
of ONE scalar lambda is a quadratic; the minimizer over the
unconstrained line is `mean(e*d)/mean(d*d)` with `d=R-B`, `e=Y-B`,
then clipped to [0,1] per spec's explicit range), computed on VAL,
applied FIXED to TEST. A3 per-query(-channel) continuous Oracle
Lambda -- same closed-form ratio but computed per-row instead of
globally, reads the TRUE future (diagnostic only, never deployable).
A4 Binary Oracle Gate -- per-row choice of lambda in {0,1}.

Granularity (section 8): 'query_channel' (primary) computes the A3/A4
ratio independently for every (query, channel) pair, summing only
over the horizon axis H. 'query' (secondary) sums over H AND channel
jointly per query. 'dataset_global' (secondary) sums over everything
-- mathematically identical to A2's own closed form, reported as a
cross-check (should equal A2 to within floating point)."""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.rng_control01 import set_global_seeds
from scripts.train_margutil01 import build_experiment
from scripts.train_r_stage2_lambda01 import load_tensors

EPS = 1e-8


def _pctl(t, q):
    return float(torch.quantile(t, q))


def _dist(t):
    return {'mean': float(t.mean()), 'std': float(t.std()), 'median': float(t.median()),
           'p10': _pctl(t, 0.10), 'p25': _pctl(t, 0.25), 'p75': _pctl(t, 0.75), 'p90': _pctl(t, 0.90)}


def closed_form_lambda(d, e, dims):
    """d=R-B, e=Y-B, both [...]; reduces over `dims`. Returns unclipped
    lambda* = sum(e*d)/sum(d*d) over those dims (equivalent to
    mean(e*d)/mean(d*d) -- the N cancels)."""
    num = (e * d).sum(dim=dims)
    den = (d * d).sum(dim=dims).clamp_min(EPS)
    return num / den


@torch.no_grad()
def run_cell(cli, device):
    set_global_seeds(cli.seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed,
    })
    channels = int(args.enc_in)
    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()

    _, B_va, R_va, Y_va = load_tensors(exp, base, cli.cache_dir, 'val', device, channels)
    _, B_te, R_te, Y_te = load_tensors(exp, base, cli.cache_dir, 'test', device, channels)

    current_lambda = json.loads(Path(cli.stage2_metrics).read_text())['lambda']

    out = {'cell': cli.cell, 'arm': cli.arm, 'current_lambda': current_lambda,
          'n_val': int(B_va.size(0)), 'n_test': int(B_te.size(0)), 'channels': channels}

    def mse(fused, Y):
        return float(((fused - Y) ** 2).mean())

    # ---- A0/A1 on test ----
    mse_base_test = mse(B_te, Y_te)
    mse_retrieval_test = mse(R_te, Y_te)
    fused_current = B_te + current_lambda * (R_te - B_te)
    mse_current_test = mse(fused_current, Y_te)
    out['mse_base_test'] = mse_base_test
    out['mse_retrieval_only_test'] = mse_retrieval_test
    out['mse_current_global_lambda_test'] = mse_current_test

    # ---- A2: validation-optimal GLOBAL lambda (closed form), fixed, applied to test ----
    d_va, e_va = R_va - B_va, Y_va - B_va
    lam_val_global = float(closed_form_lambda(d_va, e_va, dims=tuple(range(d_va.dim()))).clamp(0.0, 1.0))
    fused_val_global = B_te + lam_val_global * (R_te - B_te)
    mse_val_global_test = mse(fused_val_global, Y_te)
    out['lambda_val_optimal_global'] = lam_val_global
    out['mse_val_optimal_global_lambda_test'] = mse_val_global_test

    # ---- dataset_global oracle lambda on TEST itself (cross-check vs A2's closed form; diagnostic) ----
    d_te, e_te = R_te - B_te, Y_te - B_te
    lam_test_global_oracle = float(closed_form_lambda(d_te, e_te, dims=tuple(range(d_te.dim()))).clamp(0.0, 1.0))
    out['lambda_test_dataset_global_oracle'] = lam_test_global_oracle
    out['mse_test_dataset_global_oracle'] = mse(B_te + lam_test_global_oracle * (R_te - B_te), Y_te)

    # ---- per-(query,channel) and per-query oracle lambda + binary gate, on TEST ----
    # B_te/R_te/Y_te: [N, H, C]
    granularity_results = {}
    for gran, dims in (('query_channel', (1,)), ('query', (1, 2))):
        lam_oracle = closed_form_lambda(d_te, e_te, dims=dims).clamp(0.0, 1.0)  # [N,C] or [N]
        # broadcast back to [N,H,C] to fuse
        if gran == 'query_channel':
            lam_b = lam_oracle.unsqueeze(1)  # [N,1,C]
        else:
            lam_b = lam_oracle.unsqueeze(1).unsqueeze(-1)  # [N,1,1]
        fused_oracle = B_te + lam_b * d_te
        mse_oracle = mse(fused_oracle, Y_te)

        # per-row (query[,channel]) MSE of B alone vs R alone, for binary gate + "R better than B" fraction
        mse_b_row = (e_te ** 2).mean(dim=dims)          # [N,C] or [N]
        mse_r_row = ((Y_te - R_te) ** 2).mean(dim=dims)  # [N,C] or [N]
        r_better = (mse_r_row < mse_b_row)
        lam_binary = r_better.float()
        if gran == 'query_channel':
            lam_binary_b = lam_binary.unsqueeze(1)
        else:
            lam_binary_b = lam_binary.unsqueeze(1).unsqueeze(-1)
        fused_binary = B_te + lam_binary_b * d_te
        mse_binary = mse(fused_binary, Y_te)

        # "current fusion better/worse than B" per row, at the CURRENT global lambda
        fused_cur_row = B_te + current_lambda * d_te
        mse_cur_row = ((fused_cur_row - Y_te) ** 2).mean(dim=dims)
        current_better = (mse_cur_row < mse_b_row)

        lam_flat = lam_oracle.reshape(-1)
        granularity_results[gran] = {
            'mse_oracle_continuous': mse_oracle,
            'mse_oracle_binary': mse_binary,
            'oracle_lambda_dist': _dist(lam_flat),
            'frac_lambda_eq_0': float((lam_flat <= 1e-6).float().mean()),
            'frac_lambda_eq_1': float((lam_flat >= 1 - 1e-6).float().mean()),
            'frac_lambda_interior': float(((lam_flat > 1e-6) & (lam_flat < 1 - 1e-6)).float().mean()),
            'frac_R_better_than_B': float(r_better.float().mean()),
            'frac_current_fusion_better_than_B': float(current_better.float().mean()),
            'frac_current_fusion_worse_than_B': float((~current_better).float().mean()),
        }
    out['granularity'] = granularity_results

    # ---- headroom metrics (section 10), on TEST ----
    out['headroom'] = {
        'global_lambda_headroom_pct': (mse_current_test - mse_val_global_test) / mse_current_test * 100.0,
        'oracle_utilization_headroom_pct_query_channel':
            (mse_current_test - granularity_results['query_channel']['mse_oracle_continuous']) / mse_current_test * 100.0,
        'base_relative_oracle_gain_pct_query_channel':
            (mse_base_test - granularity_results['query_channel']['mse_oracle_continuous']) / mse_base_test * 100.0,
    }

    # ---- retrieval gain distribution g_q = MSE(B_q,Y_q) - MSE(R_q,Y_q), per (query,channel) ----
    g = ((Y_te - B_te) ** 2).mean(dim=1) - ((Y_te - R_te) ** 2).mean(dim=1)  # [N,C]
    g_flat = g.reshape(-1)
    out['retrieval_gain_distribution'] = {**_dist(g_flat), 'positive_fraction': float((g_flat > 0).float().mean())}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--stage2_metrics', required=True,
                    help='path to the EXISTING Stage2 metrics.json for this cell/arm (to read the current lambda, never retrained)')
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = run_cell(cli, device)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_{cli.arm}_oracle_lambda.json').write_text(json.dumps(out, indent=2))

    qc = out['granularity']['query_channel']
    print(f'[oracle_lambda] {cli.cell}/{cli.arm}: base={out["mse_base_test"]:.6f} '
         f'retrieval={out["mse_retrieval_only_test"]:.6f} current={out["mse_current_global_lambda_test"]:.6f} '
         f'(lambda={out["current_lambda"]:.4f}) val_opt_global={out["mse_val_optimal_global_lambda_test"]:.6f} '
         f'(lambda={out["lambda_val_optimal_global"]:.4f}) '
         f'oracle_qc={qc["mse_oracle_continuous"]:.6f} oracle_qc_binary={qc["mse_oracle_binary"]:.6f}')
    print(f'[oracle_lambda] {cli.cell}/{cli.arm}: headroom={out["headroom"]} ')
    print(f'[oracle_lambda] {cli.cell}/{cli.arm}: oracle_lambda_dist(qc)={qc["oracle_lambda_dist"]} '
         f'frac0={qc["frac_lambda_eq_0"]:.4f} frac1={qc["frac_lambda_eq_1"]:.4f} '
         f'frac_R_better={qc["frac_R_better_than_B"]:.4f}')


if __name__ == '__main__':
    main()
