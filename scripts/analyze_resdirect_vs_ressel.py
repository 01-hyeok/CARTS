#!/usr/bin/env python3
"""Is retrieval complementary to direct residual correction, per query?

ResDirect beats ResSel on average. That does not settle whether some queries
genuinely need the historical correction. This asks it directly, reusing the
already-trained residual selector -- nothing is retrained.

Two controls make the answer interpretable, because the headline number alone
cannot decide anything:

  shuffle floor   an oracle min over two predictors is *always* better than
                  both, and two equally-good predictors with independent noise
                  already produce a large apparent "gain". Shuffling ResSel
                  across queries keeps its error distribution and destroys only
                  the query correspondence, so the gain it still shows is the
                  noise floor the real gain has to clear.

  identifiability a regime that exists but cannot be recognized from observable
                  past features is not actionable. Delta(q) is correlated
                  against past-only statistics here; no router is trained.

Query futures appear only in the oracle selection and the error terms.
"""

import argparse, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap
from utils.utility_selection import (
    build_selection_cache, forecast_from_selection, utility_from_residuals,
)
from scripts.analyze_residual_oracle import prepare
from scripts.train_query_residual_selector import ResidualPredictor

EPS = 1e-8
FEATURES = [
    'past_mean', 'past_std', 'past_var', 'past_last', 'past_slope',
    'past_mean_abs_diff', 'past_acf1',
    'base_forecast_magnitude', 'predicted_residual_norm',
]
SUMMARY_COLUMNS = [
    'dataset', 'pred_len', 'queries',
    'base_mse', 'base_mae', 'current_carts_mse', 'current_carts_mae',
    'resdirect_mse', 'resdirect_mae', 'ressel_mse', 'ressel_mae',
    'oracle_hybrid_mse', 'oracle_hybrid_mae',
    'ressel_win_rate', 'resdirect_win_rate',
    'mean_sel_gain_when_sel_wins', 'median_sel_gain_when_sel_wins',
    'p90_sel_gain_when_sel_wins', 'mean_direct_gain_when_direct_wins',
    'oracle_hybrid_gain_vs_resdirect', 'oracle_hybrid_gain_ratio',
    'shuffled_hybrid_mse', 'shuffled_hybrid_gain_ratio', 'excess_gain_ratio',
    'max_abs_delta_feature_corr', 'best_delta_feature', 'verdict', 'checkpoint',
]


def per_query_mse(pred, true):
    """[Q] mean squared error over horizon and channels for each query."""
    return (pred - true).square().mean(dim=(1, 2))


def per_query_mae(pred, true):
    return (pred - true).abs().mean(dim=(1, 2))


@torch.no_grad()
def past_features(data, predicted_residual):
    """Observable-at-inference query descriptors; no future is touched."""
    x = data['query_x']                      # [Q, L, C]
    length = x.size(1)
    t = torch.arange(length, dtype=x.dtype, device=x.device)
    t = (t - t.mean()) / (t.std() + EPS)
    centered = x - x.mean(1, keepdim=True)
    diff = x[:, 1:] - x[:, :-1]
    lag = (centered[:, 1:] * centered[:, :-1]).mean(1) / (x.var(1) + EPS)
    return {
        'past_mean': x.mean((1, 2)),
        'past_std': x.std(1).mean(-1),
        'past_var': x.var(1).mean(-1),
        'past_last': x[:, -1].mean(-1),
        'past_slope': (centered * t.view(1, -1, 1)).mean(1).mean(-1),
        'past_mean_abs_diff': diff.abs().mean((1, 2)),
        'past_acf1': lag.mean(-1),
        'base_forecast_magnitude': data['query_base'].abs().mean((1, 2)),
        'predicted_residual_norm': predicted_residual.square().mean((1, 2)).sqrt(),
    }


def correlate(a, b):
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + EPS))


@torch.no_grad()
def analyse(stage2_ckpt, selector_ckpt, max_batches=0, seed=0, tau=0.1):
    experiment, args = load_stage2(stage2_ckpt)
    model = unwrap(experiment.model)
    experiment._ensure_memory(); experiment._build_key_bank(force=True)
    device = experiment.device

    bundle = torch.load(selector_ckpt, map_location='cpu')
    predictor = ResidualPredictor(bundle['seq_len'], bundle['pred_len'])
    predictor.load_state_dict(bundle['state_dict'])
    predictor = predictor.to(device).eval()
    pool_m, alpha = bundle['pool_m'], bundle['alpha']

    data = prepare(experiment, 'test', max_batches)
    cache = build_selection_cache(experiment, model, data, pool_m, alpha)
    true = data['query_y']

    predicted_residual = predictor(data['query_x'])
    direct = data['query_base'] + alpha * predicted_residual

    scores = torch.zeros_like(cache['utility'])
    for slot_i, c in enumerate(cache['targets']):
        k_res = data['memory_residual'][:, :, c]
        u = utility_from_residuals(predicted_residual[:, :, c], k_res, alpha)
        scores[:, slot_i] = u.gather(1, cache['pool_idx'][:, slot_i].to(device)).cpu()
    selected = forecast_from_selection(experiment, data, cache, scores, top_r=1)

    # today's CARTS on the same pool, for a common reference point
    retriever = cache['retriever_score']
    current = forecast_from_selection(
        experiment, data, cache, retriever, top_r=10,
        weights=torch.softmax(retriever / tau, dim=-1))

    e_direct = per_query_mse(direct, true)
    e_sel = per_query_mse(selected, true)
    delta = e_direct - e_sel                       # >0 means ResSel wins
    sel_wins = delta > 0

    hybrid = torch.where(sel_wins.view(-1, 1, 1), selected, direct)
    # Shuffle keeps ResSel's marginal error distribution and removes only the
    # query pairing, which is exactly the part "complementary regime" claims.
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(selected.size(0), generator=generator)
    shuffled = selected[permutation]
    e_shuffled = per_query_mse(shuffled, true)
    shuffled_hybrid = torch.where(
        (e_direct - e_shuffled > 0).view(-1, 1, 1), shuffled, direct)

    base_mse, base_mae = mse_mae(data['query_base'], true)
    cur_mse, cur_mae = mse_mae(current, true)
    d_mse, d_mae = mse_mae(direct, true)
    s_mse, s_mae = mse_mae(selected, true)
    h_mse, h_mae = mse_mae(hybrid, true)
    sh_mse, _ = mse_mae(shuffled_hybrid, true)

    gain_ratio = (d_mse - h_mse) / (d_mse + EPS)
    shuffled_ratio = (d_mse - sh_mse) / (d_mse + EPS)

    features = past_features(data, predicted_residual)
    correlations = {name: correlate(value.cpu(), delta.cpu())
                    for name, value in features.items()}
    best_feature = max(correlations, key=lambda k: abs(correlations[k]))

    sel_gain = delta[sel_wins]
    dir_gain = (-delta)[~sel_wins]
    q = lambda t, p: float(t.quantile(p)) if t.numel() else float('nan')

    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len),
        'queries': int(delta.numel()),
        'base_mse': base_mse, 'base_mae': base_mae,
        'current_carts_mse': cur_mse, 'current_carts_mae': cur_mae,
        'resdirect_mse': d_mse, 'resdirect_mae': d_mae,
        'ressel_mse': s_mse, 'ressel_mae': s_mae,
        'oracle_hybrid_mse': h_mse, 'oracle_hybrid_mae': h_mae,
        'ressel_win_rate': float(sel_wins.float().mean()),
        'resdirect_win_rate': float((~sel_wins).float().mean()),
        'mean_sel_gain_when_sel_wins': float(sel_gain.mean()) if sel_gain.numel() else 0.0,
        'median_sel_gain_when_sel_wins': q(sel_gain, 0.5),
        'p90_sel_gain_when_sel_wins': q(sel_gain, 0.9),
        'mean_direct_gain_when_direct_wins': float(dir_gain.mean()) if dir_gain.numel() else 0.0,
        'oracle_hybrid_gain_vs_resdirect': d_mse - h_mse,
        'oracle_hybrid_gain_ratio': gain_ratio,
        'shuffled_hybrid_mse': sh_mse,
        'shuffled_hybrid_gain_ratio': shuffled_ratio,
        # The part of the hybrid gain that survives the noise floor.
        'excess_gain_ratio': gain_ratio - shuffled_ratio,
        'max_abs_delta_feature_corr': max(abs(v) for v in correlations.values()),
        'best_delta_feature': best_feature,
        'checkpoint': stage2_ckpt,
    }
    row.update({f'delta_corr_{k}': v for k, v in correlations.items()})
    row['verdict'] = classify(row)

    query_rows = {
        'query_index': torch.arange(delta.numel()),
        'e_direct': e_direct.cpu(), 'e_sel': e_sel.cpu(), 'delta': delta.cpu(),
        'sel_wins': sel_wins.cpu().int(),
    }
    query_rows.update({k: v.cpu() for k, v in features.items()})
    return row, query_rows


def classify(row, floor=0.02, signal=0.10):
    """Real gain has to clear the shuffle floor before it means anything."""
    if row['excess_gain_ratio'] < floor:
        return 'RETRIEVAL_REDUNDANT'
    if row['max_abs_delta_feature_corr'] < signal:
        return 'COMPLEMENTARY_BUT_UNIDENTIFIABLE'
    return 'RETRIEVAL_COMPLEMENTARY'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selector', required=True)
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--csv', default='')
    p.add_argument('--query_csv', default='')
    a = p.parse_args()

    row, queries = analyse(a.checkpoint, a.selector, a.max_batches)
    print(f"=== {row['dataset']}/{row['pred_len']} ResDirect vs ResSel ===")
    for key in SUMMARY_COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    print('  delta correlations:', {k[len('delta_corr_'):]: round(v, 3)
                                    for k, v in row.items() if k.startswith('delta_corr_')})
    if a.csv:
        append_row(a.csv, row, SUMMARY_COLUMNS + [f'delta_corr_{f}' for f in FEATURES])
    if a.query_csv:
        import csv as _csv
        path = Path(a.query_csv); path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(queries)
        with open(path, 'w', newline='') as handle:
            writer = _csv.writer(handle)
            writer.writerow(['dataset', 'pred_len'] + keys)
            for i in range(queries['delta'].numel()):
                writer.writerow([row['dataset'], row['pred_len']]
                                + [float(queries[k][i]) for k in keys])


if __name__ == '__main__':
    main()
