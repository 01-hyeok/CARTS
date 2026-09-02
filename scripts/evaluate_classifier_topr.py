#!/usr/bin/env python3
"""STEP 1/5 -- reference methods on the shared pool, plus classifier Top-r.

No training. Everything here is scored on the same pool the learned selectors
use, so the comparison isolates how a candidate is *chosen*, not which pool it
came from.

    base                 no correction
    current_topk_avg     the retriever's own weights over Top-K  (today's CARTS)
    classifier_soft      P(U>0) as a weight over the whole pool  (previous step)
    classifier_top{r}    the r highest-probability candidates
    oracle_best_single   the pool's best candidate               (upper bound)
    global_utility       utility Top-K over the whole bank       (upper bound)

The classifier is the one already trained in the filtering pipeline, reused
here as a *ranker* rather than a gate -- the whole point of STEP 1 is to tell
those two uses apart.
"""

import argparse, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap
from utils.utility_selection import (
    EPS, build_selection_cache, forecast_from_selection, masked_utility,
    selection_metrics, utility_from_residuals,
)
from scripts.analyze_residual_oracle import prepare
from scripts.train_utility_classifier import PairClassifier

COLUMNS = [
    'dataset', 'pred_len', 'method', 'pool_m', 'top_r', 'split',
    'positive_at_1', 'selected_utility_at_1', 'oracle_pool_utility',
    'random_utility', 'utility_regret_at_1', 'selection_recovery_at_1',
    'top1_identity_accuracy', 'forecast_mse', 'forecast_mae', 'base_mse',
    'checkpoint',
]
TOP_RS = (1, 2, 3, 5, 10)


@torch.no_grad()
def classifier_scores(bundle, experiment, model, data, cache, device, chunk=128):
    if bundle is None:
        return None
    net = PairClassifier(bundle['dim'])
    net.load_state_dict(bundle['state_dict'])
    net = net.to(device).eval()
    out = torch.zeros_like(cache['utility'])
    for slot_i, c in enumerate(cache['targets']):
        sources = model.source_channels(c)
        bank = experiment.key_bank[c, sources.index(c)].to(device).float()
        for start in range(0, out.size(0), chunk):
            stop = min(start + chunk, out.size(0))
            idx = cache['pool_idx'][start:stop, slot_i].to(device)
            z_q = cache['z_query'][start:stop, slot_i].unsqueeze(1).to(device)
            out[start:stop, slot_i] = torch.sigmoid(net(z_q, bank[idx])).cpu()
    return out


def emit(rows, cache, scores, experiment, data, method, top_r, meta, csv, weights=None):
    row = selection_metrics(cache, scores, top_r)
    pred = forecast_from_selection(experiment, data, cache, scores, top_r, weights)
    mse, mae = mse_mae(pred, data['query_y'])
    row.update(meta); row.update({
        'method': method, 'top_r': top_r,
        'forecast_mse': mse, 'forecast_mae': mae,
    })
    rows.append(row)
    if csv:
        append_row(csv, row, COLUMNS)
    return row


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--classifier', default='')
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--csv', default='')
    a = p.parse_args()

    experiment, saved = load_stage2(a.checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory(); experiment._build_key_bank(force=True)
    device = experiment.device
    data = prepare(experiment, a.split, a.max_batches)
    cache = build_selection_cache(experiment, model, data, a.pool_m, a.alpha)

    base_mse, base_mae = mse_mae(data['query_base'], data['query_y'])
    meta = {'dataset': saved.data, 'pred_len': int(saved.pred_len),
            'pool_m': a.pool_m, 'split': a.split, 'base_mse': base_mse,
            'checkpoint': a.checkpoint}
    rows = []

    # base: no correction at all
    rows.append({**meta, 'method': 'base', 'top_r': 0,
                 'forecast_mse': base_mse, 'forecast_mae': base_mae})
    if a.csv:
        append_row(a.csv, rows[-1], COLUMNS)

    retriever = cache['retriever_score']
    # today's CARTS: retriever weights over its own Top-K
    emit(rows, cache, retriever, experiment, data, 'current_topk_avg', 10,
         meta, a.csv, weights=torch.softmax(retriever / 0.1, dim=-1))

    bundle = torch.load(a.classifier, map_location='cpu') if a.classifier else None
    probability = classifier_scores(bundle, experiment, model, data, cache, device)
    if probability is not None:
        emit(rows, cache, probability, experiment, data, 'classifier_soft',
             cache['utility'].size(-1), meta, a.csv, weights=probability)
        for r in TOP_RS:
            emit(rows, cache, probability, experiment, data,
                 f'classifier_top{r}', r, meta, a.csv)

    # upper bounds
    oracle = masked_utility(cache)
    emit(rows, cache, oracle, experiment, data, 'oracle_best_single', 1, meta, a.csv)
    emit(rows, cache, oracle, experiment, data, 'oracle_positive_avg',
         cache['utility'].size(-1), meta, a.csv,
         weights=(oracle > 0).float())

    global_pred = data['query_base'].clone()
    for c in range(data['query_y'].size(-1)):
        k_res = data['memory_residual'][:, :, c]
        for start in range(0, global_pred.size(0), 256):
            stop = min(start + 256, global_pred.size(0))
            mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
            u = utility_from_residuals(
                data['query_residual'][start:stop, :, c], k_res, a.alpha
            ).masked_fill(~mask.to(device), float('-inf'))
            idx = u.topk(min(10, k_res.size(0)), dim=-1).indices
            global_pred[start:stop, :, c] += a.alpha * k_res[idx].mean(1)
    g_mse, g_mae = mse_mae(global_pred, data['query_y'])
    rows.append({**meta, 'method': 'global_utility_oracle', 'top_r': 10,
                 'forecast_mse': g_mse, 'forecast_mae': g_mae})
    if a.csv:
        append_row(a.csv, rows[-1], COLUMNS)

    print(f"=== {saved.data}/{saved.pred_len} pool M={a.pool_m} [{a.split}] ===")
    for r in rows:
        print(f"  {r['method']:<24} r={r.get('top_r', 0):<3} "
              f"MSE={r['forecast_mse']:.4f}  MAE={r['forecast_mae']:.4f}  "
              f"pos@1={r.get('positive_at_1', float('nan')):.3f}  "
              f"recov={r.get('selection_recovery_at_1', float('nan')):.3f}")


if __name__ == '__main__':
    main()
