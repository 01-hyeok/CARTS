#!/usr/bin/env python3
"""STEP 3 -- is "similar residual" a better retrieval target than "similar future"?

Stage-1 currently retrieves candidates whose *future* resembles the query's.
What forecasting actually needs is a candidate whose *error* resembles the
query's, because that error is what the correction adds back.

    R_i = Y_i - Yhat_i^base        base forecast from past only

  future oracle    Top-K by MSE(Y_q, Y_k), predict the retrieved futures
  residual oracle  Top-K by MSE(R_q, R_k), predict base + alpha * mean(R_k)

Both are Oracles: they use query futures to *select*, which no deployable model
can do. They are upper bounds that say which target is worth learning, not
models. Base forecasts never see a future -- neither the query's nor a
candidate's.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import (  # noqa: E402
    ALPHA_GRID, append_row, base_forecast, load_stage2, mse_mae, unwrap,
)

COLUMNS = [
    'dataset', 'pred_len', 'top_k',
    'base_mse', 'base_mae',
    'future_oracle_mse', 'future_oracle_mae',
    'residual_oracle_uniform_alpha1_mse', 'residual_oracle_uniform_alpha1_mae',
    'residual_oracle_weighted_alpha1_mse', 'residual_oracle_weighted_alpha1_mae',
    'residual_oracle_best_alpha', 'residual_oracle_best_mse', 'residual_oracle_best_mae',
    'future_oracle_gain', 'residual_oracle_gain', 'verdict', 'checkpoint',
]


@torch.no_grad()
def prepare(experiment, split, max_batches=0):
    """Base forecasts and residuals for the split's queries and every candidate."""
    experiment._ensure_memory()
    model = unwrap(experiment.model)
    device = experiment.device

    memory_x = torch.from_numpy(experiment.memory_bank.memory_x).float().to(device)
    memory_y = experiment.memory_y.to(device)            # [N, pred_len, C]
    # Candidate residuals: candidate future minus the base forecast made from
    # the candidate's own past. No future is ever fed to the base head.
    memory_base = base_forecast(model, memory_x)
    memory_residual = memory_y - memory_base

    _, loader = experiment._get_data(flag=split, shuffle=False)
    q = {'x': [], 'y': [], 'start': []}
    for index, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx
        )
        q['x'].append(batch_x)
        q['y'].append(batch_y)
        q['start'].append(batch_start_idx)
    query_x = torch.cat(q['x'])
    query_y = torch.cat(q['y'])
    query_start = torch.cat(q['start'])
    query_base = base_forecast(model, query_x)
    return {
        'query_x': query_x, 'query_y': query_y, 'query_start': query_start,
        'query_base': query_base, 'query_residual': query_y - query_base,
        'memory_x': memory_x, 'memory_y': memory_y,
        'memory_residual': memory_residual,
    }


def _pair_mse(a, b):
    """[B, T] vs [N, T] -> [B, N] mean squared difference."""
    return (
        a.square().mean(-1, keepdim=True)
        + b.square().mean(-1).unsqueeze(0)
        - 2.0 * torch.matmul(a, b.transpose(0, 1)) / a.size(-1)
    ).clamp_min(0.0)


@torch.no_grad()
def oracle_predictions(experiment, data, top_k, tau=0.1, chunk=256):
    """Future-oracle and residual-oracle corrections for every query."""
    device = experiment.device
    channels = data['query_y'].size(-1)
    pred_len = data['query_y'].size(1)
    n_query = data['query_y'].size(0)

    future_pred = torch.zeros(n_query, pred_len, channels, device=device)
    residual_uniform = torch.zeros_like(future_pred)
    residual_weighted = torch.zeros_like(future_pred)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        starts = data['query_start'][start:stop]
        cand_mask, _ = experiment._candidate_mask(starts)
        cand_mask = cand_mask.to(device)

        for c in range(channels):
            q_future = data['query_y'][start:stop, :, c]
            k_future = data['memory_y'][:, :, c]
            q_last = data['query_x'][start:stop, -1, c]
            k_last = data['memory_x'][:, -1, c]
            # delta_last frame: compare and transplant futures relative to each
            # window's own last observed value, matching Stage-2's value space.
            q_delta = q_future - q_last.unsqueeze(1)
            k_delta = k_future - k_last.unsqueeze(1)

            future_mse = _pair_mse(q_delta, k_delta).masked_fill(~cand_mask, float('inf'))
            idx = future_mse.topk(min(top_k, k_delta.size(0)), dim=-1, largest=False).indices
            future_pred[start:stop, :, c] = (
                k_delta[idx].mean(dim=1) + q_last.unsqueeze(1)
            )

            q_res = data['query_residual'][start:stop, :, c]
            k_res = data['memory_residual'][:, :, c]
            res_mse = _pair_mse(q_res, k_res).masked_fill(~cand_mask, float('inf'))
            res_idx = res_mse.topk(min(top_k, k_res.size(0)), dim=-1, largest=False).indices
            selected = k_res[res_idx]                       # [b, K, T]
            residual_uniform[start:stop, :, c] = selected.mean(dim=1)
            # Same softmax weighting Stage-2 applies to its retrieved values.
            weights = torch.softmax(
                -res_mse.gather(1, res_idx) / tau, dim=-1
            ).unsqueeze(-1)
            residual_weighted[start:stop, :, c] = (selected * weights).sum(dim=1)

    return future_pred, residual_uniform, residual_weighted


def classify(row, tol=0.002):
    base = row['base_mse']
    residual = row['residual_oracle_best_mse']
    future = row['future_oracle_mse']
    if residual < future - tol and residual < base - tol:
        return 'RESIDUAL_TARGET_PROMISING'
    if residual < base - tol and abs(residual - future) <= tol:
        return 'RESIDUAL_TARGET_COMPARABLE'
    if residual >= base - tol:
        return 'RESIDUAL_TARGET_NOT_USEFUL'
    return 'RESIDUAL_TARGET_COMPARABLE'


@torch.no_grad()
def analyse(checkpoint_path, top_k=10, max_batches=0):
    experiment, args = load_stage2(checkpoint_path)

    val = prepare(experiment, 'val', max_batches)
    test = prepare(experiment, 'test', max_batches)
    _, val_res_uniform, _ = oracle_predictions(experiment, val, top_k)
    future_pred, res_uniform, res_weighted = oracle_predictions(experiment, test, top_k)

    base_mse, base_mae = mse_mae(test['query_base'], test['query_y'])
    future_mse_, future_mae_ = mse_mae(future_pred, test['query_y'])
    uni_mse, uni_mae = mse_mae(test['query_base'] + res_uniform, test['query_y'])
    wgt_mse, wgt_mae = mse_mae(test['query_base'] + res_weighted, test['query_y'])

    best_alpha, best_val = 0.0, float('inf')
    for alpha in ALPHA_GRID:
        candidate = mse_mae(val['query_base'] + alpha * val_res_uniform, val['query_y'])[0]
        if candidate < best_val:
            best_alpha, best_val = float(alpha), candidate
    best_mse, best_mae = mse_mae(
        test['query_base'] + best_alpha * res_uniform, test['query_y']
    )

    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len), 'top_k': top_k,
        'base_mse': base_mse, 'base_mae': base_mae,
        'future_oracle_mse': future_mse_, 'future_oracle_mae': future_mae_,
        'residual_oracle_uniform_alpha1_mse': uni_mse,
        'residual_oracle_uniform_alpha1_mae': uni_mae,
        'residual_oracle_weighted_alpha1_mse': wgt_mse,
        'residual_oracle_weighted_alpha1_mae': wgt_mae,
        'residual_oracle_best_alpha': best_alpha,
        'residual_oracle_best_mse': best_mse, 'residual_oracle_best_mae': best_mae,
        'future_oracle_gain': base_mse - future_mse_,
        'residual_oracle_gain': base_mse - best_mse,
        'checkpoint': checkpoint_path,
    }
    row['verdict'] = classify(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = analyse(args.checkpoint, args.top_k, args.max_batches)
    print(f"=== {row['dataset']}/{row['pred_len']} residual oracle ===")
    for key in COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_row(args.csv, row, COLUMNS)
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
