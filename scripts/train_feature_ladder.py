#!/usr/bin/env python3
"""OBSERVABILITY 1 -- the oracle feature ladder.

One shortlist, one backbone, one loss. The only thing that changes between arms
is what the reranker is allowed to see:

    A  past only                                        deployable
    B  + candidate historical residual                  deployable
    C  + predicted query residual (from query past)     deployable
    D  + TRUE query residual                            ORACLE DIAGNOSTIC
    E  + query future                                   ORACLE DIAGNOSTIC

The point is not to build D or E. It is to separate two explanations of the
learned reranker's failure: if D suddenly recovers the oracle headroom then the
information the selector needs is the query's own correction, which the past
does not reveal (H2). If D changes little, the pairwise formulation itself is
what is insufficient.

Arms A-C never touch a query future or a true residual. Arms D and E do, are
marked non-deployable in every row they write, and exist only to bound what is
knowable.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utility_reranker import (  # noqa: E402
    FEATURE_LADDER, LadderReranker, QueryResidualPredictor,
)
from scripts.train_utility_reranker import (  # noqa: E402
    NEG_INF, batched_residual, candidate_embeddings, choose, load_cache,
    production_forecast, ranking_metrics,
)
from utils.retrieval_diagnostics import append_row, unwrap  # noqa: E402
from utils.utility_teacher import load_stage2_reference  # noqa: E402

LADDER_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'top_k', 'arm', 'deployable', 'target',
    'split', 'queries', 'spearman', 'pearson', 'ndcg_at_10', 'ndcg_at_50',
    'utility_at_1', 'utility_at_5', 'utility_at_10', 'positive_rate_at_10',
    'random_utility_at_10', 'oracle_utility_at_10', 'gap_recovery_at_10',
    'residual_pred_mse', 'total_params', 'trainable_params', 'best_epoch',
    'checkpoint',
]
FORECAST_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'top_k', 'arm', 'deployable', 'target',
    'split', 'queries', 'forecast_mse', 'forecast_mae', 'set_utility',
    'base_mse', 'original_mse', 'oracle_mse', 'available_gain',
    'recovered_gain', 'checkpoint',
]


def group_inputs(cache, data, arm, predicted_residual, device):
    """The optional feature tensors this arm is allowed to consume."""
    uses, _ = FEATURE_LADDER[arm]
    use_candidate, use_predicted, use_true, use_future = uses
    channels = cache['ids'].size(1)
    out = {}
    if use_candidate:
        out['candidate_residual'] = data['residual']
    if use_predicted:
        out['predicted_query_residual'] = channel_flat(predicted_residual, channels, device)
    if use_true:
        out['true_query_residual'] = channel_flat(cache['query_residual'], channels, device)
    if use_future:
        out['query_future'] = channel_flat(cache['query_future'], channels, device)
    return out


def channel_flat(tensor, channels, device):
    """[Q, H, C] -> [Q*C, H], matching the (query, channel) row order."""
    return tensor.permute(0, 2, 1).reshape(-1, tensor.size(1)).to(device)


def score_all(model_r, data, extras, chunk=2048):
    outputs = []
    with torch.no_grad():
        for start in range(0, data['z_q'].size(0), chunk):
            stop = start + chunk
            outputs.append(model_r(
                data['z_q'][start:stop], data['z_k'][start:stop],
                data['score'][start:stop],
                **{k: v[start:stop] for k, v in extras.items()},
            ))
    return torch.cat(outputs)


def train_predictor(caches, device, epochs, lr, batch):
    """Arm C's residual predictor: train split only, then frozen."""
    train = caches['train']
    predictor = QueryResidualPredictor(
        train['query_x'].size(1), train['query_residual'].size(1)).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    x = train['query_x'].to(device)
    y = train['query_residual'].to(device)
    n = x.size(0)
    for epoch in range(epochs):
        predictor.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(predictor(x[index]), y[index])
            loss.backward()
            optimizer.step()
            total += float(loss) * index.numel()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  [predictor] epoch {epoch + 1}/{epochs} loss={total / n:.5f}')
    predictor.eval()
    out, errors = {}, {}
    with torch.no_grad():
        for split, cache in caches.items():
            predicted = predictor(cache['query_x'].to(device))
            out[split] = predicted.cpu()
            errors[split] = float(
                (predicted.cpu() - cache['query_residual']).square().mean())
    print(f"  [predictor] residual MSE test={errors['test']:.4f}")
    return out, errors


def train(model_r, data, extras, target, epochs, lr, batch, tau_u, tau_r,
          val_data, val_extras, top_k):
    optimizer = torch.optim.Adam(model_r.parameters(), lr=lr)
    rows = data['z_q'].size(0)
    utility, valid = data['utility'], data['valid']
    best = {'score': -float('inf'), 'epoch': -1, 'state': None}
    for epoch in range(epochs):
        model_r.train()
        order = torch.randperm(rows, device=utility.device)
        total = 0.0
        for start in range(0, rows, batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            scores = model_r(
                data['z_q'][index], data['z_k'][index], data['score'][index],
                **{k: v[index] for k, v in extras.items()},
            )
            keep = valid[index]
            if target == 'regression':
                loss = (((scores - utility[index]) ** 2) * keep).sum() / keep.sum().clamp_min(1)
            else:
                teacher = torch.softmax(
                    (utility[index] / tau_u).masked_fill(~keep, NEG_INF), dim=-1)
                loss = nn.functional.kl_div(
                    torch.log_softmax((scores / tau_r).masked_fill(~keep, NEG_INF), dim=-1),
                    teacher, reduction='batchmean')
            loss.backward()
            optimizer.step()
            total += float(loss) * index.numel()
        model_r.eval()
        metric = ranking_metrics(
            score_all(model_r, val_data, val_extras), val_data['utility'],
            val_data['valid'], top_k)['gap_recovery_at_10']
        if metric > best['score']:
            best = {'score': metric, 'epoch': epoch + 1,
                    'state': {k: v.detach().clone() for k, v in model_r.state_dict().items()}}
        print(f'  epoch {epoch + 1}/{epochs} loss={total / rows:.5f} '
              f'val_gap_recovery={metric:.4f} best@{best["epoch"]}')
    if best['state'] is not None:
        model_r.load_state_dict(best['state'])
    model_r.eval()
    return best['epoch']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--arm', required=True, choices=sorted(FEATURE_LADDER))
    p.add_argument('--target', default='regression',
                   choices=['regression', 'listwise_kl'])
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--predictor_epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--tau_u', type=float, default=0.1)
    p.add_argument('--tau_r', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ladder_csv', default='')
    p.add_argument('--forecast_csv', default='')
    p.add_argument('--baselines', type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    experiment, _ = load_stage2_reference(a.checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    device = experiment.device

    stem = f'{a.dataset}_{a.pred_len}_top{a.pool_m}'
    caches = {split: load_cache(Path(a.cache_dir) / f'{stem}_{split}.pt')
              for split in ('train', 'val', 'test')}
    memory_residual = caches['train']['memory_residual']
    horizon = memory_residual.size(1)
    channels = caches['train']['ids'].size(1)
    uses, deployable = FEATURE_LADDER[a.arm]
    needs_candidate_residual = uses[0]

    data = {}
    for split, cache in caches.items():
        embeddings = candidate_embeddings(model, experiment.key_bank, cache['ids'], device)
        width = cache['ids'].size(2)
        block = {
            'z_q': cache['z_query'].reshape(-1, 1, cache['z_query'].size(-1)).to(device),
            'z_k': embeddings.reshape(-1, width, embeddings.size(-1)).to(device),
            'score': cache['scores'].reshape(-1, width).to(device),
            'utility': cache['utility'].reshape(-1, width).to(device),
            'valid': cache['valid'].reshape(-1, width).to(device),
        }
        if needs_candidate_residual:
            block['residual'] = batched_residual(
                memory_residual, cache['ids'], channels, device)
        data[split] = block

    predicted, predictor_error = ({}, {'test': float('nan')})
    if uses[1]:
        predicted, predictor_error = train_predictor(
            caches, device, a.predictor_epochs, a.lr, 64)

    extras = {split: group_inputs(caches[split], data[split], a.arm,
                                  predicted.get(split), device)
              for split in caches}

    dim = data['train']['z_q'].size(-1)
    reranker = LadderReranker(
        dim, horizon, use_candidate_residual=uses[0],
        use_predicted_query_residual=uses[1], use_true_query_residual=uses[2],
        use_query_future=uses[3], dropout=a.dropout).to(device)
    total_params = sum(t.numel() for t in reranker.parameters())
    print(f'{a.arm} (deployable={deployable}) params={total_params}')
    best_epoch = train(reranker, data['train'], extras['train'], a.target,
                       a.epochs, a.lr, a.batch, a.tau_u, a.tau_r,
                       data['val'], extras['val'], a.top_k)

    cache, test = caches['test'], data['test']
    queries = cache['queries']
    base_mse = float(cache['base_mse'].mean())
    arms = {a.arm: score_all(reranker, test, extras['test'])}
    if a.baselines:
        arms['original'] = test['score']
        arms['oracle'] = test['utility']

    results = {}
    for name, score in arms.items():
        metrics = ranking_metrics(score, test['utility'], test['valid'], a.top_k)
        metrics['ndcg_at_50'] = ranking_metrics(
            score, test['utility'], test['valid'], 50)['ndcg_at_10']
        chosen = choose(score, cache['valid'], cache['ids'], a.top_k)
        mse, mae = production_forecast(experiment, model, cache, chosen, a.top_k, queries)
        results[name] = (mse, mae)
        is_arm = name == a.arm
        print(f"  [{name}] gap_recovery={metrics['gap_recovery_at_10']:.4f} "
              f"spearman={metrics['spearman']:.4f} MSE={mse:.4f}")
        if a.ladder_csv:
            append_row(a.ladder_csv, {
                'dataset': a.dataset, 'pred_len': a.pred_len, 'pool_m': a.pool_m,
                'top_k': a.top_k, 'arm': name,
                'deployable': int(deployable if is_arm else True),
                'target': a.target if is_arm else 'none', 'split': 'test',
                'queries': queries,
                'residual_pred_mse': predictor_error.get('test', float('nan')) if is_arm else float('nan'),
                'total_params': total_params if is_arm else 0,
                'trainable_params': total_params if is_arm else 0,
                'best_epoch': best_epoch if is_arm else 0,
                'checkpoint': a.checkpoint, **metrics,
            }, LADDER_COLUMNS)

    original_mse = results.get('original', (float('nan'),))[0]
    oracle_mse = results.get('oracle', (float('nan'),))[0]
    for name, (mse, mae) in results.items():
        is_arm = name == a.arm
        if a.forecast_csv:
            append_row(a.forecast_csv, {
                'dataset': a.dataset, 'pred_len': a.pred_len, 'pool_m': a.pool_m,
                'top_k': a.top_k, 'arm': name,
                'deployable': int(deployable if is_arm else True),
                'target': a.target if is_arm else 'none', 'split': 'test',
                'queries': queries, 'forecast_mse': mse, 'forecast_mae': mae,
                'set_utility': base_mse - mse, 'base_mse': base_mse,
                'original_mse': original_mse, 'oracle_mse': oracle_mse,
                'available_gain': original_mse - oracle_mse,
                'recovered_gain': (original_mse - mse) / (original_mse - oracle_mse + 1e-12),
                'checkpoint': a.checkpoint,
            }, FORECAST_COLUMNS)


if __name__ == '__main__':
    main()
