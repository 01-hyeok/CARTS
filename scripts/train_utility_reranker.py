#!/usr/bin/env python3
"""PHASE 2-5 -- train a shortlist reranker and put it on the production path.

Arms, all on the identical frozen shortlist:

    original          the retriever's own order                (no training)
    past_pair         pair MLP over frozen Stage-1 embeddings
    residual_aware    + the candidate's historical residual
    oracle            ordered by measured utility              (upper bound)

Targets: `regression` (fit U directly) and `listwise_kl` (match softmax(U/tau)
over the shortlist). One at a time, never summed.

Two evaluations, kept apart:
  Phase 4  does the score rank utility better than the retriever's score
  Phase 5  does the resulting Top-K, put back through the *unmodified*
           production Stage-2, forecast better

Phase 5 restricts the memory mask to the chosen ids and runs the real forward.
That reproduces production output bit-for-bit when the chosen ids are the
retriever's own Top-K, which is what makes the intervention legitimate.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utility_reranker import build_reranker  # noqa: E402
from scripts.analyze_oracle_rerank_headroom import (  # noqa: E402
    mse_mae_per_channel, restricted_mask,
)
from scripts.analyze_set_level_utility import selected_candidates  # noqa: E402
from utils.retrieval_diagnostics import append_row, rank_correlations, unwrap  # noqa: E402
from utils.utility_teacher import load_stage2_reference  # noqa: E402

NEG_INF = -1e9

RERANK_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'top_k', 'arm', 'target', 'split',
    'queries', 'spearman', 'pearson', 'ndcg_at_10',
    'utility_at_1', 'utility_at_5', 'utility_at_10',
    'positive_rate_at_10', 'random_utility_at_10', 'oracle_utility_at_10',
    'gap_recovery_at_10', 'params', 'best_epoch', 'checkpoint',
]
FORECAST_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'top_k', 'arm', 'target', 'split',
    'queries', 'forecast_mse', 'forecast_mae', 'base_mse',
    'original_mse', 'oracle_mse', 'available_gain', 'recovered_gain',
    'checkpoint',
]


def load_cache(path):
    cache = torch.load(path, map_location='cpu')
    cache['ids'] = cache['ids'].long()
    for key in ('scores', 'utility', 'z_query', 'base_mse'):
        cache[key] = cache[key].float()
    return cache


def candidate_embeddings(model, key_bank, ids, device):
    """Frozen Stage-1 embeddings of the shortlisted ids, [Q, C, M, d]."""
    out = []
    for slot_i, c in enumerate(model.target_channels()):
        slot = model.source_channels(c).index(c)
        bank = model._branch_memory(key_bank, c, slot, c, torch.float32, device)
        out.append(bank.index_select(0, ids[:, slot_i].reshape(-1).to(device))
                   .view(ids.size(0), ids.size(2), -1))
    return torch.stack(out, dim=1)


def batched_residual(memory_residual, ids, channels, device, chunk=4096):
    """[Q, C, M] ids -> [Q*C, M, H] residuals without a Python loop per row."""
    queries, _, width = ids.shape
    rows = queries * channels
    horizon = memory_residual.size(1)
    out = torch.empty(rows, width, horizon)
    flat_ids = ids.reshape(rows, width)
    channel_of = torch.arange(channels).repeat(queries)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = flat_ids[start:stop]
        picked = memory_residual.index_select(0, block.reshape(-1))       # [n*M, H, C]
        picked = picked.view(stop - start, width, horizon, -1)
        index = channel_of[start:stop].view(-1, 1, 1, 1).expand(-1, width, horizon, 1)
        out[start:stop] = picked.gather(3, index).squeeze(-1)
    return out.to(device)


def score_all(model_r, data, chunk=2048):
    outputs = []
    with torch.no_grad():
        for start in range(0, data['z_q'].size(0), chunk):
            stop = start + chunk
            outputs.append(model_r(
                data['z_q'][start:stop], data['z_k'][start:stop],
                data['score'][start:stop],
                data['residual'][start:stop] if 'residual' in data else None,
            ))
    return torch.cat(outputs)


def ranking_metrics(score, utility, valid, top_k=10):
    """Phase 4 numbers, all against the same shortlist and validity mask."""
    score = score.masked_fill(~valid, NEG_INF)
    utility = utility.masked_fill(~valid, NEG_INF)
    width = int(valid.sum(-1).min().clamp_min(1))
    depth = min(top_k, width)
    by_score = score.topk(depth, dim=-1).indices
    by_utility = utility.topk(depth, dim=-1).indices

    relevance = utility.clamp_min(0.0).masked_fill(~valid, 0.0)
    discount = 1.0 / torch.log2(
        torch.arange(depth, device=score.device, dtype=torch.float64) + 2.0)
    dcg = (relevance.gather(1, by_score).double() * discount).sum(-1)
    idcg = (relevance.gather(1, by_utility).double() * discount).sum(-1)
    keep = idcg > 0

    picked = utility.gather(1, by_score)
    random_utility = (
        utility.masked_fill(~valid, 0.0).sum(-1) / valid.sum(-1).clamp_min(1))
    oracle_utility = utility.gather(1, by_utility).mean(-1)
    retrieved = picked.mean(-1)
    pearson, spearman = rank_correlations(score, utility, valid)
    return {
        'spearman': spearman, 'pearson': pearson,
        'ndcg_at_10': float((dcg[keep] / idcg[keep]).mean()) if bool(keep.any()) else float('nan'),
        'utility_at_1': float(picked[:, 0].mean()),
        'utility_at_5': float(picked[:, :min(5, depth)].mean()),
        'utility_at_10': float(retrieved.mean()),
        'positive_rate_at_10': float((picked > 0).float().mean()),
        'random_utility_at_10': float(random_utility.mean()),
        'oracle_utility_at_10': float(oracle_utility.mean()),
        'gap_recovery_at_10': float(
            ((retrieved - random_utility) / (oracle_utility - random_utility + 1e-8)).mean()),
    }


def train(model_r, data, target, epochs, lr, batch, tau_u, tau_r, normalize,
          val_data, top_k):
    optimizer = torch.optim.Adam(model_r.parameters(), lr=lr)
    rows = data['z_q'].size(0)
    utility = data['utility']
    if normalize == 'query_zscore':
        mean = utility.masked_fill(~data['valid'], 0.0).sum(-1, keepdim=True) / \
            data['valid'].sum(-1, keepdim=True).clamp_min(1)
        std = ((utility - mean) * data['valid']).square().sum(-1, keepdim=True)
        std = (std / data['valid'].sum(-1, keepdim=True).clamp_min(1)).sqrt().clamp_min(1e-6)
        target_utility = (utility - mean) / std
    else:
        target_utility = utility

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
                data['residual'][index] if 'residual' in data else None,
            )
            valid = data['valid'][index]
            if target == 'regression':
                loss = (((scores - target_utility[index]) ** 2) * valid).sum() / \
                    valid.sum().clamp_min(1)
            else:
                teacher = torch.softmax(
                    (utility[index] / tau_u).masked_fill(~valid, NEG_INF), dim=-1)
                student = torch.log_softmax(
                    (scores / tau_r).masked_fill(~valid, NEG_INF), dim=-1)
                loss = nn.functional.kl_div(student, teacher, reduction='batchmean')
            loss.backward()
            optimizer.step()
            total += float(loss) * index.numel()
        model_r.eval()
        # Selected on val utility gap recovery: the quantity the study is about,
        # and available without touching a forecast.
        metric = ranking_metrics(
            score_all(model_r, val_data), val_data['utility'], val_data['valid'], top_k
        )['gap_recovery_at_10']
        if metric > best['score']:
            best = {'score': metric, 'epoch': epoch + 1,
                    'state': {k: v.detach().clone() for k, v in model_r.state_dict().items()}}
        print(f'  epoch {epoch + 1}/{epochs} loss={total / rows:.5f} '
              f'val_gap_recovery={metric:.4f} best@{best["epoch"]}')
    if best['state'] is not None:
        model_r.load_state_dict(best['state'])
    model_r.eval()
    return best['epoch']


@torch.no_grad()
def production_forecast(experiment, model, cache, chosen, top_k, max_queries):
    """Phase 5: chosen ids -> restricted mask -> the real Stage-2 forward."""
    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    _, loader = experiment._get_data(flag=cache['split'], shuffle=False)
    predictions, targets, offset = [], [], 0
    for batch_x, batch_y, batch_start_idx in loader:
        if offset >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        keep = min(batch_x.size(0), max_queries - offset)
        batch_x, batch_y, batch_start_idx = batch_x[:keep], batch_y[:keep], batch_start_idx[:keep]
        if not torch.equal(batch_start_idx.cpu(), cache['query_start'][offset:offset + keep]):
            raise ValueError('cache and loader are out of order; the shortlist would '
                             'be applied to the wrong queries')
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        ids = chosen[offset:offset + keep].to(device)
        columns = []
        previous = model.top_k
        model.top_k = ids.size(-1)
        try:
            for c in range(ids.size(1)):
                mask = restricted_mask(cand_mask, ids[:, c])
                y = experiment.model(
                    batch_x=batch_x, memory_y=memory_y, valid_mask=mask,
                    key_bank=experiment.key_bank,
                    memory_x_last=experiment.memory_x_last,
                )[0]
                columns.append(y[:, :, c])
        finally:
            model.top_k = previous
        predictions.append(torch.stack(columns, dim=-1).cpu())
        targets.append(batch_y.cpu())
        offset += keep
    mse, mae = mse_mae_per_channel(torch.cat(predictions), torch.cat(targets))
    return float(mse.mean()), float(mae.mean())


def choose(score, valid, ids, top_k):
    """Top-K ids per (query, channel) under a score, [Q, C, K]."""
    queries, channels, width = ids.shape
    score = score.detach().cpu()
    flat = score.masked_fill(~valid.reshape(-1, width), NEG_INF)
    depth = min(top_k, width)
    picked = flat.topk(depth, dim=-1).indices.view(queries, channels, depth)
    return ids.gather(2, picked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--arm', default='past_pair', choices=['past_pair', 'residual_aware'])
    p.add_argument('--target', default='regression',
                   choices=['regression', 'listwise_kl'])
    p.add_argument('--normalize', default='raw', choices=['raw', 'query_zscore'])
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--tau_u', type=float, default=0.1)
    p.add_argument('--tau_r', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rerank_csv', default='')
    p.add_argument('--forecast_csv', default='')
    p.add_argument('--baselines', type=int, default=1,
                   help='also write the original and oracle rows')
    a = p.parse_args()

    torch.manual_seed(a.seed)
    experiment, args = load_stage2_reference(a.checkpoint)
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
    use_residual = a.arm == 'residual_aware'

    data = {}
    for split, cache in caches.items():
        embeddings = candidate_embeddings(model, experiment.key_bank, cache['ids'], device)
        block = {
            'z_q': cache['z_query'].reshape(-1, 1, cache['z_query'].size(-1)).to(device),
            'z_k': embeddings.reshape(-1, cache['ids'].size(2), embeddings.size(-1)).to(device),
            'score': cache['scores'].reshape(-1, cache['ids'].size(2)).to(device),
            'utility': cache['utility'].reshape(-1, cache['ids'].size(2)).to(device),
            'valid': cache['valid'].reshape(-1, cache['ids'].size(2)).to(device),
        }
        if use_residual:
            block['residual'] = batched_residual(
                memory_residual, cache['ids'], channels, device)
        data[split] = block

    dim = data['train']['z_q'].size(-1)
    reranker = build_reranker(a.arm, dim, horizon, dropout=a.dropout).to(device)
    params = sum(p_.numel() for p_ in reranker.parameters())
    print(f'{a.arm}/{a.target} M={a.pool_m} params={params}')
    best_epoch = train(reranker, data['train'], a.target, a.epochs, a.lr, a.batch,
                       a.tau_u, a.tau_r, a.normalize, data['val'], a.top_k)

    test_cache = caches['test']
    test = data['test']
    queries = test_cache['queries']
    base_mse = float(test_cache['base_mse'].mean())

    arms = {a.arm: score_all(reranker, test)}
    if a.baselines:
        arms['original'] = test['score']
        arms['oracle'] = test['utility']

    results = {}
    for name, score in arms.items():
        target = a.target if name == a.arm else 'none'
        metrics = ranking_metrics(score, test['utility'], test['valid'], a.top_k)
        chosen = choose(score, test_cache['valid'], test_cache['ids'], a.top_k)
        mse, mae = production_forecast(experiment, model, test_cache, chosen,
                                       a.top_k, queries)
        results[name] = (mse, mae)
        print(f"  [{name}/{target}] gap_recovery={metrics['gap_recovery_at_10']:.4f} "
              f"spearman={metrics['spearman']:.4f} MSE={mse:.4f}")
        if a.rerank_csv:
            append_row(a.rerank_csv, {
                'dataset': a.dataset, 'pred_len': a.pred_len, 'pool_m': a.pool_m,
                'top_k': a.top_k, 'arm': name, 'target': target, 'split': 'test',
                'queries': queries, 'params': params if name == a.arm else 0,
                'best_epoch': best_epoch if name == a.arm else 0,
                'checkpoint': a.checkpoint, **metrics,
            }, RERANK_COLUMNS)

    original_mse = results.get('original', (float('nan'),))[0]
    oracle_mse = results.get('oracle', (float('nan'),))[0]
    for name, (mse, mae) in results.items():
        available = original_mse - oracle_mse
        if a.forecast_csv:
            append_row(a.forecast_csv, {
                'dataset': a.dataset, 'pred_len': a.pred_len, 'pool_m': a.pool_m,
                'top_k': a.top_k, 'arm': name,
                'target': a.target if name == a.arm else 'none', 'split': 'test',
                'queries': queries, 'forecast_mse': mse, 'forecast_mae': mae,
                'base_mse': base_mse, 'original_mse': original_mse,
                'oracle_mse': oracle_mse, 'available_gain': available,
                'recovered_gain': (original_mse - mse) / (available + 1e-12),
                'checkpoint': a.checkpoint,
            }, FORECAST_COLUMNS)


if __name__ == '__main__':
    main()
