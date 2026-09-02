#!/usr/bin/env python3
"""Arms C-F -- pick one historical correction, with or without source context.

    s(q,k) = MLP([z_q^ctx, z_k, |z_q^ctx - z_k|, z_q^ctx * z_k] (+ f(R_k)))
    k*     = argmax_k s(q,k)
    Yhat   = Yhat_base + alpha * R_{k*}

    C  query target-only,     candidate target-only
    D  query cross-channel,   candidate target-only          <- the core arm
    E  query cross-channel,   candidate cross-channel
    F  query cross-channel,   candidate target-only + R_k feature

The candidate pool is the frozen Stage-2 retriever's Top-M and is identical for
every arm, so a better number is a better *choice* inside the pool rather than
an easier pool. Query futures build the utility target only; candidate futures
and residuals are memory-side and observable at inference.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_channel_context import ContextEncoder, count_parameters  # noqa: E402
from models.utility_pair_scorer import UtilityPairScorer  # noqa: E402
from utils.cross_channel_setup import (  # noqa: E402
    encode_candidates, load_context, write_attention_rows, write_source_rows,
)
from utils.retrieval_diagnostics import append_row, mse_mae  # noqa: E402
from utils.utility_selection import (  # noqa: E402
    EPS, forecast_from_selection, masked_utility, selection_metrics,
)

COLUMNS = [
    'dataset', 'pred_len', 'arm', 'cross_channel', 'candidate_cross_channel',
    'candidate_residual', 'topk', 'loss', 'pool_m', 'top_r', 'split',
    'positive_at_1', 'selected_utility_at_1', 'selected_best_utility_at_1',
    'oracle_pool_utility', 'random_utility', 'utility_regret_at_1',
    'selection_recovery_at_1', 'top1_identity_accuracy',
    'forecast_mse', 'forecast_mae', 'base_mse', 'gamma',
    'total_params', 'trainable_params', 'epochs', 'best_epoch', 'checkpoint',
]

ARM_NAMES = {
    (0, 0, 0): 'target_only_ressel',
    (1, 0, 0): 'query_cross_channel_ressel',
    (1, 1, 0): 'query_candidate_cross_channel_ressel',
    (1, 0, 1): 'query_cross_channel_residual_aware_ressel',
    (0, 0, 1): 'target_only_residual_aware_ressel',
}


class CrossChannelSelector(nn.Module):
    def __init__(self, seq_len, source_index, horizon=0, d_model=128, d_ff=256,
                 hidden=256, dropout=0.1, use_cross_channel_context=True,
                 candidate_cross_channel=False, scale_init=1e-2):
        super().__init__()
        self.context = ContextEncoder(
            seq_len, source_index, d_model, d_ff, dropout,
            use_cross_channel_context, scale_init,
        )
        self.scorer = UtilityPairScorer(d_model, horizon, hidden=hidden,
                                        dropout=dropout)
        self.candidate_cross_channel = bool(candidate_cross_channel)
        if candidate_cross_channel and not use_cross_channel_context:
            raise ValueError(
                'candidate_cross_channel needs the mixer; it is the same module'
            )

    def query_embedding(self, x, channel, z_channels=None):
        return self.context(x, channel, z_channels=z_channels)

    def candidate_embedding(self, memory_x, index, channel):
        return encode_candidates(
            self.context, memory_x, index, channel,
            candidate_context=self.candidate_cross_channel,
        )

    def forward(self, x, memory_x, index, channel, residual=None):
        z_q = self.query_embedding(x, channel).unsqueeze(1)
        z_k = self.candidate_embedding(memory_x, index, channel)
        return self.scorer(z_q, z_k, residual)


def pool_scores(model, data, cache, device, chunk=128, use_residual=False):
    out = torch.zeros_like(cache['utility'])
    memory_x = data['memory_x']
    with torch.no_grad():
        for slot_i, channel in enumerate(cache['targets']):
            k_res = data['memory_residual'][:, :, channel]
            for start in range(0, out.size(0), chunk):
                stop = min(start + chunk, out.size(0))
                index = cache['pool_idx'][start:stop, slot_i].to(device)
                residual = k_res[index] if use_residual else None
                out[start:stop, slot_i] = model(
                    data['query_x'][start:stop], memory_x, index, channel,
                    residual,
                ).float().cpu()
    return out


def forecast_mse_for(model, experiment, data, cache, device, top_r,
                     use_residual):
    scores = pool_scores(model, data, cache, device, use_residual=use_residual)
    prediction = forecast_from_selection(experiment, data, cache, scores, top_r)
    return scores, mse_mae(prediction, data['query_y'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--use_cross_channel_context', type=int, default=0)
    p.add_argument('--candidate_cross_channel_context', type=int, default=0)
    p.add_argument('--use_candidate_residual_feature', type=int, default=0)
    p.add_argument('--cross_channel_topk', type=int, default=5)
    p.add_argument('--cross_channel_source_mode', default='pearson_topk')
    p.add_argument('--cross_channel_scale_init', type=float, default=1e-2)
    p.add_argument('--utility_selection_loss', default='top1_ce',
                   choices=['top1_ce', 'soft_utility_kl'])
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--top_r', type=int, default=1)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--metrics_root', default='./metrics')
    p.add_argument('--csv', default='')
    p.add_argument('--attention_csv', default='')
    p.add_argument('--source_csv', default='')
    p.add_argument('--save', default='')
    a = p.parse_args()

    torch.manual_seed(a.seed)
    context = load_context(
        a.checkpoint, a.cross_channel_topk, a.cross_channel_source_mode,
        a.max_batches, a.metrics_root, need_pool=True, pool_m=a.pool_m,
        alpha=a.alpha,
    )
    experiment, saved = context['experiment'], context['saved']
    data, caches = context['data'], context['caches']
    device = experiment.device
    train, train_cache = data['train'], caches['train']

    horizon = train['memory_residual'].size(1) if a.use_candidate_residual_feature else 0
    model = CrossChannelSelector(
        seq_len=train['query_x'].size(1),
        source_index=context['source_index'],
        horizon=horizon, d_model=a.d_model, d_ff=a.d_ff, hidden=a.hidden,
        dropout=a.dropout,
        use_cross_channel_context=bool(a.use_cross_channel_context),
        candidate_cross_channel=bool(a.candidate_cross_channel_context),
        scale_init=a.cross_channel_scale_init,
    ).to(device)
    key = (int(a.use_cross_channel_context),
           int(a.candidate_cross_channel_context),
           int(a.use_candidate_residual_feature))
    arm = ARM_NAMES.get(key, 'cross_channel_ressel_' + ''.join(map(str, key)))
    total_params, trainable_params = count_parameters(model)
    print(f'{arm}: params={total_params} trainable={trainable_params} '
          f'pool={a.pool_m} loss={a.utility_selection_loss}')

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    utility = masked_utility(train_cache)
    n_query = utility.size(0)
    use_residual = horizon > 0
    best = {'val': float('inf'), 'epoch': -1, 'state': None}

    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(n_query)
        total, seen = 0.0, 0
        for start in range(0, n_query, a.batch):
            rows = order[start:start + a.batch]
            optimizer.zero_grad()
            loss = 0.0
            for slot_i, channel in enumerate(train_cache['targets']):
                index = train_cache['pool_idx'][rows, slot_i].to(device)
                residual = (
                    train['memory_residual'][:, :, channel][index]
                    if use_residual else None
                )
                scores = model(
                    train['query_x'][rows], train['memory_x'], index, channel,
                    residual,
                )
                u = utility[rows, slot_i].to(device)
                keep = torch.isfinite(u).all(-1)
                if not bool(keep.any()):
                    continue
                scores, u = scores[keep], u[keep]
                if a.utility_selection_loss == 'top1_ce':
                    loss = loss + nn.functional.cross_entropy(
                        scores / a.tau, u.argmax(-1)
                    )
                else:
                    teacher = torch.softmax(u / max(u.std().item(), EPS), -1)
                    loss = loss + nn.functional.kl_div(
                        torch.log_softmax(scores / a.tau, -1), teacher,
                        reduction='batchmean',
                    )
            if not torch.is_tensor(loss):
                continue
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
            seen += len(rows)
        model.eval()
        _, (val_mse, _) = forecast_mse_for(
            model, experiment, data['val'], caches['val'], device, a.top_r,
            use_residual,
        )
        if val_mse < best['val']:
            best = {
                'val': val_mse, 'epoch': epoch + 1,
                'state': {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        print(f'  epoch {epoch + 1}/{a.epochs} loss={total / max(seen, 1):.5f} '
              f'val_mse={val_mse:.5f} best@{best["epoch"]}')

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    gamma = (
        float(model.context.mixer.gamma.detach().mean())
        if model.context.mixer is not None else 0.0
    )

    for split in ('train', 'val', 'test'):
        cache = caches[split]
        scores = pool_scores(model, data[split], cache, device,
                             use_residual=use_residual)
        row = selection_metrics(cache, scores, a.top_r)
        prediction = forecast_from_selection(
            experiment, data[split], cache, scores, a.top_r
        )
        mse, mae = mse_mae(prediction, data[split]['query_y'])
        base_mse, _ = mse_mae(data[split]['query_base'], data[split]['query_y'])
        row.update({
            'dataset': saved.data, 'pred_len': int(saved.pred_len), 'arm': arm,
            'cross_channel': int(a.use_cross_channel_context),
            'candidate_cross_channel': int(a.candidate_cross_channel_context),
            'candidate_residual': int(a.use_candidate_residual_feature),
            'topk': int(a.cross_channel_topk) if a.use_cross_channel_context else 0,
            'loss': a.utility_selection_loss, 'pool_m': a.pool_m,
            'top_r': a.top_r, 'split': split, 'forecast_mse': mse,
            'forecast_mae': mae, 'base_mse': base_mse, 'gamma': gamma,
            'total_params': total_params, 'trainable_params': trainable_params,
            'epochs': a.epochs, 'best_epoch': best['epoch'],
            'checkpoint': a.checkpoint,
        })
        print(f"[{split}] MSE={mse:.4f} (base {base_mse:.4f})  "
              f"pos@1={row['positive_at_1']:.3f}  "
              f"regret={row['utility_regret_at_1']:.4f}  "
              f"recovery={row['selection_recovery_at_1']:.3f}")
        if a.csv:
            append_row(a.csv, row, COLUMNS)

    if a.source_csv:
        write_source_rows(context, saved.data, saved.pred_len,
                          a.cross_channel_topk, a.source_csv)
    if a.attention_csv:
        write_attention_rows(model.context, data['test']['query_x'], context,
                             saved.data, saved.pred_len, arm, a.attention_csv)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'source_index': context['source_index'],
            'correlations': context['correlations'],
            'channel_names': context['channel_names'],
            'args': vars(a),
        }, a.save)
        print(f'saved {a.save}')


if __name__ == '__main__':
    main()
