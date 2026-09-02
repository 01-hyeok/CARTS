#!/usr/bin/env python3
"""Arms A and B -- direct residual correction, without and with source context.

    Rhat_q = h_R(z_q^ctx)              z_q^ctx = z_E              (A, target-only)
                                       z_q^ctx = z_E + g*z_src    (B, cross-channel)
    Yhat   = Yhat_base + Rhat_q

No retrieval anywhere: this is the control the selection arms have to beat. A
and B differ by the mixer alone -- same encoder class, same width, same head --
so B < A means the source channels carry usable information about the target's
own error, independent of any retrieval question.

The query residual is the training target and never an input.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_channel_context import (  # noqa: E402
    ContextEncoder, ResidualHead, count_parameters,
)
from utils.cross_channel_setup import (  # noqa: E402
    load_context, residual_stats, write_attention_rows, write_source_rows,
)
from utils.retrieval_diagnostics import append_row, mse_mae  # noqa: E402

COLUMNS = [
    'dataset', 'pred_len', 'arm', 'cross_channel', 'topk', 'split',
    'forecast_mse', 'forecast_mae', 'base_mse', 'base_mae',
    'residual_pred_mse', 'residual_pred_corr', 'residual_pred_cosine',
    'residual_norm_error', 'gamma', 'total_params', 'trainable_params',
    'epochs', 'best_epoch', 'checkpoint',
]


class CrossChannelResDirect(nn.Module):
    """Contextual embedding per target channel -> that channel's residual."""

    def __init__(self, seq_len, pred_len, source_index, d_model=128, d_ff=256,
                 hidden=256, dropout=0.1, use_cross_channel_context=True,
                 scale_init=1e-2):
        super().__init__()
        self.context = ContextEncoder(
            seq_len, source_index, d_model, d_ff, dropout,
            use_cross_channel_context, scale_init,
        )
        self.head = ResidualHead(d_model, pred_len, hidden, dropout)
        self.channels = int(self.context.source_index.size(0))

    def forward(self, x, return_attention=False):
        """x [B, L, C] -> Rhat [B, T, C]."""
        z_channels = self.context.encode_channels(x)
        outputs, attentions = [], []
        for channel in range(self.channels):
            out = self.context(
                x, channel, z_channels=z_channels, return_attention=return_attention
            )
            z_ctx, attention = out if return_attention else (out, None)
            outputs.append(self.head(z_ctx))
            attentions.append(attention)
        stacked = torch.stack(outputs, dim=-1)                   # [B, T, C]
        if return_attention:
            return stacked, attentions
        return stacked


def evaluate(model, data, alpha):
    with torch.no_grad():
        predicted = model(data['query_x'])
    forecast = data['query_base'] + alpha * predicted
    mse, mae = mse_mae(forecast, data['query_y'])
    base_mse, base_mae = mse_mae(data['query_base'], data['query_y'])
    row = residual_stats(predicted, data['query_residual'])
    row.update({
        'forecast_mse': mse, 'forecast_mae': mae,
        'base_mse': base_mse, 'base_mae': base_mae,
    })
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--use_cross_channel_context', type=int, default=0)
    p.add_argument('--cross_channel_topk', type=int, default=5)
    p.add_argument('--cross_channel_source_mode', default='pearson_topk')
    p.add_argument('--cross_channel_scale_init', type=float, default=1e-2)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=30)
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
        a.max_batches, a.metrics_root, need_pool=False,
    )
    saved, data = context['saved'], context['data']
    device = context['experiment'].device
    train = data['train']

    model = CrossChannelResDirect(
        seq_len=train['query_x'].size(1),
        pred_len=train['query_y'].size(1),
        source_index=context['source_index'],
        d_model=a.d_model, d_ff=a.d_ff, hidden=a.hidden, dropout=a.dropout,
        use_cross_channel_context=bool(a.use_cross_channel_context),
        scale_init=a.cross_channel_scale_init,
    ).to(device)
    arm = 'cross_channel_resdirect' if a.use_cross_channel_context else 'target_only_resdirect'
    total_params, trainable_params = count_parameters(model)
    print(f'{arm}: params={total_params} trainable={trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = train['query_x'].size(0)
    best = {'val': float('inf'), 'epoch': -1, 'state': None}
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, a.batch):
            rows = order[start:start + a.batch]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(
                model(train['query_x'][rows]), train['query_residual'][rows]
            )
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
        model.eval()
        # Selection on val, identically for every arm, so that no arm is quietly
        # allowed to peek at its own test curve.
        val_mse = evaluate(model, data['val'], a.alpha)['forecast_mse']
        if val_mse < best['val']:
            best = {
                'val': val_mse, 'epoch': epoch + 1,
                'state': {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  epoch {epoch + 1}/{a.epochs} train={total / n:.5f} '
                  f'val_mse={val_mse:.5f} best@{best["epoch"]}')

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()

    gamma = (
        float(model.context.mixer.gamma.detach().mean())
        if model.context.mixer is not None else 0.0
    )
    for split in ('train', 'val', 'test'):
        row = evaluate(model, data[split], a.alpha)
        row.update({
            'dataset': saved.data, 'pred_len': int(saved.pred_len), 'arm': arm,
            'cross_channel': int(a.use_cross_channel_context),
            'topk': int(a.cross_channel_topk) if a.use_cross_channel_context else 0,
            'split': split, 'gamma': gamma, 'total_params': total_params,
            'trainable_params': trainable_params, 'epochs': a.epochs,
            'best_epoch': best['epoch'], 'checkpoint': a.checkpoint,
        })
        print(f"[{split}] MSE={row['forecast_mse']:.4f} (base {row['base_mse']:.4f})  "
              f"Rhat mse={row['residual_pred_mse']:.4f} corr={row['residual_pred_corr']:.3f}")
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
