#!/usr/bin/env python3
"""STEP 4 -- predict the query's own error, then match it against history.

At alpha=1 the utility of a candidate is
    U(q,k) = 2<R_q,R_k>/T - ||R_k||^2/T
Everything on the right is memory-side except R_q. So instead of learning a
similarity, learn the one missing quantity and compute utility directly:

    Rhat_q = h(x_q)                      past only, trained against R_q
    Uhat(q,k) = 2<Rhat_q,R_k>/T - ||R_k||^2/T
    pick argmax_k Uhat, correct with that candidate's real residual

The decisive control is in the same script: `base + Rhat_q` uses the prediction
as the correction itself. If that wins, retrieval is not earning its place. If
selection wins, the predictor is better understood as a query representation for
matching than as an error forecast.
"""

import argparse, sys
from pathlib import Path
import torch, torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap
from utils.utility_selection import (
    build_selection_cache, forecast_from_selection, selection_metrics,
    utility_from_residuals,
)
from scripts.analyze_residual_oracle import prepare

COLUMNS = [
    'dataset', 'pred_len', 'arm', 'pool_m', 'top_r', 'split',
    'positive_at_1', 'selected_utility_at_1', 'oracle_pool_utility',
    'random_utility', 'utility_regret_at_1', 'selection_recovery_at_1',
    'top1_identity_accuracy', 'residual_pred_mse', 'residual_pred_corr',
    'forecast_mse', 'forecast_mae', 'direct_correction_mse',
    'direct_correction_mae', 'base_mse', 'checkpoint',
]


class ResidualPredictor(nn.Module):
    """Past window -> predicted base-forecast error, shared across channels."""

    def __init__(self, seq_len, pred_len, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, pred_len),
        )

    def forward(self, x):
        # x [B, L, C] -> per channel through a shared net -> [B, T, C]
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--top_r', type=int, default=1)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--csv', default='')
    p.add_argument('--save', default='')
    a = p.parse_args()

    experiment, saved = load_stage2(a.checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory(); experiment._build_key_bank(force=True)
    device = experiment.device

    splits, caches = {}, {}
    for split in ('train', 'val', 'test'):
        splits[split] = prepare(experiment, split, a.max_batches)
        caches[split] = build_selection_cache(
            experiment, model, splits[split], a.pool_m, a.alpha)

    train = splits['train']
    predictor = ResidualPredictor(
        train['query_x'].size(1), train['query_y'].size(1)).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=a.lr)
    n = train['query_x'].size(0)
    for epoch in range(a.epochs):
        predictor.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, a.batch):
            rows = order[start:start + a.batch]
            optimizer.zero_grad()
            # Target is the query residual; it is a label, never an input.
            loss = nn.functional.mse_loss(
                predictor(train['query_x'][rows]), train['query_residual'][rows])
            loss.backward(); optimizer.step()
            total += float(loss) * len(rows)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  epoch {epoch + 1}/{a.epochs} loss={total / n:.5f}')

    predictor.eval()
    for split in ('train', 'val', 'test'):
        data, cache = splits[split], caches[split]
        with torch.no_grad():
            predicted = predictor(data['query_x'])
        residual_mse, _ = mse_mae(predicted, data['query_residual'])
        flat_p = predicted.reshape(-1).double()
        flat_t = data['query_residual'].reshape(-1).double()
        corr = float(
            ((flat_p - flat_p.mean()) * (flat_t - flat_t.mean())).sum()
            / ((flat_p - flat_p.mean()).norm() * (flat_t - flat_t.mean()).norm() + 1e-12)
        )

        scores = torch.zeros_like(cache['utility'])
        with torch.no_grad():
            for slot_i, c in enumerate(cache['targets']):
                k_res = data['memory_residual'][:, :, c]
                predicted_u = utility_from_residuals(
                    predicted[:, :, c], k_res, a.alpha)
                scores[:, slot_i] = predicted_u.gather(
                    1, cache['pool_idx'][:, slot_i].to(device)).cpu()

        row = selection_metrics(cache, scores, a.top_r)
        pred = forecast_from_selection(experiment, data, cache, scores, a.top_r)
        mse, mae = mse_mae(pred, data['query_y'])
        direct_mse, direct_mae = mse_mae(
            data['query_base'] + a.alpha * predicted, data['query_y'])
        base_mse, _ = mse_mae(data['query_base'], data['query_y'])
        row.update({
            'dataset': saved.data, 'pred_len': int(saved.pred_len),
            'arm': 'predicted_residual_selector', 'pool_m': a.pool_m,
            'top_r': a.top_r, 'split': split,
            'residual_pred_mse': residual_mse, 'residual_pred_corr': corr,
            'forecast_mse': mse, 'forecast_mae': mae,
            'direct_correction_mse': direct_mse, 'direct_correction_mae': direct_mae,
            'base_mse': base_mse, 'checkpoint': a.checkpoint,
        })
        print(f"[{split}] select MSE={mse:.4f}  direct MSE={direct_mse:.4f}  "
              f"base={base_mse:.4f}  Rhat corr={corr:.3f}  "
              f"recovery={row['selection_recovery_at_1']:.3f}")
        if a.csv:
            append_row(a.csv, row, COLUMNS)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': predictor.state_dict(),
                    'seq_len': train['query_x'].size(1),
                    'pred_len': train['query_y'].size(1),
                    'pool_m': a.pool_m, 'alpha': a.alpha}, a.save)


if __name__ == '__main__':
    main()
