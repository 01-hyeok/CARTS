#!/usr/bin/env python3
"""STEP 2/3 -- rank candidates by forecast utility instead of similarity.

The previous diagnosis ruled out every variant of "make the global embedding
rank the Oracle Top-10 better". This does something different: it scores a
*pair* inside an already-retrieved pool, so no single embedding space has to
order all 8449 candidates for every query at once.

    s(q,k) = MLP([z_q, z_k, |z_q-z_k|, z_q*z_k])          past_pair
           = MLP([z_q, z_k, f(R_k), |z_q-z_k|, z_q*z_k])  + candidate residual

Loss arms are compared, never summed:
    ce   pick the pool's best candidate            (selection, not ranking)
    kl   match softmax(U/tau) over the pool        (full relative ordering)
    reg  regress U directly                        (optional)

R_k is memory-side and observable at inference. The query residual is not, and
never enters the model -- it only builds the utility target.
"""

import argparse, sys
from pathlib import Path
import torch, torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap
from utils.utility_selection import (
    EPS, build_selection_cache, forecast_from_selection, masked_utility,
    selection_metrics,
)
from scripts.analyze_residual_oracle import prepare

COLUMNS = [
    'dataset', 'pred_len', 'arm', 'loss', 'features', 'pool_m', 'top_r', 'split',
    'positive_at_1', 'selected_utility_at_1', 'selected_best_utility_at_1',
    'oracle_pool_utility', 'random_utility', 'utility_regret_at_1',
    'selection_recovery_at_1', 'top1_identity_accuracy',
    'forecast_mse', 'forecast_mae', 'base_mse', 'checkpoint',
]


class PairScorer(nn.Module):
    """Scores one (query, candidate) pair; no global metric is implied."""

    def __init__(self, dim, horizon=0, residual_dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.residual_proj = (
            nn.Sequential(nn.Linear(horizon, residual_dim), nn.GELU())
            if horizon else None
        )
        width = 4 * dim + (residual_dim if horizon else 0)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_q, z_k, residual=None):
        z_q = z_q.expand_as(z_k)
        parts = [z_q, z_k, (z_q - z_k).abs(), z_q * z_k]
        if self.residual_proj is not None:
            parts.append(self.residual_proj(residual))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


def _bank_for(experiment, model, channel):
    sources = model.source_channels(channel)
    return experiment.key_bank[channel, sources.index(channel)]


@torch.no_grad()
def score_all(scorer, experiment, model, data, cache, device, chunk=128):
    """Scores for every pooled candidate, shaped like cache['utility']."""
    out = torch.zeros_like(cache['utility'])
    for slot_i, c in enumerate(cache['targets']):
        bank = _bank_for(experiment, model, c).to(device).float()
        k_res = data['memory_residual'][:, :, c]
        for start in range(0, out.size(0), chunk):
            stop = min(start + chunk, out.size(0))
            idx = cache['pool_idx'][start:stop, slot_i].to(device)
            z_q = cache['z_query'][start:stop, slot_i].unsqueeze(1).to(device)
            residual = k_res[idx] if scorer.residual_proj is not None else None
            out[start:stop, slot_i] = scorer(z_q, bank[idx], residual).cpu()
    return out


def train(scorer, experiment, model, data, cache, device, loss_mode,
          epochs, lr, batch, tau=1.0):
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    utility = masked_utility(cache)
    n_query = utility.size(0)
    for epoch in range(epochs):
        scorer.train()
        order = torch.randperm(n_query)
        total, seen = 0.0, 0
        for start in range(0, n_query, batch):
            rows = order[start:start + batch]
            optimizer.zero_grad()
            loss = 0.0
            for slot_i, c in enumerate(cache['targets']):
                bank = _bank_for(experiment, model, c).to(device).float()
                idx = cache['pool_idx'][rows, slot_i].to(device)
                z_q = cache['z_query'][rows, slot_i].unsqueeze(1).to(device)
                residual = (
                    data['memory_residual'][:, :, c][idx]
                    if scorer.residual_proj is not None else None
                )
                scores = scorer(z_q, bank[idx], residual)
                u = utility[rows, slot_i].to(device)
                keep = torch.isfinite(u).all(-1)
                if not bool(keep.any()):
                    continue
                scores, u = scores[keep], u[keep]
                if loss_mode == 'ce':
                    loss = loss + nn.functional.cross_entropy(
                        scores / tau, u.argmax(-1))
                elif loss_mode == 'kl':
                    teacher = torch.softmax(u / max(u.std().item(), EPS), -1)
                    loss = loss + nn.functional.kl_div(
                        torch.log_softmax(scores / tau, -1), teacher,
                        reduction='batchmean')
                elif loss_mode == 'reg':
                    loss = loss + nn.functional.mse_loss(scores, u)
                else:
                    raise ValueError(f'unknown loss: {loss_mode}')
            if not torch.is_tensor(loss):
                continue
            loss.backward(); optimizer.step()
            total += float(loss) * len(rows); seen += len(rows)
        print(f'  epoch {epoch + 1}/{epochs} loss={total / max(seen, 1):.5f}')
    return scorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--loss', default='ce', choices=['ce', 'kl', 'reg'])
    p.add_argument('--use_candidate_residual', type=int, default=0)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--eval_pool_m', type=int, default=0)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--top_r', type=int, default=1)
    p.add_argument('--epochs', type=int, default=15)
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
        print(f'  {split}: queries={caches[split]["utility"].size(0)} '
              f'pool={a.pool_m}')

    dim = caches['train']['z_query'].size(-1)
    horizon = splits['train']['memory_residual'].size(1) if a.use_candidate_residual else 0
    scorer = PairScorer(dim, horizon).to(device)
    features = 'past_pair_plus_candidate_residual' if horizon else 'past_pair_only'
    arm = f"{'residual_aware' if horizon else 'utility'}_ranker_{a.loss}"

    train(scorer, experiment, model, splits['train'], caches['train'], device,
          a.loss, a.epochs, a.lr, a.batch)
    scorer.eval()

    for split in ('train', 'val', 'test'):
        cache = caches[split]
        scores = score_all(scorer, experiment, model, splits[split], cache, device)
        row = selection_metrics(cache, scores, a.top_r)
        pred = forecast_from_selection(
            experiment, splits[split], cache, scores, a.top_r)
        mse, mae = mse_mae(pred, splits[split]['query_y'])
        base_mse, _ = mse_mae(splits[split]['query_base'], splits[split]['query_y'])
        row.update({
            'dataset': saved.data, 'pred_len': int(saved.pred_len),
            'arm': arm, 'loss': a.loss, 'features': features,
            'pool_m': a.pool_m, 'top_r': a.top_r, 'split': split,
            'forecast_mse': mse, 'forecast_mae': mae, 'base_mse': base_mse,
            'checkpoint': a.checkpoint,
        })
        print(f"[{split}] MSE={mse:.4f} (base {base_mse:.4f})  "
              f"pos@1={row['positive_at_1']:.3f}  "
              f"recovery={row['selection_recovery_at_1']:.3f}")
        if a.csv:
            append_row(a.csv, row, COLUMNS)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': scorer.state_dict(), 'dim': dim,
                    'horizon': horizon, 'pool_m': a.pool_m, 'alpha': a.alpha,
                    'loss': a.loss}, a.save)
        print(f'saved {a.save}')


if __name__ == '__main__':
    main()
