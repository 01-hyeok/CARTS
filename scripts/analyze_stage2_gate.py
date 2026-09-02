#!/usr/bin/env python3
"""STEP 1 -- does Stage-2 actually use retrieval?

Inference-time ablation inside ONE trained checkpoint, so nothing differs but
the retrieval contribution itself. Comparing against a separately trained
no-retrieval model would confound the ablation with a different optimum.

  learned gate      the checkpoint as-is
  gate = 0          y_base only, retrieval contribution removed
  best alpha        scalar mixing weight chosen on validation, applied to test
  shuffled          retrieval content permuted across queries; the gate, the
                    base forecast and the magnitude of y_ret all stay intact,
                    so only the query-candidate correspondence is destroyed

If shuffling costs nothing, the retrieval was not carrying query-specific
information regardless of what the gate says.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import (  # noqa: E402
    ALPHA_GRID, alpha_grid_search, append_row, collect_split, load_stage2, mse_mae,
)

COLUMNS = [
    'dataset', 'pred_len', 'learned_gate_mean', 'learned_gate_std',
    'learned_mse', 'learned_mae', 'gate0_mse', 'gate0_mae',
    'best_alpha', 'best_alpha_val_mse', 'best_alpha_test_mse', 'best_alpha_test_mae',
    'shuffled_mse', 'shuffled_mae',
    'delta_learned_vs_gate0', 'delta_learned_vs_shuffled', 'verdict', 'checkpoint',
]


def classify(row, tol=1e-4):
    """Label each setting from the four ablations."""
    learned, gate0 = row['learned_mse'], row['gate0_mse']
    shuffled, best_alpha = row['shuffled_mse'], row['best_alpha']
    gate_mean = row['learned_gate_mean']

    helps = learned < gate0 - tol
    shuffling_hurts = shuffled > learned + tol
    if helps and shuffling_hurts:
        return 'RETRIEVAL_USEFUL'
    # A positive best alpha that the learned gate never reached means the
    # information was there and the gate failed to use it.
    if best_alpha > 0.0 and gate_mean < 0.05 and row['best_alpha_val_mse'] < gate0 - tol:
        return 'GATE_OPTIMIZATION_ISSUE'
    if best_alpha == 0.0 and abs(learned - gate0) <= max(tol, 0.002 * abs(gate0)):
        return 'RETRIEVAL_NOT_USEFUL'
    return 'AMBIGUOUS'


@torch.no_grad()
def analyse(checkpoint_path, max_batches=0, seed=0):
    experiment, args = load_stage2(checkpoint_path)
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    if getattr(model, 'fusion_mode', 'residual') != 'residual':
        raise ValueError(f'expects residual fusion; got {model.fusion_mode}')

    val = collect_split(experiment, 'val', max_batches)
    test = collect_split(experiment, 'test', max_batches)

    learned_pred = test['base'] + _broadcast(test['lam'], test['ret']) * test['ret']
    learned_mse, learned_mae = mse_mae(learned_pred, test['true'])
    gate0_mse, gate0_mae = mse_mae(test['base'], test['true'])

    best_alpha, best_val_mse, _ = alpha_grid_search(
        val['base'], val['ret'], val['true'], ALPHA_GRID
    )
    alpha_pred = test['base'] + best_alpha * test['ret']
    alpha_mse, alpha_mae = mse_mae(alpha_pred, test['true'])

    # Destroy only the query-candidate correspondence.
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(test['ret'].size(0), generator=generator)
    shuffled_pred = (
        test['base'] + _broadcast(test['lam'], test['ret']) * test['ret'][permutation]
    )
    shuffled_mse, shuffled_mae = mse_mae(shuffled_pred, test['true'])

    row = {
        'dataset': args.data,
        'pred_len': int(args.pred_len),
        'learned_gate_mean': float(test['lam'].mean()),
        'learned_gate_std': float(test['lam'].std()),
        'learned_mse': learned_mse, 'learned_mae': learned_mae,
        'gate0_mse': gate0_mse, 'gate0_mae': gate0_mae,
        'best_alpha': best_alpha, 'best_alpha_val_mse': best_val_mse,
        'best_alpha_test_mse': alpha_mse, 'best_alpha_test_mae': alpha_mae,
        'shuffled_mse': shuffled_mse, 'shuffled_mae': shuffled_mae,
        'delta_learned_vs_gate0': learned_mse - gate0_mse,
        'delta_learned_vs_shuffled': learned_mse - shuffled_mse,
        'checkpoint': checkpoint_path,
    }
    row['verdict'] = classify(row)
    return row


def _broadcast(lam, y_ret):
    """lambda is [B, C] (scalar gate) or [B, C, pred_len]; y_ret is [B, pred_len, C]."""
    if lam.dim() == 2:
        return lam.unsqueeze(1)
    if lam.dim() == 3:
        return lam.permute(0, 2, 1)
    raise ValueError(f'unexpected lambda shape {tuple(lam.shape)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = analyse(args.checkpoint, args.max_batches)
    print(f"=== {row['dataset']}/{row['pred_len']} gate ablation ===")
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
