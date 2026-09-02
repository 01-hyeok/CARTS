#!/usr/bin/env python3
"""Is a near-zero Stage-2 gate the right answer, or a training failure?

At long horizons the learned gate collapses to lambda ~= 0, which makes every
Stage-1 arm produce the same forecast. Two explanations look identical from the
outside: the retrieved futures are genuinely useless, or the gate failed to
learn a useful mixing weight.

Stage-2 fuses residually, `y_final = y_base + lambda * y_ret`. This loads a
trained Stage-2 checkpoint, caches y_base and y_ret on the validation split, and
sweeps a single scalar alpha in place of the learned lambda:

    best_alpha ~= 0            -> retrieval genuinely does not help
    best_alpha >> learned mean -> the gate, not the retrieval, is the problem
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage2_relation import Exp_Stage2_Relation  # noqa: E402


@torch.no_grad()
def sweep(checkpoint_path, split, alphas, max_batches):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'args' not in checkpoint:
        raise ValueError(f'checkpoint has no saved args: {checkpoint_path}')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    experiment = Exp_Stage2_Relation(args)
    experiment.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    experiment.model.eval()

    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    if getattr(model, 'fusion_mode', 'residual') != 'residual':
        raise ValueError(
            f'gate sweep assumes residual fusion; got {model.fusion_mode}'
        )

    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    data, loader = experiment._get_data(flag=split, shuffle=False)
    experiment._build_retrieval_cache(split, loader)

    base_parts, ret_parts, true_parts, lam_parts = [], [], [], []
    for index, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx
        )
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        retrieval_cache = experiment._cached_retrieval_for_batch(split, batch_start_idx)
        _, y_base, y_ret, _, lam, _ = experiment.model(
            batch_x=batch_x,
            memory_y=experiment.memory_y,
            valid_mask=cand_mask,
            key_bank=experiment.key_bank,
            memory_x_last=experiment.memory_x_last,
            retrieval_cache=retrieval_cache,
            target_y=batch_y,
            teacher_key_bank=getattr(experiment, 'teacher_key_bank', None),
        )
        base_parts.append(y_base.detach().float().cpu())
        ret_parts.append(y_ret.detach().float().cpu())
        true_parts.append(batch_y.detach().float().cpu())
        lam_parts.append(lam.detach().float().cpu())

    y_base = torch.cat(base_parts)
    y_ret = torch.cat(ret_parts)
    y_true = torch.cat(true_parts)
    if y_true.shape != y_base.shape:
        raise ValueError(
            f'target/prediction shape mismatch: {tuple(y_true.shape)} vs {tuple(y_base.shape)}'
        )
    learned_lambda = float(torch.cat(lam_parts).mean())

    rows = []
    for alpha in alphas:
        pred = y_base + alpha * y_ret
        mse = float((pred - y_true).square().mean())
        mae = float((pred - y_true).abs().mean())
        rows.append({'alpha': alpha, 'mse': mse, 'mae': mae})

    best = min(rows, key=lambda r: r['mse'])
    return {
        'dataset': args.data,
        'pred_len': int(args.pred_len),
        'split': split,
        'learned_gate_mean': learned_lambda,
        'best_alpha': best['alpha'],
        'best_alpha_val_mse': best['mse'],
        'best_alpha_val_mae': best['mae'],
        'alpha0_val_mse': next(r['mse'] for r in rows if r['alpha'] == 0.0),
        'checkpoint': checkpoint_path,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Stage-2 checkpoint.pth')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--alphas', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    parser.add_argument('--max_batches', type=int, default=0, help='0 = whole split')
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(',') if a.strip()]
    summary, rows = sweep(args.checkpoint, args.split, alphas, args.max_batches)

    print(f"=== {summary['dataset']}/{summary['pred_len']} gate sweep ({args.split}) ===")
    for row in rows:
        marker = ' <- best' if row['alpha'] == summary['best_alpha'] else ''
        print(f"  alpha={row['alpha']:.2f}  mse={row['mse']:.6f}  mae={row['mae']:.6f}{marker}")
    print(f"  learned_gate_mean: {summary['learned_gate_mean']:.6f}")
    print(f"  best_alpha: {summary['best_alpha']}  (alpha=0 mse {summary['alpha0_val_mse']:.6f})")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with open(path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
