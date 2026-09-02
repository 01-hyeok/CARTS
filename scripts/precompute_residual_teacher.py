#!/usr/bin/env python3
"""Base-forecast residuals for the Residual-KL teacher, at any pool size.

Unlike measured utility, the residual target needs no Stage-2 forward per
candidate -- only each window's base forecast. Caching the residuals themselves
rather than a precomputed score matrix is what lets this teacher scale: scoring
100 or 34000 candidates is the same matmul at train time, and the cache stays a
few hundred megabytes instead of growing with the pool.

    R_q = Y_q - base(q)        query residuals, per split
    R_k = Y_k - base(k)        memory residuals, once
    S_R(q, k) = -MSE(R_q, R_k)

Base forecasts come from the reference Stage-2's own forward, never from a
reconstruction.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import load_stage2, unwrap  # noqa: E402
from utils.utility_teacher import base_forecast_via_forward  # noqa: E402

CACHE_VERSION = 1


@torch.no_grad()
def build(experiment, splits):
    model = unwrap(experiment.model)
    model.eval()
    experiment._ensure_memory()
    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    memory_x = torch.from_numpy(experiment.memory_bank.memory_x).float().to(device)
    memory_residual = (
        memory_y - base_forecast_via_forward(model, memory_x, memory_y, memory_x_last)
    ).cpu()
    del memory_x

    out = {'memory_residual': memory_residual, 'splits': {}}
    for split in splits:
        _, loader = experiment._get_data(flag=split, shuffle=False)
        residuals, starts = [], []
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x, batch_y, batch_start_idx = experiment._move_batch(
                batch_x, batch_y, batch_start_idx)
            base = base_forecast_via_forward(model, batch_x, memory_y, memory_x_last)
            if base.shape != batch_y.shape:
                raise ValueError(f'base {tuple(base.shape)} != target {tuple(batch_y.shape)}')
            residuals.append((batch_y - base).cpu())
            starts.extend(int(value) for value in batch_start_idx.cpu().tolist())
        out['splits'][split] = {
            'query_residual': torch.cat(residuals),
            'starts': torch.tensor(starts, dtype=torch.long),
            'start_to_row': {start: row for row, start in enumerate(starts)},
        }
        print(f'  {split}: {len(starts)} queries')
    out['meta'] = {
        'version': CACHE_VERSION, 'dataset': experiment.args.data,
        'pred_len': int(experiment.args.pred_len), 'seq_len': int(experiment.args.seq_len),
        'reference_stage2': experiment.reference_checkpoint,
    }
    return out


def load(path):
    cache = torch.load(path, map_location='cpu')
    if cache.get('meta', {}).get('version') != CACHE_VERSION:
        raise ValueError(f'residual cache at {path} has the wrong version; rebuild it')
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_dir', default='./cache/residual_teacher')
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    experiment, saved = load_stage2(args.checkpoint)
    experiment.reference_checkpoint = args.checkpoint
    target = Path(args.out_dir) / f'{saved.data}_pred{saved.pred_len}.pt'
    if target.exists() and not args.force:
        print(f'[skip] {target} already exists')
        return
    cache = build(experiment, [s.strip() for s in args.splits.split(',')])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, target)
    print(f'[done] {target} memory_residual={tuple(cache["memory_residual"].shape)}')


if __name__ == '__main__':
    main()
