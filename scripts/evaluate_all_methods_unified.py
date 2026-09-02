#!/usr/bin/env python3
"""Audit: one evaluator, every method, and a check that it reproduces the original.

Two bugs made the recent diagnostics incomparable to the original CARTS numbers,
both from the same fact: RelationStage2 fuses in delta space and restores the
last-value offset only at the boundary.

  BUG 1  base_forecast called base_head(x) without restoring the offset
  BUG 2  y_final was rebuilt as y_base + lam*y_ret, but y_ret already carries
         the offset, so the sum double-counted it by lam*offset

The canonical metric is the one the original pipeline already used --
`mean((pred - target)**2)` over sample, horizon and channel, in normalized
space, on the best-validation checkpoint. Nothing new is defined here; this
script proves the reproduction and then scores every method with it.

Predictions are written to disk with metadata so evaluator differences and model
differences can never be conflated again.
"""

import argparse, hashlib, json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from utils.retrieval_diagnostics import (
    append_row, base_forecast, collect_split, load_stage2, unwrap,
)
from utils.utility_selection import (
    build_selection_cache, forecast_from_selection, utility_from_residuals,
)
from scripts.analyze_residual_oracle import prepare
from scripts.train_query_residual_selector import ResidualPredictor

COLUMNS = ['dataset', 'pred_len', 'method', 'unified_mse', 'unified_mae',
           'n_samples', 'shape', 'space', 'note']


def canonical_metrics(pred, target):
    """The original pipeline's metric: global mean over sample/horizon/channel.

    Shapes must match exactly -- broadcasting here is what silently turns a
    per-channel prediction into a different number.
    """
    if pred.shape != target.shape:
        raise ValueError(f'shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}')
    return (float((pred - target).square().mean()),
            float((pred - target).abs().mean()))


def reduction_breakdown(pred, target):
    """Every reduction order, to show which one the original evaluator uses."""
    e = (pred - target).square()
    return {
        'global_mean': float(e.mean()),
        'mean_over_horizon_then_all': float(e.mean(dim=1).mean()),
        'mean_over_channel_then_all': float(e.mean(dim=2).mean()),
        'mean_over_sample_then_all': float(e.mean(dim=0).mean()),
        'target_channel_only_OT': float(e[..., -1].mean()),
    }


def tensor_stats(name, t):
    return {'name': name, 'shape': tuple(t.shape), 'mean': float(t.mean()),
            'std': float(t.std()), 'min': float(t.min()), 'max': float(t.max())}


@torch.no_grad()
def run(stage2_ckpt, selector_ckpt, out_dir, csv, audit_csv, top_k=10, tau=0.1):
    experiment, args = load_stage2(stage2_ckpt)
    model = unwrap(experiment.model)
    tag = f'{args.data}_{args.pred_len}'
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- the original evaluator, called exactly as the original run did -----
    experiment._ensure_memory(); experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag='test', shuffle=False)
    experiment._build_retrieval_cache('test', loader)
    original = experiment._run_loader(loader, optimizer=None, split='test',
                                      epoch=0, setting='audit')
    original_mse = float(original['final_mse'])
    original_mae = float(original['final_mae'])

    collected = collect_split(experiment, 'test')
    true = collected['true']
    preds = {
        'original_carts_stage2': collected['final'],
        'stage2_base_head': collected['base'],
    }
    # gate=0 is the base head; alpha ablations must scale the pure correction
    preds['stage2_gate0'] = collected['base']
    preds['stage2_alpha1_pure'] = collected['base'] + collected['ret_pure']

    # --- residual-based methods, on the same split --------------------------
    if selector_ckpt and Path(selector_ckpt).exists():
        data = prepare(experiment, 'test')
        bundle = torch.load(selector_ckpt, map_location='cpu')
        predictor = ResidualPredictor(bundle['seq_len'], bundle['pred_len'])
        predictor.load_state_dict(bundle['state_dict'])
        predictor = predictor.to(experiment.device).eval()
        alpha = bundle['alpha']
        predicted = predictor(data['query_x'])
        preds['resdirect'] = (data['query_base'] + alpha * predicted).cpu()

        cache = build_selection_cache(experiment, model, data, bundle['pool_m'], alpha)
        scores = torch.zeros_like(cache['utility'])
        for slot_i, c in enumerate(cache['targets']):
            u = utility_from_residuals(predicted[:, :, c],
                                       data['memory_residual'][:, :, c], alpha)
            scores[:, slot_i] = u.gather(1, cache['pool_idx'][:, slot_i].to(u.device)).cpu()
        preds['ressel'] = forecast_from_selection(
            experiment, data, cache, scores, top_r=1).cpu()
        preds['residual_diagnostic_basehead'] = data['query_base'].cpu()

    # --- sanity: the canonical metric must reproduce the original ----------
    canon_mse, canon_mae = canonical_metrics(preds['original_carts_stage2'], true)
    reproduces = abs(canon_mse - original_mse) < 1e-4
    print(f'>> original evaluator : mse={original_mse:.6f} mae={original_mae:.6f}')
    print(f'>> canonical evaluator: mse={canon_mse:.6f} mae={canon_mae:.6f}  '
          f'reproduces={reproduces}')

    breakdown = reduction_breakdown(preds['original_carts_stage2'], true)
    print('>> reduction breakdown:', {k: round(v, 4) for k, v in breakdown.items()})

    # --- persist predictions with identity metadata ------------------------
    starts = collected['start']
    index_hash = hashlib.sha256(starts.numpy().tobytes()).hexdigest()[:16]
    for name, tensor in preds.items():
        torch.save(tensor, out_dir / f'{tag}__{name}.pt')
    torch.save(true, out_dir / f'{tag}__target.pt')
    torch.save(starts, out_dir / f'{tag}__sample_indices.pt')
    (out_dir / f'{tag}__metadata.json').write_text(json.dumps({
        'dataset': args.data, 'seq_len': int(args.seq_len),
        'pred_len': int(args.pred_len), 'split': 'test',
        'channels': int(true.size(-1)), 'n_samples': int(true.size(0)),
        'space': 'normalized (no inverse_transform)',
        'target_channels': 'all',
        'reduction': 'global mean over sample/horizon/channel',
        'sample_indices_sha256_16': index_hash,
        'stage2_checkpoint': stage2_ckpt, 'selector_checkpoint': selector_ckpt,
        'original_evaluator_mse': original_mse,
        'canonical_evaluator_mse': canon_mse,
        'reproduces_original': reproduces,
        'reduction_breakdown': breakdown,
        'methods': sorted(preds),
    }, indent=2))

    for name, tensor in sorted(preds.items()):
        mse, mae = canonical_metrics(tensor, true)
        note = ''
        if name == 'original_carts_stage2':
            note = 'reproduces original log' if reproduces else 'MISMATCH'
        if name == 'residual_diagnostic_basehead':
            note = 'base_head+offset; NOT the original No-Retrieval model'
        row = {'dataset': args.data, 'pred_len': int(args.pred_len),
               'method': name, 'unified_mse': mse, 'unified_mae': mae,
               'n_samples': int(true.size(0)), 'shape': str(tuple(tensor.shape)),
               'space': 'normalized', 'note': note}
        print(f'   {name:<32} mse={mse:.4f} mae={mae:.4f} {note}')
        if csv:
            append_row(csv, row, COLUMNS)

    if audit_csv:
        stats = [tensor_stats(n, t) for n, t in sorted(preds.items())]
        stats.append(tensor_stats('target', true))
        for s in stats:
            s.update({'dataset': args.data, 'pred_len': int(args.pred_len)})
            append_row(audit_csv, s,
                       ['dataset', 'pred_len', 'name', 'shape', 'mean', 'std', 'min', 'max'])
    return reproduces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selector', default='')
    p.add_argument('--out_dir', default='./predictions/evaluation_audit')
    p.add_argument('--csv', default='')
    p.add_argument('--audit_csv', default='')
    a = p.parse_args()
    ok = run(a.checkpoint, a.selector, a.out_dir, a.csv, a.audit_csv)
    print('REPRODUCES_ORIGINAL' if ok else 'REPRODUCTION_FAILED')


if __name__ == '__main__':
    main()
