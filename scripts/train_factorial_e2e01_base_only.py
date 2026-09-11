#!/usr/bin/env python3
"""TRACK-A-FACTORIAL-E2E01 -- Independent Base-only Forecaster.

THE primary forecasting reference for this experiment. Trained with retrieval
fully absent: no retrieval input, no retrieved future, no fusion, no gate.
One run per cell, four cells, identical protocol.

Relationship to the other two Stage-2 numbers this experiment reports
--------------------------------------------------------------------
  Independent Base-only Forecaster       <- THIS FILE. The forecasting
                                            baseline. `delta_mse_vs_independent_base`
                                            is measured against it.
  Frozen Host Retrieval-Ablated Branch   <- the host's own `y_base`, i.e. the
                                            branch that was CO-TRAINED with
                                            retrieval under
                                            `y_final = y_base + lambda*y_ret`.
                                            Diagnostic only. It is not a
                                            baseline and must never be called
                                            one: on ETTh1 H96 it measures
                                            0.6467 against the host's own
                                            0.3731, because it is half of a
                                            two-part model.
  Frozen Host Original Retrieval         <- the host's unforced `y_final`.
                                            Diagnostic reference only.

Reuse and provenance
--------------------
`models.RelationStage2.BaseForecastHead` is the SAME class production Stage-2
instantiates as `self.base_head`; `base_head_mode` defaults to the host's own
`shared_target_linear`. Data loading goes through `build_experiment`, the same
path the factorial Stage-1 trainer uses, so split, normalisation, seq_len,
pred_len and batch size are identical by construction.

The training loop is `scripts/train_oracle_scratch01_base_only.py`'s, kept
deliberately unchanged in behaviour. What IS different, on purpose: the test
metric here is computed with the *same* implementation the factorial Stage-2
evaluator uses -- sum of squared errors over element count, plus MAE, over the
same `valid_query` population -- because the spec requires an identical
MSE/MAE implementation between the baseline and the retrieval-augmented
numbers. The old script reported a batch-size-weighted mean of per-batch means
and no MAE, so it is not reused verbatim; it is also not modified.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.train_margutil01 import build_experiment


def train_epoch(base_head, loader, optimizer, device):
    base_head.train(True)
    total, n = 0.0, 0
    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        optimizer.zero_grad()
        y_pred = base_head(batch_x) + batch_x[:, -1:, :]
        loss = (y_pred - batch_y).pow(2).mean()
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch_x.size(0)
        n += batch_x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(base_head, exp, loader, device, top_k):
    """Identical metric implementation to
    `scripts/eval_factorial_e2e01_stage2.py`: element-wise SSE / element count,
    over the same `valid_query = counts >= top_k` population."""
    base_head.train(False)
    se, ae, n = 0.0, 0.0, 0.0
    has_nan = False
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        _, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= top_k
        y_pred = base_head(batch_x) + batch_x[:, -1:, :]
        if bool(torch.isnan(y_pred).any() or torch.isinf(y_pred).any()):
            has_nan = True
        yp, by = y_pred[valid_query], batch_y[valid_query]
        se += float((yp - by).pow(2).sum())
        ae += float((yp - by).abs().sum())
        n += float(yp.numel())
    return {'mse': se / max(n, 1), 'mae': ae / max(n, 1),
            'n_elements': n, 'has_nan': has_nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True,
                    help='Stage-1 reference checkpoint -- ARGS ONLY, weights never loaded')
    ap.add_argument('--stage2_host', required=True,
                    help='host checkpoint; read ONLY to copy base_head_mode and assert agreement')
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_factorial_e2e')
    ap.add_argument('--out_dir', default='results/track_a_factorial_e2e')
    ap.add_argument('--train_epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--base_head_mode', default=None)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_args = dict(torch.load(cli.stage2_host, map_location='cpu')['args'])
    hp_source = {}

    def resolve(name, ref_key, default, default_note):
        v = getattr(cli, name)
        if v is not None:
            hp_source[name] = {'value': v, 'source': 'CLI override'}
        elif ref_key in ref_args and ref_args[ref_key] is not None:
            setattr(cli, name, ref_args[ref_key])
            hp_source[name] = {'value': ref_args[ref_key],
                               'source': f'Stage-1 reference args["{ref_key}"]'}
        else:
            setattr(cli, name, default)
            hp_source[name] = {'value': default, 'source': default_note}
        return getattr(cli, name)

    resolve('learning_rate', 'learning_rate', 1e-3, 'Stage-1 default')
    resolve('batch_size', 'batch_size', 32, 'Stage-1 default')
    resolve('train_epochs', 'train_epochs', 10, 'Stage-1 default')
    resolve('patience', 'patience', 5, 'Stage-1 default')
    resolve('seed', 'seed', 0, 'Stage-1 default')
    resolve('top_k', 'top_k', 10, 'Stage-1 default')
    resolve('weight_decay', 'weight_decay', 0.0,
            'Stage-1 exp/exp_stage1_relation.py::_select_optimizer -> Adam default (0.0)')
    if cli.base_head_mode is None:
        cli.base_head_mode = host_args.get('base_head_mode', 'shared_target_linear')
        hp_source['base_head_mode'] = {
            'value': cli.base_head_mode,
            'source': 'Stage-2 host args["base_head_mode"] (architecture must match the '
                      'branch the retrieval-augmented arms are compared against)'}
    hp_source['optimizer'] = {'value': 'Adam',
                              'source': 'Stage-1 exp_stage1_relation.py::_select_optimizer'}
    hp_source['lr_schedule'] = {'value': 'constant',
                                'source': 'same constant-LR choice as the factorial Stage-1 arms'}
    hp_source['checkpoint_criterion'] = {'value': 'min validation MSE',
                                         'source': 'this experiment (no retrieval metric exists here)'}

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.seed,
    })
    torch.manual_seed(int(cli.seed))
    device = exp.device
    exp._ensure_memory()

    base_head = BaseForecastHead(int(args.seq_len), int(args.pred_len), int(args.enc_in),
                                 mode=cli.base_head_mode).to(device)
    optimizer = torch.optim.Adam(base_head.parameters(), lr=float(cli.learning_rate),
                                 weight_decay=float(cli.weight_decay))

    ckpt_dir = Path(cli.checkpoints) / cli.cell / '_independent_base_only'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    fingerprint = {
        'cell': cli.cell, 'role': 'Independent Base-only Forecaster',
        'retrieval_used': False, 'fusion_used': False, 'gate_used': False,
        'base_head_mode': cli.base_head_mode,
        'trainable_params': sum(p.numel() for p in base_head.parameters()),
        'seq_len': int(args.seq_len), 'pred_len': int(args.pred_len),
        'enc_in': int(args.enc_in), 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'weight_decay': cli.weight_decay,
        'train_epochs': cli.train_epochs, 'patience': cli.patience, 'seed': cli.seed,
        'top_k_for_valid_query_filter': cli.top_k,
        'metric_implementation': 'SSE/element_count over valid_query -- identical to '
                                 'scripts/eval_factorial_e2e01_stage2.py',
        'hyperparameter_provenance': hp_source,
        'reference_ckpt': cli.reference_ckpt, 'stage2_host_read_for': 'base_head_mode only',
        'is_smoke_run': bool(cli.limit_batches),
    }
    print(f"[base_only] {cli.cell} params={fingerprint['trainable_params']} "
          f"mode={cli.base_head_mode} lr={cli.learning_rate} epochs={cli.train_epochs}")

    best = {'val': float('inf'), 'epoch': -1}
    history = []
    t0 = time.time()
    for epoch in range(1, int(cli.train_epochs) + 1):
        tr = train_epoch(base_head, train_loader, optimizer, device)
        va = evaluate(base_head, exp, val_loader, device, cli.top_k)
        history.append({'epoch': epoch, 'train_mse': tr, 'val_mse': va['mse'],
                        'val_mae': va['mae'], 'val_has_nan': va['has_nan']})
        payload = {'base_head_state_dict': base_head.state_dict(), 'args': vars(args),
                   'epoch': epoch, 'val_mse': va['mse'], 'base_head_mode': cli.base_head_mode,
                   'fingerprint': fingerprint}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if va['mse'] < best['val']:
            best = {'val': va['mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[base_only] {cli.cell} epoch {epoch} train_mse={tr:.6f} "
              f"val_mse={va['mse']:.6f} val_mae={va['mae']:.6f}")
        if epoch - best['epoch'] >= int(cli.patience):
            print(f"[base_only] early stop at epoch {epoch} (best={best['epoch']})")
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    base_head.load_state_dict(bl['base_head_state_dict'])
    te = evaluate(base_head, exp, test_loader, device, cli.top_k)
    if te['has_nan']:
        raise SystemExit('[ABORT] Independent Base-only produced NaN/Inf on test')

    summary = {
        'cell': cli.cell, 'role': 'Independent Base-only Forecaster',
        'best_epoch': best['epoch'], 'best_val_mse': best['val'],
        'independent_base_only_mse': te['mse'], 'independent_base_only_mae': te['mae'],
        'n_test_elements': te['n_elements'],
        'final_epoch': history[-1]['epoch'], 'history': history,
        'wall_clock_seconds': time.time() - t0,
        'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / 'independent_base_only.json').write_text(json.dumps(summary, indent=2))
    with open(cell_dir / 'epoch_metrics_independent_base_only.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        for r in history:
            w.writerow(r)
    print(f"[base_only] done. {cli.cell} best_epoch={best['epoch']} "
          f"test_mse={te['mse']:.6f} test_mae={te['mae']:.6f}")


if __name__ == '__main__':
    main()
