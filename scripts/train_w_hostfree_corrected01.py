#!/usr/bin/env python3
"""TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 -- CASE B corrected copy of
`train_setoracle_hostfree01.py` for B Host-Free (TRACK-A-SETORACLE-HOSTFREE01).

This file is a byte-for-byte copy of the original EXCEPT for exactly one
change, marked below with `# TRACK-W CORRECTION`: the checkpoint-selection
/early-stopping criterion is changed from
`val_free_running_aggregate_future_mse` (free-running, eval-only metric)
to `val_tf_choice_ce` (teacher-forced Choice CE -- the SAME objective the
model is actually trained on: `hard_choice_ce`, teacher-forced). The
original file is left completely untouched; this is a NEW, separate
script used ONLY for the 5 (cell, arm) combinations whose OLD run
early-stopped before the full intended epoch trajectory existed (CASE B
per the correction's own completeness rule -- pure re-selection from
already-existing epochs is not valid there, since training under the
OLD criterion may have stopped before the NEW criterion's own optimum
was reached).

No training math, teacher, candidate mask, model, SetConditioner,
optimizer, LR, batch size, or seed differs from the original in any way.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.rng_control01 import (batch_order_sha256, make_loader_generator,
                                   set_global_seeds)
from scripts.train_factorial_e2e01 import (candidate_weights, encode_raw, eval_epoch,
                                           log_oracle_compute_resolution,
                                           param_displacement,
                                           representation_diagnostics,
                                           resolve_oracle_compute_impl, run_sequence,
                                           state_sha, step_rank_diagnostics, train_epoch)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_setoracle_hostfree01 import (ARMS, UniformHost,
                                                 _build_fixed_subset,
                                                 _materialize_loader,
                                                 teacher_forced_diagnostics)


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm_name', choices=list(ARMS), required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_w_checkpoint_criterion_correction01/B_hostfree_stage1')
    ap.add_argument('--out_dir', default='results/TRACK-W-CHECKPOINT-CRITERION-CORRECTION01/B_hostfree_stage1')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--test_epochs', default='1,5,10')
    ap.add_argument('--train_subset_size', type=int, default=512)
    ap.add_argument('--train_subset_seed', type=int, default=0)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--oracle_compute_impl', choices=['reference', 'optimized', 'safe'],
                    default='safe')
    ap.add_argument('--channelwise_backward', action='store_true')
    ap.add_argument('--memsafe', action='store_true')
    cli = ap.parse_args()
    cli.target = ARMS[cli.arm_name]
    cli.prefix_policy = 'tf'

    cli.resolved_oracle_compute = resolve_oracle_compute_impl(cli.oracle_compute_impl, cli.chunk_size)
    log_oracle_compute_resolution(cli.oracle_compute_impl, cli.resolved_oracle_compute, cli.chunk_size)

    cli.tau_choice = 0.1
    cli.tau_topk = 1.0  # irrelevant under UniformHost -- kept only for interface parity

    # [MODEL-SELECTION AUDIT] -- mandatory per the project's checkpoint-selection
    # governance policy (2026-10-01): print before any expensive GPU run.
    print('[MODEL-SELECTION AUDIT]')
    print('Training objective:          hard_choice_ce (teacher-forced Choice CE)')
    print('Validation selection metric:  val_tf_choice_ce (teacher-forced Choice CE)')
    print('Early-stopping metric:        val_tf_choice_ce (SAME as selection metric)')
    print('Are they identical?          YES')
    print('Decision:                     APPROVED (TYPE A/B match; no TYPE C diagnostic metric used for selection)')

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = UniformHost(cli.top_k, tau_topk=cli.tau_topk)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.init_seed, 'top_k': cli.top_k,
        'tau_topk': cli.tau_choice,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))
    metric = None

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)

    code_commit = _git_commit()
    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['model_state_dict'])
        set_conditioner.load_state_dict(blob['sc_state_dict'])
        got_model = state_sha(model.state_dict())
        got_sc = state_sha(set_conditioner.state_dict())
        if got_model != blob['encoder_init_sha256'] or got_sc != blob['set_conditioner_init_sha256']:
            raise SystemExit(f'[ISSUE][ABORT] shared-init SHA mismatch: model {got_model} != '
                             f'{blob["encoder_init_sha256"]} or sc {got_sc} != '
                             f'{blob["set_conditioner_init_sha256"]}')
        if int(blob.get('seed', -1)) != int(cli.init_seed):
            raise SystemExit(f'[ISSUE][ABORT] shared-init seed={blob.get("seed")} != '
                             f'this run\'s init_seed={cli.init_seed}')
        encoder_init_sha256, set_conditioner_init_sha256 = got_model, got_sc
    else:
        encoder_init_sha256 = state_sha(model.state_dict())
        set_conditioner_init_sha256 = state_sha(set_conditioner.state_dict())
        if cli.shared_init_out:
            Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'sc_state_dict': {k: v.detach().cpu() for k, v in set_conditioner.state_dict().items()},
                        'encoder_init_sha256': encoder_init_sha256,
                        'set_conditioner_init_sha256': set_conditioner_init_sha256,
                        'seed': cli.init_seed, 'cell': cli.cell, 'code_commit': code_commit},
                       cli.shared_init_out)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    param_count = (sum(p.numel() for p in model.parameters())
                  + sum(p.numel() for p in set_conditioner.parameters()))

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(set_conditioner.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    sub_x, sub_y, sub_start, sub_idx = _build_fixed_subset(exp, cli.train_subset_size,
                                                            cli.train_subset_seed)
    val_x, val_y, val_start = _materialize_loader(val_loader)
    test_x, test_y, test_start = _materialize_loader(test_loader)
    subset_path = cell_dir / f'train_tf_subset_indices_{cli.arm_name}.json'
    subset_path.write_text(json.dumps({
        'subset_size': int(sub_x.size(0)), 'subset_seed': cli.train_subset_seed,
        'indices_sha256': hashlib.sha256(sub_idx.numpy().tobytes()).hexdigest(),
        'indices': sub_idx.tolist(),
    }, indent=2))

    fingerprint = {
        'exp': 'TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 (CASE B retrain of TRACK-A-SETORACLE-HOSTFREE01)',
        'cell': cli.cell, 'arm': cli.arm_name,
        'axis_oracle': cli.target, 'axis_prefix': 'tf', 'axis_score': 'cosine',
        'aggregation_weighting': 'uniform (UniformHost, no Stage-2 host, no embeddings in aggregation)',
        'loss_name': 'hard_choice_ce', 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'channels': channels,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'param_count': param_count,
        'top_k': cli.top_k, 'tau_choice': cli.tau_choice,
        'learning_rate': cli.learning_rate, 'batch_size': cli.batch_size,
        'epochs': cli.train_epochs, 'patience': cli.patience, 'weight_decay': cli.weight_decay,
        # TRACK-W CORRECTION: this is the ONLY substantive change vs the original script.
        'checkpoint_criterion': 'min val_tf_choice_ce (teacher-forced Choice CE -- CORRECTED, '
                                'matches training objective; OLD was min val_free_running_aggregate_future_mse)',
        'optimizer': 'Adam', 'reference_ckpt': cli.reference_ckpt,
        'oracle_compute_impl': cli.oracle_compute_impl, 'channelwise_backward': cli.channelwise_backward,
        'memsafe': cli.memsafe, 'train_subset_size': int(sub_x.size(0)),
        'train_subset_seed': cli.train_subset_seed, 'code_commit': code_commit,
        'reused_run_sequence': True, 'reused_train_eval_epoch': True,
        'run_sequence_source': 'scripts.train_factorial_e2e01.run_sequence (unmodified dispatch)',
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[train_w_hostfree_corrected] {cli.cell}/{cli.arm_name} target={cli.target} '
         f'encoder_init_sha={encoder_init_sha256[:16]} sc_init_sha={set_conditioner_init_sha256[:16]} '
         f'channels={channels} init_seed={cli.init_seed} loader_seed={cli.loader_seed} '
         f'CORRECTED_CRITERION=val_tf_choice_ce')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, metric, cli, train_loader,
                         channels, device, channelwise_backward=cli.channelwise_backward,
                         memsafe=cli.memsafe, record_batch_order=True)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')

        va_fr, _ = eval_epoch(exp, args, host, model, set_conditioner, metric, cli, val_loader,
                              channels, device, memsafe=cli.memsafe)
        va_tf, _ = teacher_forced_diagnostics(exp, args, host, model, set_conditioner, metric, cli,
                                              sub_x, sub_y, sub_start, channels, device,
                                              memsafe=cli.memsafe, batch_size=cli.batch_size)
        val_split_tf, _ = teacher_forced_diagnostics(
            exp, args, host, model, set_conditioner, metric, cli,
            val_x, val_y, val_start, channels, device,
            memsafe=cli.memsafe, batch_size=cli.batch_size)

        with torch.no_grad():
            rep = representation_diagnostics(encode_raw(model, exp.memory_x[:256], channels[0]))
        rep['encoder_param_displacement'] = param_displacement(model, init_state)

        row = {'epoch': epoch, **tr,
               'val_free_running_aggregate_future_mse': va_fr['free_running_aggregate_future_mse'],
               **{f'val_fr_{k}': v for k, v in va_fr.items() if k != 'free_running_aggregate_future_mse'},
               **{f'train_subset_tf_{k}': v for k, v in va_tf.items()},
               **{f'val_tf_{k}': v for k, v in val_split_tf.items()},
               **{f'rep_{k}': v for k, v in rep.items() if not isinstance(v, list)}}
        if epoch in test_at:
            te_fr, _ = eval_epoch(exp, args, host, model, set_conditioner, metric, cli, test_loader,
                                  channels, device, memsafe=cli.memsafe)
            row['test_free_running_aggregate_future_mse'] = te_fr['free_running_aggregate_future_mse']

        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(),
                   'set_conditioner_state_dict': set_conditioner.state_dict(),
                   'args': vars(args), 'epoch': epoch, 'fingerprint': fingerprint,
                   'val_free_running_aggregate_future_mse': row['val_free_running_aggregate_future_mse'],
                   'val_tf_choice_ce': val_split_tf['tf_choice_ce']}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        # TRACK-W CORRECTION: selection criterion changed from
        # row['val_free_running_aggregate_future_mse'] to val_split_tf['tf_choice_ce'].
        # Nothing else in this training loop differs from the original script.
        if val_split_tf['tf_choice_ce'] < best['val']:
            best = {'val': val_split_tf['tf_choice_ce'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')

        print(f"[train_w_hostfree_corrected] {cli.arm_name} epoch {epoch} "
             f"train_ce={tr['train_choice_ce']:.5f} val_tf_choice_ce={val_split_tf['tf_choice_ce']:.6f} "
             f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
             f"train_subset_tf_acc={va_tf['oracle_action_acc']:.4f} val_tf_acc={val_split_tf['oracle_action_acc']:.4f} "
             f"enc_gn={tr['encoder_grad_norm']:.5f} (best={best['epoch']}:{best['val']:.6f}) "
             f"batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[train_w_hostfree_corrected] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te_fr, test_stepwise_fr = eval_epoch(exp, args, host, model, set_conditioner, metric, cli,
                                         test_loader, channels, device, collect_stepwise=True,
                                         memsafe=cli.memsafe)
    te_tf, test_stepwise_tf = teacher_forced_diagnostics(
        exp, args, host, model, set_conditioner, metric, cli,
        test_x, test_y, test_start, channels, device,
        memsafe=cli.memsafe, batch_size=cli.batch_size)
    train_subset_tf_best, train_subset_stepwise_tf = teacher_forced_diagnostics(
        exp, args, host, model, set_conditioner, metric, cli, sub_x, sub_y, sub_start,
        channels, device, memsafe=cli.memsafe, batch_size=cli.batch_size)

    with open(cell_dir / f'epoch_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    with open(cell_dir / f'stepwise_free_running_test_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in test_stepwise_fr for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in test_stepwise_fr:
            w.writerow(r)
    with open(cell_dir / f'stepwise_tf_test_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in test_stepwise_tf for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in test_stepwise_tf:
            w.writerow(r)
    with open(cell_dir / f'stepwise_tf_train_subset_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in train_subset_stepwise_tf for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in train_subset_stepwise_tf:
            w.writerow(r)
    (cell_dir / f'batch_order_hashes_{cli.arm_name}.json').write_text(json.dumps(batch_order_hashes, indent=2))

    summary = {
        'exp': 'TRACK-W-CHECKPOINT-CRITERION-CORRECTION01', 'cell': cli.cell, 'arm': cli.arm_name,
        'target': cli.target, 'best_epoch': best['epoch'],
        'best_val_tf_choice_ce': best['val'],
        'case': 'B (fresh retrain -- OLD criterion early-stopped before full trajectory existed)',
        'test_free_running_aggregate_future_mse': te_fr['free_running_aggregate_future_mse'],
        'test_free_running_internal': te_fr, 'test_tf': te_tf, 'train_subset_tf_at_best': train_subset_tf_best,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2, default=str))
    (cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[train_w_hostfree_corrected] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
         f"best_val_tf_choice_ce={best['val']:.6f} test_fr_agg={te_fr['free_running_aggregate_future_mse']:.6f} "
         f"test_tf_acc={te_tf['oracle_action_acc']:.4f}")


if __name__ == '__main__':
    main()
