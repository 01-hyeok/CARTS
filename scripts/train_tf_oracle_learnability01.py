#!/usr/bin/env python3
"""TRACK-A-TF-ORACLE-LEARNABILITY01 -- Teacher-Forced Individual vs Greedy Set
Oracle learnability, Hard Choice CE only (no new losses).

One invocation = one arm (`individual_tf_cosine` or `set_tf_cosine`) for one
cell. `--target` selects Individual vs Greedy Set; prefix_policy is ALWAYS
`tf` (Oracle-prefix teacher forcing, spec's `S_t* = S_{t-1}* U {i_t*}`) and
the scorer is ALWAYS cosine -- this experiment does not sweep those axes.

Structurally this is TRACK-A-SET-LOSS-CONTROL02's pattern (RNG separation +
batch-order hashing via `scripts.rng_control01`, `run_sequence` imported not
duplicated) applied on top of `scripts.train_factorial_e2e01`'s OWN
`train_epoch`/`eval_epoch` (imported, not reimplemented -- unlike CONTROL02,
those functions already generalize over `--target`/`--channelwise_backward`/
`--memsafe`/`--oracle_compute_impl`, so there is nothing arm-specific left to
duplicate). `train_epoch` gained an additive `record_batch_order=True` opt-in
this round (default False = no behavior change) purely to reuse
`batch_order_sha256` instead of re-deriving it.

What is genuinely NEW here (not present in train_factorial_e2e01.py or any
CONTROL0x trainer): `teacher_forced_diagnostics()`, a no-grad, eval-mode pass
that walks `run_sequence(..., prefix_policy='tf', free_running=False)` --
i.e. the SAME oracle-prefix teacher-forcing `run_sequence` already
implements, just called outside the training loop so dropout/optimizer
updates cannot contaminate the numbers the spec calls "train learnability"
(online training-loop statistics are explicitly NOT trusted for this). Run
three ways per epoch:
  - a FIXED train-query subset (seed=0-selected once by `_build_fixed_subset`,
    saved as an artifact, reused identically across every epoch and both
    arms of a cell) -- true train-TF learnability, uncontaminated.
  - the full val split -- val-TF.
  - the full test split, at `--test_epochs` -- test-TF.
Free-running (FR-Agg + free-running Oracle-action diagnostics) is the
EXISTING `eval_epoch` primary pass, unmodified, called separately so TF and
free-running are never conflated (spec's two evaluation modes).
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
from scripts.train_factorial_e2e01 import (HostScorer, candidate_weights, encode_raw,
                                           eval_epoch, log_oracle_compute_resolution,
                                           param_displacement,
                                           representation_diagnostics,
                                           resolve_oracle_compute_impl, run_sequence,
                                           state_sha, step_rank_diagnostics, train_epoch)
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = {'individual_tf_cosine': 'individual', 'set_tf_cosine': 'greedy_set'}


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


def _build_fixed_subset(exp, subset_size, subset_seed):
    """Materializes a FIXED random subset of `subset_size` train queries,
    selected once by `subset_seed` (default 0), reused identically across
    every epoch and both arms of a cell (caller passes the same tensors to
    both). Train/val/test are disjoint chronological splits by construction
    (Dataset_Custom / Dataset_Solar), so this can never overlap val/test.
    Returns (x, y, start_idx, selected_indices) -- indices saved by the
    caller as an artifact.
    """
    _, loader = exp._get_data(flag='train', shuffle=False)
    xs, ys, starts = [], [], []
    for bx, by, bstart in loader:
        xs.append(bx)
        ys.append(by)
        starts.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    start_all = torch.cat(starts, dim=0)
    n = x_all.size(0)
    k = min(subset_size, n)
    g = torch.Generator().manual_seed(subset_seed)
    idx = torch.randperm(n, generator=g)[:k]
    idx, _ = idx.sort()
    return x_all[idx], y_all[idx], start_all[idx], idx


@torch.no_grad()
def _materialize_loader(loader):
    xs, ys, starts = [], [], []
    for bx, by, bstart in loader:
        xs.append(bx)
        ys.append(by)
        starts.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
    return torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(starts, 0)


@torch.no_grad()
def teacher_forced_diagnostics(exp, args, host, model, set_conditioner, metric, cli,
                               batch_x, batch_y, batch_start_idx, channels, device,
                               memsafe=False, batch_size=32):
    """No-grad, eval-mode Oracle-prefix teacher-forced pass (prefix_policy=
    'tf', free_running=False) -- NOT the online training loop, NOT the
    free-running rollout. `batch_x`/`batch_y`/`batch_start_idx` are already-
    materialized tensors (a fixed subset or a full split collected once by
    the caller); chunked into `batch_size`-sized pieces here only to bound
    peak memory, which changes nothing about what is computed.
    """
    model.train(False)
    set_conditioner.train(False)
    if metric is not None:
        metric.train(False)
    oc = getattr(cli, 'resolved_oracle_compute', None) or {
        'choice_ce_impl': 'reference', 'individual_impl': 'reference', 'greedy_set_impl': 'reference'}

    ce_sum, ce_n = 0.0, 0
    step_sums, step_n = {}, {}
    n = batch_x.size(0)
    for s in range(0, n, batch_size):
        bx = batch_x[s:s + batch_size].float().to(device)
        by = batch_y[s:s + batch_size].float().to(device)
        bstart = batch_start_idx[s:s + batch_size]
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, bx, c)
            memory_c, offset_c = memory_value(args, bx, exp.memory_y, exp.memory_x_last, c)
            query_future = by[:, :, c]
            host_scores = host.scores(bx, c, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, cli.tau_topk)

            if memsafe:
                losses, diags, picks, steps = run_sequence(
                    z_q, E, cand_mask, set_conditioner, metric, w_host,
                    None, query_future, cli.target, 'tf',
                    cli.tau_choice, cli.top_k, cli.chunk_size, free_running=False,
                    choice_ce_impl=oc['choice_ce_impl'], individual_impl=oc['individual_impl'],
                    greedy_set_impl=oc['greedy_set_impl'],
                    memsafe=True, memory_c=memory_c, offset_c=offset_c)
                futures_diag = memory_c + offset_c.view(-1, 1, 1)
            else:
                futures = memory_c + offset_c.view(-1, 1, 1)
                losses, diags, picks, steps = run_sequence(
                    z_q, E, cand_mask, set_conditioner, metric, w_host,
                    futures, query_future, cli.target, 'tf',
                    cli.tau_choice, cli.top_k, cli.chunk_size, free_running=False,
                    choice_ce_impl=oc['choice_ce_impl'], individual_impl=oc['individual_impl'],
                    greedy_set_impl=oc['greedy_set_impl'])
                futures_diag = futures

            for t, (loss_t, d) in enumerate(zip(losses, steps)):
                sdiag = step_rank_diagnostics(d['u_hat'], d['u_target'], d['valid_now'],
                                              d['oracle_idx'], d['model_idx'],
                                              futures_diag, query_future)
                sdiag['choice_ce'] = float(loss_t)
                ce_sum += float(loss_t)
                ce_n += 1
                for kk, vv in sdiag.items():
                    if isinstance(vv, float) and vv == vv:
                        key = f't{t + 1}::{kk}'
                        step_sums[key] = step_sums.get(key, 0.0) + vv
                        step_n[key] = step_n.get(key, 0) + 1

    per_step = {kk: step_sums[kk] / max(step_n[kk], 1) for kk in step_sums}
    out = {'tf_choice_ce': ce_sum / max(ce_n, 1)}
    for name in ('oracle_action_acc', 'expert_rank_mean', 'expert_rank_median',
                 'expert_regret_mean', 'expert_containment_at_1', 'expert_containment_at_5',
                 'expert_containment_at_10', 'ndcg_at_10', 'spearman'):
        vals = [per_step[k] for k in per_step if k.endswith('::' + name)]
        out[name] = sum(vals) / max(len(vals), 1) if vals else float('nan')
    stepwise = []
    for t in range(cli.top_k):
        row = {'step': t + 1}
        for kk, vv in per_step.items():
            if kk.startswith(f't{t + 1}::'):
                row[kk.split('::', 1)[1]] = vv
        stepwise.append(row)
    return out, stepwise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--arm_name', choices=list(ARMS), required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_tf_oracle_learnability01')
    ap.add_argument('--out_dir', default='results/TRACK-A-TF-ORACLE-LEARNABILITY01')
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

    host_ck = torch.load(cli.stage2_host, map_location='cpu')
    host_args = dict(host_ck['args'])
    cli.tau_choice = 0.1
    cli.tau_topk = float(host_args['tau_topk'])

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    if int(host.top_k) != int(cli.top_k):
        raise SystemExit(f'[ABORT] host top_k={host.top_k} != top_k={cli.top_k}')
    stage2_host_sha256 = _file_sha256(cli.stage2_host)

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

    # ---- fixed train-query subset for uncontaminated train-TF learnability ----
    sub_x, sub_y, sub_start, sub_idx = _build_fixed_subset(exp, cli.train_subset_size,
                                                            cli.train_subset_seed)
    # materialized ONCE (deterministic, shuffle=False) -- reused every epoch,
    # not re-iterated from the DataLoader each time.
    val_x, val_y, val_start = _materialize_loader(val_loader)
    test_x, test_y, test_start = _materialize_loader(test_loader)
    subset_path = cell_dir / f'train_tf_subset_indices_{cli.arm_name}.json'
    subset_path.write_text(json.dumps({
        'subset_size': int(sub_x.size(0)), 'subset_seed': cli.train_subset_seed,
        'indices_sha256': hashlib.sha256(sub_idx.numpy().tobytes()).hexdigest(),
        'indices': sub_idx.tolist(),
    }, indent=2))

    fingerprint = {
        'exp': 'TRACK-A-TF-ORACLE-LEARNABILITY01', 'cell': cli.cell, 'arm': cli.arm_name,
        'axis_oracle': cli.target, 'axis_prefix': 'tf', 'axis_score': 'cosine',
        'loss_name': 'hard_choice_ce', 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'channels': channels,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'stage2_host_sha256': stage2_host_sha256, 'param_count': param_count,
        'top_k': cli.top_k, 'tau_choice': cli.tau_choice, 'tau_topk': cli.tau_topk,
        'learning_rate': cli.learning_rate, 'batch_size': cli.batch_size,
        'epochs': cli.train_epochs, 'patience': cli.patience, 'weight_decay': cli.weight_decay,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse (all channels)',
        'optimizer': 'Adam', 'reference_ckpt': cli.reference_ckpt, 'stage2_host': cli.stage2_host,
        'oracle_compute_impl': cli.oracle_compute_impl, 'channelwise_backward': cli.channelwise_backward,
        'memsafe': cli.memsafe, 'train_subset_size': int(sub_x.size(0)),
        'train_subset_seed': cli.train_subset_seed, 'code_commit': code_commit,
        'reused_run_sequence': True, 'reused_train_eval_epoch': True,
        'run_sequence_source': 'scripts.train_factorial_e2e01.run_sequence (unmodified dispatch)',
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[tf_oracle_learnability01] {cli.cell}/{cli.arm_name} target={cli.target} '
         f'encoder_init_sha={encoder_init_sha256[:16]} sc_init_sha={set_conditioner_init_sha256[:16]} '
         f'channels={channels} init_seed={cli.init_seed} loader_seed={cli.loader_seed}')

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
                   'val_free_running_aggregate_future_mse': row['val_free_running_aggregate_future_mse']}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if row['val_free_running_aggregate_future_mse'] < best['val']:
            best = {'val': row['val_free_running_aggregate_future_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')

        print(f"[tf_oracle_learnability01] {cli.arm_name} epoch {epoch} "
             f"train_ce={tr['train_choice_ce']:.5f} val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
             f"train_subset_tf_acc={va_tf['oracle_action_acc']:.4f} val_tf_acc={val_split_tf['oracle_action_acc']:.4f} "
             f"enc_gn={tr['encoder_grad_norm']:.5f} batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[tf_oracle_learnability01] early stop at epoch {epoch} (best={best["epoch"]})')
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
        'exp': 'TRACK-A-TF-ORACLE-LEARNABILITY01', 'cell': cli.cell, 'arm': cli.arm_name,
        'target': cli.target, 'best_epoch': best['epoch'],
        'best_val_free_running_aggregate_future_mse': best['val'],
        'test_free_running_aggregate_future_mse': te_fr['free_running_aggregate_future_mse'],
        'test_free_running_internal': te_fr, 'test_tf': te_tf, 'train_subset_tf_at_best': train_subset_tf_best,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[tf_oracle_learnability01] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
         f"best_val_fr_agg={best['val']:.6f} test_fr_agg={te_fr['free_running_aggregate_future_mse']:.6f} "
         f"test_tf_acc={te_tf['oracle_action_acc']:.4f}")


if __name__ == '__main__':
    main()
