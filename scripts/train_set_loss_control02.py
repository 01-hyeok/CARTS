#!/usr/bin/env python3
"""TRACK-A-SET-LOSS-CONTROL02 -- corrects TRACK-A-SET-LOSS-CONTROL01's
UNAUTHORIZED seed change (--seed 1 was used without approval; the
Factorial baseline this experiment's A0_hard_choice arm must reproduce,
`set_onpolicy_cosine`, uses seed=0). This is a same-purpose, corrected
re-run, not a new experiment -- same 4 arms, same t=0-hard-CE / t>=1-arm-
dispatch loss protocol, same on-policy-only scope, seed=0 throughout.

Two structural fixes beyond the seed itself, per this round's spec:

1. RNG separation (`scripts/rng_control01.py`): model-init RNG,
   DataLoader-shuffle RNG, and CUDA RNG are controlled independently
   (`--init_seed`, `--loader_seed`) instead of a single `torch.manual_seed`
   call whose effect on DataLoader shuffling depended on how many random
   draws model construction happened to consume first. Every epoch's
   `batch_start_idx` order is hashed (SHA256) and recorded; all four arms
   are meant to share the identical epoch-by-epoch order, checked by
   `scripts/gate_set_loss_control02_batch_order.py` before Stage-1 launch
   proceeds past A0.

2. No duplicated sequence runner (spec section 7): unlike CONTROL01, this
   trainer does NOT reimplement `run_sequence`. It imports
   `scripts.train_factorial_e2e01.run_sequence` directly and passes a
   `loss_fn` callback (t=0 -> Hard Choice CE always, t>=1 -> the arm's own
   loss) -- `run_sequence`'s only change for this (a `loss_fn=None`
   parameter, `tests/test_factorial_e2e01.py` still green, 23/23) is
   documented in that module's own docstring. Everything else (Oracle
   target computation, on-policy prefix advance, free-running contract)
   is Factorial's own code, unmodified, so A0 here is STRUCTURALLY
   incapable of silently diverging from `set_onpolicy_cosine`'s sequence
   logic the way a hand-copied duplicate could.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.rng_control01 import (batch_order_sha256, make_loader_generator,
                                   set_global_seeds)
from scripts.train_factorial_e2e01 import (HostScorer, arm_score, candidate_weights,
                                           encode_raw, free_running_aggregate_future_mse,
                                           grad_norm, greedy_set_utility, param_displacement,
                                           representation_diagnostics, run_sequence, state_sha,
                                           step_rank_diagnostics)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_multipos_choice01 import ALPHA, multipos_choice_step_loss
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_set_loss_control01 import _shared_teacher_diag
from utils.set_loss_experimental import set_utility_soft_ce_loss, soft_regret_mass_loss

ARMS = ('A0_hard_choice', 'A1_adaptive_multipos', 'A2_srm', 'A3_setutility_softce')
M_TOPM = 10


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


def compute_step_loss(arm, t, u_hat, u_target, valid_now, tau):
    """Byte-identical dispatch to train_set_loss_control01.compute_step_loss
    (t=0 always Hard CE regardless of arm; t>=1 dispatches per arm) --
    reimplemented here (not imported) only because the sibling module's
    version is private to its own file structure; the underlying loss
    functions (`oracle_choice_step_loss`, `multipos_choice_step_loss`,
    `soft_regret_mass_loss`, `set_utility_soft_ce_loss`) are the SAME
    imports, unmodified."""
    if t == 0 or arm == 'A0_hard_choice':
        loss, diag = oracle_choice_step_loss(u_hat, u_target, valid_now, tau)
    elif arm == 'A1_adaptive_multipos':
        loss, diag = multipos_choice_step_loss(u_hat, u_target, valid_now, tau, alpha=ALPHA)
    elif arm == 'A2_srm':
        loss, diag = soft_regret_mass_loss(u_hat, u_target, valid_now, tau, m=M_TOPM)
    elif arm == 'A3_setutility_softce':
        loss, diag = set_utility_soft_ce_loss(u_hat, u_target, valid_now, tau, m=M_TOPM)
    else:
        raise ValueError(arm)
    diag = dict(diag)
    diag.update(_shared_teacher_diag(u_hat, u_target, valid_now, tau))
    diag['is_t0_hard_ce'] = float(t == 0)
    return loss, diag


def make_loss_fn(arm, tau_choice):
    """Closure passed as `run_sequence(..., loss_fn=...)`. Signature must
    match run_sequence's call site: (t, u_hat, u_target, valid_now, tau)."""
    def _fn(t, u_hat, u_target, valid_now, tau):
        return compute_step_loss(arm, t, u_hat, u_target, valid_now, tau)
    return _fn


def _agg_add(agg, aggn, kk, vv):
    if isinstance(vv, float) and vv == vv:
        agg[kk] = agg.get(kk, 0.0) + vv
        aggn[kk] = aggn.get(kk, 0) + 1


def train_epoch(exp, args, host, model, set_conditioner, cli, loader, channels, device):
    model.train(True)
    set_conditioner.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    tot_loss, nb = 0.0, 0
    agg, aggn = {}, {}
    agg_t0, aggn_t0, agg_tge1, aggn_tge1 = {}, {}, {}, {}
    agg_ch, aggn_ch = {}, {}
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0
    batch_starts_this_epoch = []
    loss_fn = make_loss_fn(cli.arm, cli.tau_choice)

    for batch_x, batch_y, batch_start_idx in loader:
        batch_starts_this_epoch.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                                       else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, cli.tau_topk)

            losses, diags, picks, steps = run_sequence(
                z_q, E, cand_mask, set_conditioner, None, w_host, futures, query_future,
                'greedy_set', 'onpolicy', cli.tau_choice, cli.top_k, cli.chunk_size,
                free_running=False, greedy_set_impl='reference', loss_fn=loss_fn)
            ch_loss = sum(losses) / cli.top_k
            batch_loss = batch_loss + ch_loss

            with torch.no_grad():
                fr = free_running_aggregate_future_mse(picks, host_scores, futures, query_future, cli.tau_topk)
                for t, (d, diag) in enumerate(zip(steps, diags)):
                    s = step_rank_diagnostics(d['u_hat'], d['u_target'], d['valid_now'],
                                              d['oracle_idx'], d['model_idx'], futures, query_future)
                    s.update(diag)
                    for kk, vv in s.items():
                        _agg_add(agg, aggn, kk, vv)
                        if t == 0:
                            _agg_add(agg_t0, aggn_t0, kk, vv)
                        else:
                            _agg_add(agg_tge1, aggn_tge1, kk, vv)
                _agg_add(agg, aggn, 'train_prefix_aggregate_future_mse', float(fr.mean()))
                _agg_add(agg_ch, aggn_ch, f'ch{c}_train_prefix_aggregate_future_mse', float(fr.mean()))

        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters()))
        gn_n += 1
        cli.optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1),
           'encoder_grad_norm': enc_gn / max(gn_n, 1), 'score_layer_grad_norm': sc_gn / max(gn_n, 1),
           'batch_order_sha256': batch_order_sha256(batch_starts_this_epoch)}
    out.update({f'train_{k}': v / max(aggn.get(k, 1), 1) for k, v in agg.items()})
    out.update({f'train_t0_{k}': v / max(aggn_t0.get(k, 1), 1) for k, v in agg_t0.items()})
    out.update({f'train_tge1_{k}': v / max(aggn_tge1.get(k, 1), 1) for k, v in agg_tge1.items()})
    out.update({f'train_{k}': v / max(aggn_ch.get(k, 1), 1) for k, v in agg_ch.items()})
    return out


@torch.no_grad()
def eval_epoch(exp, args, host, model, set_conditioner, cli, loader, channels, device,
              collect_stepwise=False):
    model.train(False)
    set_conditioner.train(False)
    fr_sum, fr_n = 0.0, 0
    fr_ch_sum, fr_ch_n = {c: 0.0 for c in channels}, {c: 0 for c in channels}
    step_sums, step_n = {}, {}
    stepwise_rows = []
    all_picks_by_channel = {c: [] for c in channels}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            host_scores = host.scores(batch_x, c, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, cli.tau_topk)

            _, _, picks, _ = run_sequence(
                z_q, E, cand_mask, set_conditioner, None, w_host, futures, query_future,
                'greedy_set', 'onpolicy', cli.tau_choice, cli.top_k, cli.chunk_size,
                free_running=True, greedy_set_impl='reference')
            all_picks_by_channel[c].append(picks.cpu())
            fr = free_running_aggregate_future_mse(picks, host_scores, futures, query_future, cli.tau_topk)
            fr_sum += float(fr.sum()); fr_n += int(fr.numel())
            fr_ch_sum[c] += float(fr.sum()); fr_ch_n[c] += int(fr.numel())

            selected = torch.zeros_like(cand_mask)
            for t in range(cli.top_k):
                valid_now = cand_mask & ~selected
                if t == 0:
                    h_t = z_q
                else:
                    m_ = E[picks[:, :t]].mean(dim=1)
                    h_t = set_conditioner(z_q, m_)
                u_hat = arm_score(h_t, E, None)
                u_target = greedy_set_utility(picks[:, :t], w_host, futures, query_future, cli.chunk_size)
                neg_inf = torch.finfo(u_hat.dtype).min / 4
                oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                model_idx = picks[:, t]
                s = step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx, model_idx, futures, query_future)
                _, diag = compute_step_loss(cli.arm, t, u_hat, u_target, valid_now, cli.tau_choice)
                s.update(diag)
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        key = f't{t}::{kk}'
                        step_sums[key] = step_sums.get(key, 0.0) + vv
                        step_n[key] = step_n.get(key, 0) + 1
                selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    out = {'free_running_aggregate_future_mse': fr_sum / max(fr_n, 1)}
    for c in channels:
        out[f'ch{c}_free_running_aggregate_future_mse'] = fr_ch_sum[c] / max(fr_ch_n[c], 1)
    per_step = {kk: step_sums[kk] / max(step_n[kk], 1) for kk in step_sums}
    if collect_stepwise:
        for t in range(cli.top_k):
            row = {'step': t}
            for kk, vv in per_step.items():
                if kk.startswith(f't{t}::'):
                    row[kk.split('::', 1)[1]] = vv
            stepwise_rows.append(row)
    for name in ('oracle_action_acc', 'expert_rank_median', 'expert_regret_mean',
                 'expert_containment_at_10', 'ndcg_at_10', 'spearman',
                 'chosen_raw_regret', 'chosen_normalized_regret', 'support_size',
                 'teacher_entropy', 'normalized_teacher_entropy', 'effective_positive_count',
                 'teacher_ess', 'student_entropy', 'topm_probability_mass'):
        all_v = [per_step[k] for k in per_step if k.endswith('::' + name)]
        t0_v = [per_step[k] for k in per_step if k.startswith('t0::' + name)]
        tge1_v = [per_step[k] for k in per_step if k.endswith('::' + name) and not k.startswith('t0::')]
        out[name] = sum(all_v) / max(len(all_v), 1) if all_v else float('nan')
        out[f'{name}_t0'] = sum(t0_v) / max(len(t0_v), 1) if t0_v else float('nan')
        out[f'{name}_tge1'] = sum(tge1_v) / max(len(tge1_v), 1) if tge1_v else float('nan')

    picks_concat = {c: torch.cat(v, dim=0) for c, v in all_picks_by_channel.items()}
    return out, stepwise_rows, picks_concat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_set_loss_control02')
    ap.add_argument('--out_dir', default='results/TRACK-A-SET-LOSS-CONTROL02')
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
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_ck = torch.load(cli.stage2_host, map_location='cpu')
    host_args = dict(host_ck['args'])
    cli.tau_choice = 0.1
    cli.tau_topk = float(host_args['tau_topk'])

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
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
    param_count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in set_conditioner.parameters())

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(set_conditioner.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:256]

    fingerprint = {
        'exp': 'TRACK-A-SET-LOSS-CONTROL02', 'cell': cli.cell, 'arm': cli.arm,
        'axis_oracle': 'greedy_set', 'axis_prefix': 'onpolicy', 'axis_score': 'cosine',
        'loss_name': cli.arm, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'channels': channels,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'stage2_host_sha256': stage2_host_sha256, 'param_count': param_count,
        'top_k': cli.top_k, 'tau_choice': cli.tau_choice, 'tau_topk': cli.tau_topk,
        'learning_rate': cli.learning_rate, 'batch_size': cli.batch_size,
        'epochs': cli.train_epochs, 'patience': cli.patience, 'weight_decay': cli.weight_decay,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse (all channels)',
        'optimizer': 'Adam', 'reference_ckpt': cli.reference_ckpt, 'stage2_host': cli.stage2_host,
        'alpha_multipos': ALPHA if cli.arm == 'A1_adaptive_multipos' else None,
        'm_topm': M_TOPM, 'code_commit': code_commit,
        'reused_run_sequence': True,
        'run_sequence_source': 'scripts.train_factorial_e2e01.run_sequence (loss_fn hook)',
    }
    (cell_dir / f'config_fingerprint_{cli.arm}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[set_loss_control02] {cli.cell}/{cli.arm} encoder_init_sha={encoder_init_sha256[:16]} '
         f'sc_init_sha={set_conditioner_init_sha256[:16]} channels={channels} '
         f'init_seed={cli.init_seed} loader_seed={cli.loader_seed}')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    stepwise_best = []
    batch_order_hashes = {}
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, cli, train_loader, channels, device)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va, _, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, val_loader, channels, device)

        with torch.no_grad():
            rep = representation_diagnostics(encode_raw(model, probe_x, channels[0]))
        rep['encoder_param_displacement'] = param_displacement(model, init_state)

        row = {'epoch': epoch, **tr,
               'val_free_running_aggregate_future_mse': va['free_running_aggregate_future_mse'],
               **{f'val_{k}': v for k, v in va.items() if k != 'free_running_aggregate_future_mse'},
               **{f'rep_{k}': v for k, v in rep.items() if not isinstance(v, list)}}
        if epoch in test_at:
            te, _, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader, channels, device)
            row['test_free_running_aggregate_future_mse'] = te['free_running_aggregate_future_mse']

        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(),
                   'set_conditioner_state_dict': set_conditioner.state_dict(),
                   'args': vars(args), 'epoch': epoch, 'fingerprint': fingerprint,
                   'val_free_running_aggregate_future_mse': row['val_free_running_aggregate_future_mse']}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if row['val_free_running_aggregate_future_mse'] < best['val']:
            best = {'val': row['val_free_running_aggregate_future_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')

        print(f"[set_loss_control02] {cli.arm} epoch {epoch} train_loss={tr['train_loss']:.5f} "
             f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
             f"chosen_regret_tge1={row.get('val_chosen_normalized_regret_tge1', float('nan')):.4f} "
             f"enc_gn={tr['encoder_grad_norm']:.5f} "
             f"batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[set_loss_control02] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te, stepwise_best, test_picks = eval_epoch(exp, args, host, model, set_conditioner, cli,
                                               test_loader, channels, device, collect_stepwise=True)

    torch.save({c: test_picks[c] for c in channels},
              cell_dir / f'trainer_free_running_test_picks_{cli.arm}.pt')

    with open(cell_dir / f'epoch_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    with open(cell_dir / f'stepwise_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in stepwise_best for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in stepwise_best:
            w.writerow(r)
    (cell_dir / f'batch_order_hashes_{cli.arm}.json').write_text(json.dumps(batch_order_hashes, indent=2))

    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None)
                    for e in sorted(test_at)}
    summary = {
        'exp': 'TRACK-A-SET-LOSS-CONTROL02', 'cell': cli.cell, 'arm': cli.arm,
        'best_epoch': best['epoch'], 'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'], 'matched_epoch_test': matched_test,
        'test_internal': te, 'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[set_loss_control02] done. {cli.cell}/{cli.arm} best_epoch={best['epoch']} "
         f"best_val_fr_agg={best['val']:.6f} test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
