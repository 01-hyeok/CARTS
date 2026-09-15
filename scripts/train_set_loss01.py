#!/usr/bin/env python3
"""EXP-SET-LOSS01 -- Greedy Set Oracle, 1-term loss comparison: Hard Choice
CE (existing baseline, unmodified) vs Soft Regret Mass (SRM) vs Set-Utility
Soft Cross-Entropy (SoftCE). Set Oracle only (no Individual), Cosine score
only (no asymmetric), reference Set-Oracle utility only (no OPT02/OPT03
algebraic optimization), ETTh1 only (no Weather).

Reuses `scripts/train_factorial_e2e01.py`'s own functions UNCHANGED
(`encode_raw`, `greedy_set_utility`, `arm_score`, `HostScorer`,
`free_running_aggregate_future_mse`, `representation_diagnostics`,
`param_displacement`, `grad_norm`, `step_rank_diagnostics`, `state_sha`,
`candidate_weights`) -- this file does not modify
`scripts/train_factorial_e2e01.py` in any way (zero diff); it only
imports from it. The new losses live in `utils/set_loss_experimental.py`
(also new, also does not touch any reference implementation).

All Stage-1 metrics are reported for t=0, t>=1, and overall separately
(t=0 has an empty prefix and is Individual-like; the Set-conditioning
effect only shows at t>=1).
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (HostScorer, arm_score, candidate_weights,
                                           encode_raw, free_running_aggregate_future_mse,
                                           grad_norm, greedy_set_utility, param_displacement,
                                           representation_diagnostics, state_sha,
                                           step_rank_diagnostics)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.set_loss_experimental import (compute_oracle_teacher_signal,
                                         set_utility_soft_ce_loss,
                                         soft_regret_mass_loss)

LOSS_TYPES = ('hard_ce', 'srm', 'setutility_softce')
M_TOPM = 10


def _iter_batches(loader, limit):
    for i, b in enumerate(loader):
        if limit and i >= limit:
            break
        yield b


def compute_step_loss(loss_type, u_hat, u_target, valid_now, tau_choice):
    """Returns (loss, diag) where diag ALWAYS includes the shared teacher/
    ranking-adjacent diagnostics (chosen_regret, effective_positives,
    teacher_entropy, topm_probability_mass), regardless of which loss was
    actually optimized -- so all three arms report the SAME diagnostic
    fields for a fair Stage-1 comparison."""
    teacher = compute_oracle_teacher_signal(u_target, valid_now, m=M_TOPM)
    if loss_type == 'hard_ce':
        loss, diag = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
    elif loss_type == 'srm':
        loss, diag = soft_regret_mass_loss(u_hat, u_target, valid_now, tau_choice,
                                           m=M_TOPM, teacher=teacher)
    elif loss_type == 'setutility_softce':
        loss, diag = set_utility_soft_ce_loss(u_hat, u_target, valid_now, tau_choice,
                                              m=M_TOPM, teacher=teacher)
    else:
        raise ValueError(loss_type)

    with torch.no_grad():
        model_idx = u_hat.masked_fill(~valid_now, torch.finfo(u_hat.dtype).min / 4).argmax(dim=-1)
        rows = teacher['row_has_valid']
        chosen_regret = teacher['regret_norm'].gather(1, model_idx.unsqueeze(-1)).squeeze(-1)
        chosen_regret = chosen_regret.clamp(max=1e6)  # invalid/degenerate rows can be +inf-ish
        eff_pos = teacher['p_mask'].float().sum(dim=-1)
        w = teacher['weight']
        w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pi = w / w_sum
        t_ent = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(-1)
        diag = dict(diag)
        diag['chosen_normalized_regret'] = float(chosen_regret[rows].mean()) if bool(rows.any()) else float('nan')
        diag['effective_positive_count'] = float(eff_pos[rows].mean()) if bool(rows.any()) else float('nan')
        diag['teacher_entropy'] = float(t_ent[rows].mean()) if bool(rows.any()) else float('nan')
    return loss, diag


def run_sequence(z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                 prefix_policy, loss_type, tau_choice, k, chunk_size, free_running=False):
    """Set-Oracle-only variant of train_factorial_e2e01.run_sequence, with a
    selectable loss_type. Everything else (state construction, masking,
    prefix-advance rule, argmax convention) is IDENTICAL to that file's own
    run_sequence -- copied, not reinvented, so TF/on-policy semantics match
    exactly."""
    bsz = z_q.size(0)
    device = z_q.device
    selected = torch.zeros_like(cand_mask)
    picks = []
    losses, diags, step_records = [], [], []
    neg_inf = torch.finfo(z_q.dtype).min / 4

    for t in range(k):
        if t == 0:
            h_t = z_q
        else:
            prefix = torch.stack(picks, dim=1).detach()
            m = E[prefix].mean(dim=1)
            h_t = set_conditioner(z_q, m)
        u_hat = arm_score(h_t, E, None)
        valid_now = cand_mask & ~selected

        prefix_now = (torch.stack(picks, dim=1).detach() if picks
                      else torch.zeros(bsz, 0, dtype=torch.long, device=device))

        if free_running:
            nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
            picks.append(nxt.squeeze(-1))
            selected = selected.scatter(1, nxt, True)
            continue

        with torch.no_grad():
            u_target = greedy_set_utility(prefix_now, w_host, futures, query_future, chunk_size)

        loss_t, diag_t = compute_step_loss(loss_type, u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)

        oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True)
        model_next = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
        nxt = oracle_next if prefix_policy == 'tf' else model_next

        with torch.no_grad():
            step_records.append({
                'step': t, 'oracle_idx': oracle_next.squeeze(-1), 'model_idx': model_next.squeeze(-1),
                'u_target': u_target, 'u_hat': u_hat.detach(), 'valid_now': valid_now,
            })

        picks.append(nxt.squeeze(-1).detach())
        selected = selected.scatter(1, nxt, True)

    return losses, diags, torch.stack(picks, dim=1), step_records


def _agg_add(agg, aggn, kk, vv):
    if isinstance(vv, float) and vv == vv:
        agg[kk] = agg.get(kk, 0.0) + vv
        aggn[kk] = aggn.get(kk, 0) + 1


def train_epoch(exp, args, host, model, set_conditioner, cli, loader, device):
    model.train(True)
    set_conditioner.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    tot_loss, nb = 0.0, 0
    agg, aggn = {}, {}
    agg_t0, aggn_t0 = {}, {}
    agg_tge1, aggn_tge1 = {}, {}
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0
    channel = 0

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        E = encode_raw(model, exp.memory_x, channel)
        z_q = encode_raw(model, batch_x, channel)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        query_future = batch_y[:, :, channel]
        with torch.no_grad():
            host_scores = host.scores(batch_x, channel, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, cli.tau_topk)

        losses, diags, picks, steps = run_sequence(
            z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
            cli.prefix_policy, cli.loss_type, cli.tau_choice, cli.top_k, cli.chunk_size)

        batch_loss = sum(losses) / cli.top_k
        batch_loss.backward()
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters()))
        gn_n += 1
        cli.optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

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

    out = {'train_loss': tot_loss / max(nb, 1),
           'encoder_grad_norm': enc_gn / max(gn_n, 1), 'score_layer_grad_norm': sc_gn / max(gn_n, 1)}
    out.update({f'train_{k}': v / max(aggn.get(k, 1), 1) for k, v in agg.items()})
    out.update({f'train_t0_{k}': v / max(aggn_t0.get(k, 1), 1) for k, v in agg_t0.items()})
    out.update({f'train_tge1_{k}': v / max(aggn_tge1.get(k, 1), 1) for k, v in agg_tge1.items()})
    return out


@torch.no_grad()
def eval_epoch(exp, args, host, model, set_conditioner, cli, loader, device, collect_stepwise=False):
    model.train(False)
    set_conditioner.train(False)
    channel = 0
    fr_sum, fr_n = 0.0, 0
    step_sums, step_n = {}, {}
    stepwise_rows = []

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        E = encode_raw(model, exp.memory_x, channel)
        z_q = encode_raw(model, batch_x, channel)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        query_future = batch_y[:, :, channel]
        host_scores = host.scores(batch_x, channel, cand_mask)
        w_host = candidate_weights(host_scores, cand_mask, cli.tau_topk)

        _, _, picks, _ = run_sequence(
            z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
            cli.prefix_policy, cli.loss_type, cli.tau_choice, cli.top_k, cli.chunk_size,
            free_running=True)
        fr = free_running_aggregate_future_mse(picks, host_scores, futures, query_future, cli.tau_topk)
        fr_sum += float(fr.sum())
        fr_n += int(fr.numel())

        selected = torch.zeros_like(cand_mask)
        for t in range(cli.top_k):
            valid_now = cand_mask & ~selected
            if t == 0:
                h_t = z_q
            else:
                m = E[picks[:, :t]].mean(dim=1)
                h_t = set_conditioner(z_q, m)
            u_hat = arm_score(h_t, E, None)
            u_target = greedy_set_utility(picks[:, :t], w_host, futures, query_future, cli.chunk_size)
            neg_inf = torch.finfo(u_hat.dtype).min / 4
            oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
            model_idx = picks[:, t]
            s = step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx, model_idx, futures, query_future)
            _, diag = compute_step_loss(cli.loss_type, u_hat, u_target, valid_now, cli.tau_choice)
            s.update(diag)
            for kk, vv in s.items():
                if isinstance(vv, float) and vv == vv:
                    key = f't{t}::{kk}'
                    step_sums[key] = step_sums.get(key, 0.0) + vv
                    step_n[key] = step_n.get(key, 0) + 1
            selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    out = {'free_running_aggregate_future_mse': fr_sum / max(fr_n, 1)}
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
                 'chosen_normalized_regret', 'effective_positive_count', 'teacher_entropy'):
        all_vals = [per_step[k] for k in per_step if k.endswith('::' + name)]
        tge1_vals = [per_step[k] for k in per_step if k.endswith('::' + name) and not k.startswith('t0::')]
        t0_vals = [per_step[k] for k in per_step if k.startswith('t0::' + name)]
        out[name] = sum(all_vals) / max(len(all_vals), 1) if all_vals else float('nan')
        out[f'{name}_t0'] = sum(t0_vals) / max(len(t0_vals), 1) if t0_vals else float('nan')
        out[f'{name}_tge1'] = sum(tge1_vals) / max(len(tge1_vals), 1) if tge1_vals else float('nan')
    return out, stepwise_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--loss_type', choices=LOSS_TYPES, required=True)
    ap.add_argument('--prefix_policy', choices=('tf', 'onpolicy'), required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_set_loss01')
    ap.add_argument('--out_dir', default='results/EXP-SET-LOSS01')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--test_epochs', default='1,5,10')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_ck = torch.load(cli.stage2_host, map_location='cpu')
    host_args = dict(host_ck['args'])
    cli.tau_choice = float(ref_args.get('tau_topk', 0.1))
    cli.tau_topk = float(host_args['tau_topk'])

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    if int(host.top_k) != int(cli.top_k):
        raise SystemExit(f'[ABORT] host top_k={host.top_k} != {cli.top_k}')

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.seed, 'top_k': cli.top_k, 'tau_topk': cli.tau_choice,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)

    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['state_dict'])
        got = state_sha(model.state_dict())
        if got != blob['sha256']:
            raise SystemExit(f'[ABORT] shared-init SHA mismatch: {got} != {blob["sha256"]}')
        init_sha = got
    else:
        init_sha = state_sha(model.state_dict())
        if cli.shared_init_out:
            Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'sha256': init_sha, 'cell': cli.cell}, cli.shared_init_out)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(set_conditioner.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    _, train_loader = exp._get_data(flag='train')
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:256]

    fingerprint = {
        'exp': 'EXP-SET-LOSS01', 'cell': cli.cell, 'arm': cli.arm_name,
        'loss_type': cli.loss_type, 'prefix_policy': cli.prefix_policy, 'scorer': 'cosine',
        'oracle': 'greedy_set', 'oracle_utility_impl': 'reference (utils/dense_utility.py, unmodified)',
        'm_topm': M_TOPM,
        'encoder_init_sha256': init_sha, 'seed': cli.seed, 'top_k': cli.top_k,
        'tau_choice': cli.tau_choice, 'tau_topk': cli.tau_topk,
        'learning_rate': cli.learning_rate, 'weight_decay': cli.weight_decay,
        'batch_size': cli.batch_size, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse',
        'reference_ckpt': cli.reference_ckpt, 'stage2_host': cli.stage2_host,
        'limit_batches': cli.limit_batches, 'is_smoke_run': bool(cli.limit_batches),
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[set_loss01] {cli.cell}/{cli.arm_name} loss={cli.loss_type} prefix={cli.prefix_policy} '
         f'init_sha={init_sha[:16]}')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    stepwise_best = []
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, cli, train_loader, device)
        va, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, val_loader, device)
        with torch.no_grad():
            rep = representation_diagnostics(encode_raw(model, probe_x, 0))
        rep['encoder_param_displacement'] = param_displacement(model, init_state)

        row = {'epoch': epoch, **tr,
               'val_free_running_aggregate_future_mse': va['free_running_aggregate_future_mse'],
               **{f'val_{k}': v for k, v in va.items() if k != 'free_running_aggregate_future_mse'},
               **{f'rep_{k}': v for k, v in rep.items() if not isinstance(v, list)}}

        if epoch in test_at:
            te, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader, device)
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

        print(f"[set_loss01] {cli.arm_name} epoch {epoch} train_loss={tr['train_loss']:.5f} "
             f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
             f"chosen_regret_tge1={row.get('val_chosen_normalized_regret_tge1', float('nan')):.4f} "
             f"eff_rank={rep['effective_rank']:.3f} enc_gn={tr['encoder_grad_norm']:.5f}")

        if epoch - best['epoch'] >= cli.patience:
            print(f'[set_loss01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te, stepwise_best = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader, device,
                                   collect_stepwise=True)

    with open(cell_dir / f'epoch_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    with open(cell_dir / f'stepwise_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in stepwise_best for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in stepwise_best:
            w.writerow(r)

    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None)
                    for e in sorted(test_at)}
    summary = {
        'exp': 'EXP-SET-LOSS01', 'cell': cli.cell, 'arm': cli.arm_name,
        'loss_type': cli.loss_type, 'prefix_policy': cli.prefix_policy,
        'best_epoch': best['epoch'], 'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'],
        'matched_epoch_test': matched_test, 'test_internal': te,
        'encoder_init_sha256': init_sha,
        'final_effective_rank': epoch_rows[-1].get('rep_effective_rank'),
        'final_encoder_param_displacement': epoch_rows[-1].get('rep_encoder_param_displacement'),
        'wall_clock_seconds': time.time() - t0,
        'checkpoint': str(ckpt_dir / 'checkpoint.pth'), 'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[set_loss01] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
         f"best_val_fr_agg={best['val']:.6f} test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
