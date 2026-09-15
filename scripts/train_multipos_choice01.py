#!/usr/bin/env python3
"""TRACK-A-MULTIPOS-CHOICE01 -- does Multi-positive Choice CE beat one-hot
Choice CE for the Greedy Set Oracle, under clean on-policy training?

Fixed for every arm: Oracle=greedy_set, Prefix=onpolicy, Score=cosine,
scratch trainable encoder (ONE shared init per cell, SHA-verified),
optimizer/lr/batch/epochs/patience/checkpoint-criterion identical to
TRACK-A-FACTORIAL-E2E01. Only the SUPERVISION changes:

  A0_choice_ce  -- one-hot Choice CE at every step t=1..K (identical
                   protocol to the factorial's own set_onpolicy_cosine arm;
                   retrained here, in this experiment's own output tree, as
                   a reproducibility check against that existing result).
  A1_multipos   -- t=1: one-hot Choice CE (spec S5: t=1 is a difficulty
                   TRACK-A-SET-DIFFICULTY01 found common to both Oracles,
                   not Set-specific, so it is not the target of this
                   experiment). t>=2: Adaptive Multi-positive Choice CE:
                     threshold = u_1 - ALPHA * (u_1 - u_10)
                     P = {i valid, unselected : u_i >= threshold}
                     L = -log( sum_{i in P} p(i) )
                       = -( logsumexp(logits[P]) - logsumexp(logits[valid]) )
                   computed via a masked double-logsumexp, numerically
                   stable, no explicit softmax normalisation needed.

ALPHA deviates from the spec's suggested 0.10
----------------------------------------------
`scripts/offline_check_multipos_alpha01.py`, run BEFORE any training on the
REAL Set Oracle utility distribution (TRACK-A-FACTORIAL-E2E01's own
`set_onpolicy_cosine` checkpoints, both horizons, 20 test batches, channel
0): at alpha=0.10 the median positive count was **exactly 1** at every
step t=2..10, at BOTH horizons -- the spec's own explicit exception clause
("사실상 항상 1개인 경우 ... 조정 가능") applies. A short PRE-TRAINING sweep
(alpha in {0.10, 0.20, 0.30, 0.40, 0.50, 0.60}) found:

    alpha=0.10 -> median=1 (both horizons, every step)
    alpha=0.30 -> median=2
    alpha=0.40 -> median=2, mean 2.3-2.6
    alpha=0.50 -> median=3, mean ~3.0, p90<=5, max<=9  (both horizons agree)
    alpha=0.60 -> median=4, mean ~3.6-4.1

alpha=0.50 was FIXED before any training as the value landing closest to
the centre of the spec's target band (median 2-5), consistently across
both horizons. Not re-tuned after seeing any training or Stage-2 result.
"""
import argparse
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

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (HostScorer, arm_score, encode_raw,
                                           free_running_aggregate_future_mse,
                                           greedy_set_utility, param_displacement,
                                           representation_diagnostics, state_sha,
                                           step_rank_diagnostics)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss

ALPHA = 0.50   # fixed, see module docstring
ARMS = ('A0_choice_ce', 'A1_multipos', 'A2_top3')


def multipos_choice_step_loss(u_hat, u_target, valid_now, tau, alpha=ALPHA):
    """L = -log(sum_{i in P} p(i)), P = {u_i >= u_1 - alpha*(u_1-u_10)}.
    Top-1 Oracle candidate is ALWAYS in P by construction (u_1 itself always
    satisfies u_i >= u_1 - alpha*range for alpha>=0); still force-included
    explicitly as a defensive assertion, not just an assumption."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)
    n_valid = valid_now.sum(dim=-1)
    k_eff = int(n_valid.min().clamp_min(1).clamp_max(10))
    top_vals, top_idx = masked_target.topk(k_eff, dim=-1)
    u1 = top_vals[:, 0]
    u10 = top_vals[:, -1]
    rng = (u1 - u10).clamp_min(1e-12)
    threshold = u1 - alpha * rng
    positive = (masked_target >= threshold.unsqueeze(-1)) & valid_now
    # defensive: force the argmax (oracle top-1) into the positive set
    oracle_idx = masked_target.argmax(dim=-1)
    positive = positive.scatter(1, oracle_idx.unsqueeze(-1), True)

    logits = u_hat.masked_fill(~valid_now, neg_inf) / float(tau)
    row_has_valid = valid_now.any(dim=-1)
    logZ_all = torch.logsumexp(logits, dim=-1)
    logits_pos = logits.masked_fill(~positive, neg_inf)
    logZ_pos = torch.logsumexp(logits_pos, dim=-1)
    nll = (logZ_all - logZ_pos)[row_has_valid]
    loss = nll.mean() if nll.numel() > 0 else u_hat.sum() * 0.0

    with torch.no_grad():
        pos_count = positive.float().sum(dim=-1)
        top1_in_positive = float(positive.gather(1, oracle_idx.unsqueeze(-1)).float().mean())
    diag = {
        'multipos_loss': float(loss.detach()),
        'positive_count_mean': float(pos_count.mean()),
        'positive_count_median': float(pos_count.median()),
        'positive_count_max': float(pos_count.max()),
        'top1_in_positive_rate': top1_in_positive,
        'gap_1_2': float((top_vals[:, 0] - top_vals[:, min(1, k_eff - 1)]).mean()),
        'gap_1_10': float(rng.mean()),
    }
    return loss, diag


def top3_choice_step_loss(u_hat, u_target, valid_now, tau):
    """A2 (optional): fixed top-3 Oracle candidates as positives, every
    step t>=2 (t=1 stays one-hot, same convention as A1 for comparability)."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)
    n_valid = valid_now.sum(dim=-1)
    k_eff = int(n_valid.min().clamp_min(1).clamp_max(3))
    _, top_idx = masked_target.topk(k_eff, dim=-1)
    positive = torch.zeros_like(valid_now)
    positive = positive.scatter(1, top_idx, True) & valid_now

    logits = u_hat.masked_fill(~valid_now, neg_inf) / float(tau)
    row_has_valid = valid_now.any(dim=-1)
    logZ_all = torch.logsumexp(logits, dim=-1)
    logZ_pos = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=-1)
    nll = (logZ_all - logZ_pos)[row_has_valid]
    loss = nll.mean() if nll.numel() > 0 else u_hat.sum() * 0.0
    with torch.no_grad():
        pos_count = positive.float().sum(dim=-1)
    return loss, {'positive_count_mean': float(pos_count.mean()),
                 'positive_count_median': float(pos_count.median())}


def run_sequence(z_q, E, cand_mask, set_conditioner, arm, w_host, futures, query_future,
                 tau_choice, k, free_running=False):
    """On-policy throughout (state = model's own prefix), Oracle=greedy_set
    always. Supervision rule depends on `arm` and step `t`:
      A0_choice_ce : one-hot at every t
      A1_multipos  : one-hot at t=1, multi-positive (alpha=ALPHA) at t>=2
      A2_top3      : one-hot at t=1, fixed-top-3 at t>=2
    free_running=True: selection only (argmax of u_hat), no loss computed,
    no oracle information read at all -- used for validation/test/inference.
    """
    bsz, device = z_q.size(0), z_q.device
    selected = torch.zeros_like(cand_mask)
    picks = []
    losses, diags, step_records = [], [], []
    neg_inf = torch.finfo(z_q.dtype).min / 4

    for t in range(1, k + 1):
        if t == 1:
            h_t = z_q
        else:
            m = E[torch.stack(picks, dim=1)].mean(dim=1)
            h_t = set_conditioner(z_q, m)
        u_hat = arm_score(h_t, E, None)
        valid_now = cand_mask & ~selected

        if free_running:
            nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
            picks.append(nxt.squeeze(-1))
            selected = selected.scatter(1, nxt, True)
            continue

        prefix = torch.stack(picks, dim=1) if picks else torch.zeros(
            bsz, 0, dtype=torch.long, device=device)
        with torch.no_grad():
            u_target = greedy_set_utility(prefix, w_host, futures, query_future, None)

        if t == 1 or arm == 'A0_choice_ce':
            loss_t, diag_t = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        elif arm == 'A1_multipos':
            loss_t, diag_t = multipos_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        elif arm == 'A2_top3':
            loss_t, diag_t = top3_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        else:
            raise ValueError(arm)
        losses.append(loss_t)
        diags.append(diag_t)

        nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
        with torch.no_grad():
            oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
            step_records.append({'t': t, 'u_hat': u_hat, 'u_target': u_target,
                                 'valid_now': valid_now, 'oracle_idx': oracle_idx,
                                 'model_idx': nxt.squeeze(-1)})
        picks.append(nxt.squeeze(-1))
        selected = selected.scatter(1, nxt, True)

    return losses, diags, torch.stack(picks, dim=1), step_records


def _iter_batches(loader, limit):
    """SMOKE ONLY: cap the number of batches. limit<=0 -> full loader."""
    for i, b in enumerate(loader):
        if limit and i >= limit:
            break
        yield b


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def train_epoch(exp, args, host, model, set_conditioner, optimizer, cli, loader, channels, device):
    model.train(True)
    set_conditioner.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    tot_loss, nb = 0.0, 0
    agg, aggn = {}, 0
    mp_agg, mp_n = {}, {}
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                from utils.dense_utility import candidate_weights
                w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)

            losses, diags, picks, steps = run_sequence(
                z_q, E, cand_mask, set_conditioner, cli.arm, w_host, futures, query_future,
                cli.tau_choice, cli.top_k)
            batch_loss = batch_loss + sum(losses) / cli.top_k

            with torch.no_grad():
                fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                       query_future, cli.tau_topk)
                d0 = steps[0]
                s = step_rank_diagnostics(d0['u_hat'], d0['u_target'], d0['valid_now'],
                                          d0['oracle_idx'], d0['model_idx'], futures, query_future)
                s['train_prefix_aggregate_future_mse'] = float(fr.mean())
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        agg[kk] = agg.get(kk, 0.0) + vv
                aggn += 1
                for d in diags:
                    for kk, vv in d.items():
                        if isinstance(vv, float) and vv == vv:
                            mp_agg[kk] = mp_agg.get(kk, 0.0) + vv
                            mp_n[kk] = mp_n.get(kk, 0) + 1

        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters()))
        gn_n += 1
        optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1), 'encoder_grad_norm': enc_gn / max(gn_n, 1),
          'score_layer_grad_norm': sc_gn / max(gn_n, 1)}
    out.update({f'train_{k}': v / max(aggn, 1) for k, v in agg.items()})
    out.update({f'train_{k}': mp_agg.get(k, 0.0) / max(mp_n.get(k, 1), 1)
               for k in mp_agg})
    return out


@torch.no_grad()
def eval_epoch(exp, args, host, model, set_conditioner, cli, loader, channels, device,
              collect_stepwise=False):
    model.train(False)
    set_conditioner.train(False)
    fr_sum, fr_n = 0.0, 0
    step_sums, step_n = {}, {}
    stepwise_rows = []

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            host_scores = host.scores(batch_x, c, cand_mask)
            from utils.dense_utility import candidate_weights
            w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)

            _, _, picks, _ = run_sequence(z_q, E, cand_mask, set_conditioner, cli.arm,
                                          w_host, futures, query_future, cli.tau_choice,
                                          cli.top_k, free_running=True)
            fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                   query_future, cli.tau_topk)
            fr_sum += float(fr.sum())
            fr_n += int(fr.numel())

            selected = torch.zeros_like(cand_mask)
            for t in range(1, cli.top_k + 1):
                valid_now = cand_mask & ~selected
                if t == 1:
                    h_t = z_q
                else:
                    m = E[picks[:, :t - 1]].mean(dim=1)
                    h_t = set_conditioner(z_q, m)
                u_hat = arm_score(h_t, E, None)
                prefix_now = picks[:, :t - 1]
                u_target = greedy_set_utility(prefix_now, w_host, futures, query_future, None)
                neg_inf = torch.finfo(u_hat.dtype).min / 4
                oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                model_idx = picks[:, t - 1]
                s = step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx, model_idx,
                                          futures, query_future)
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        key = f't{t}::{kk}'
                        step_sums[key] = step_sums.get(key, 0.0) + vv
                        step_n[key] = step_n.get(key, 0) + 1
                selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    out = {'free_running_aggregate_future_mse': fr_sum / max(fr_n, 1)}
    per_step = {kk: step_sums[kk] / max(step_n[kk], 1) for kk in step_sums}
    if collect_stepwise:
        for t in range(1, cli.top_k + 1):
            row = {'t': t}
            for kk, vv in per_step.items():
                if kk.startswith(f't{t}::'):
                    row[kk.split('::', 1)[1]] = vv
            stepwise_rows.append(row)
    for name in ('oracle_action_acc', 'expert_rank_median', 'expert_regret_mean',
                 'expert_containment_at_10', 'ndcg_at_10', 'spearman'):
        vals = [per_step[k] for k in per_step if k.endswith('::' + name)]
        out[name] = sum(vals) / max(len(vals), 1) if vals else float('nan')
    return out, stepwise_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_multipos_choice01')
    ap.add_argument('--out_dir', default='results/TRACK-A-MULTIPOS-CHOICE01')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--train_epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--test_epochs', default='1,5,10')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    hp_source = {}

    def resolve(name, ref_key, default, note):
        v = getattr(cli, name)
        if v is not None:
            hp_source[name] = {'value': v, 'source': 'CLI override'}
        elif ref_key in ref_args and ref_args[ref_key] is not None:
            setattr(cli, name, ref_args[ref_key])
            hp_source[name] = {'value': ref_args[ref_key], 'source': f'Stage-1 reference args["{ref_key}"]'}
        else:
            setattr(cli, name, default)
            hp_source[name] = {'value': default, 'source': note}
        return getattr(cli, name)

    resolve('learning_rate', 'learning_rate', 1e-3, 'Stage-1 default')
    resolve('batch_size', 'batch_size', 32, 'Stage-1 default')
    resolve('train_epochs', 'train_epochs', 10, 'Stage-1 default')
    resolve('patience', 'patience', 5, 'Stage-1 default')
    resolve('seed', 'seed', 0, 'Stage-1 default')
    resolve('top_k', 'top_k', 10, 'Stage-1 default')
    resolve('weight_decay', 'weight_decay', 0.0, 'Stage-1 Adam default (0.0)')
    cli.tau_choice = float(ref_args.get('tau_topk', 0.1))
    hp_source['tau_choice'] = {'value': cli.tau_choice, 'source': 'Stage-1 reference args["tau_topk"]'}
    hp_source['alpha'] = {'value': ALPHA, 'source': 'pre-registered via scripts/offline_check_multipos_alpha01.py '
                          '(median positive count = 3 at both horizons; spec default 0.10 gave median=1, '
                          'explicit exception clause invoked)'}
    hp_source['oracle'] = {'value': 'greedy_set', 'source': 'fixed by design'}
    hp_source['prefix'] = {'value': 'onpolicy', 'source': 'fixed by design'}
    hp_source['score'] = {'value': 'cosine', 'source': 'fixed by design'}

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    if int(host.top_k) != int(cli.top_k):
        raise SystemExit(f'[ABORT] host top_k={host.top_k} != Stage-1 top_k={cli.top_k}')
    cli.tau_topk = host.tau_topk  # host's own aggregation temperature (exogenous, fixed)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.seed,
        'top_k': cli.top_k, 'tau_topk': cli.tau_choice,
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
    for p in model.parameters():
        p.requires_grad_(True)

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    sc_init_sha = state_sha(set_conditioner.state_dict())
    if cli.shared_init_in:
        sc_path = str(Path(cli.shared_init_in).with_name(Path(cli.shared_init_in).stem + '_setcond.pth'))
        blob = torch.load(sc_path, map_location='cpu')
        set_conditioner.load_state_dict(blob['state_dict'])
        sc_init_sha = state_sha(set_conditioner.state_dict())
        if sc_init_sha != blob['sha256']:
            raise SystemExit('[ABORT] SetConditioner shared-init SHA mismatch')
    elif cli.shared_init_out:
        sc_out = str(Path(cli.shared_init_out).with_name(Path(cli.shared_init_out).stem + '_setcond.pth'))
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in set_conditioner.state_dict().items()},
                   'sha256': sc_init_sha}, sc_out)

    params = list(model.parameters()) + list(set_conditioner.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)
    channels = list(range(int(args.enc_in)))
    _, train_loader = exp._get_data(flag='train')
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:256]

    fingerprint = {
        'cell': cli.cell, 'arm': cli.arm,
        'encoder_init_sha256': init_sha, 'set_conditioner_init_sha256': sc_init_sha,
        'stage2_host': cli.stage2_host, 'reference_ckpt': cli.reference_ckpt,
        'top_k': cli.top_k, 'tau_choice': cli.tau_choice, 'alpha': ALPHA,
        'hyperparameter_provenance': hp_source,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse',
        'supervision_rule': ('one-hot Choice CE at every step' if cli.arm == 'A0_choice_ce' else
                             't=1 one-hot, t>=2 adaptive multi-positive (alpha=0.5)' if cli.arm == 'A1_multipos' else
                             't=1 one-hot, t>=2 fixed top-3'),
    }
    (cell_dir / f'config_fingerprint_{cli.arm}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[multipos_choice01] {cli.cell}/{cli.arm} init_sha={init_sha[:16]} sc_sha={sc_init_sha[:16]}')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, optimizer, cli,
                         train_loader, channels, device)
        va, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, val_loader, channels, device)
        with torch.no_grad():
            rep = representation_diagnostics(encode_raw(model, probe_x, 0))
        rep['encoder_param_displacement'] = param_displacement(model, init_state)

        row = {'epoch': epoch, **tr,
              'val_free_running_aggregate_future_mse': va['free_running_aggregate_future_mse'],
              **{f'val_{k}': v for k, v in va.items() if k != 'free_running_aggregate_future_mse'},
              **{f'rep_{k}': v for k, v in rep.items() if not isinstance(v, list)}}
        if epoch in test_at:
            te, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader, channels, device)
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

        pc = row.get('train_positive_count_median', float('nan'))
        print(f"[multipos_choice01] {cli.arm} epoch {epoch} train_loss={tr['train_loss']:.5f} "
              f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
              f"eff_rank={rep['effective_rank']:.3f} pair_cos={rep['mean_pairwise_cosine']:.4f} "
              f"pos_count_med={pc}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[multipos_choice01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te, stepwise_best = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader,
                                   channels, device, collect_stepwise=True)

    with open(cell_dir / f'epoch_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    with open(cell_dir / f'stepwise_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in stepwise_best for k in r})
        w = csv.DictWriter(fh, fieldnames=['t'] + [k for k in keys if k != 't'])
        w.writeheader()
        for r in stepwise_best:
            w.writerow(r)

    matched_val = {f'epoch{e}': next((r.get('val_free_running_aggregate_future_mse')
                                      for r in epoch_rows if r['epoch'] == e), None) for e in sorted(test_at)}
    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None) for e in sorted(test_at)}
    summary = {
        'cell': cli.cell, 'arm': cli.arm, 'best_epoch': best['epoch'],
        'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'],
        'matched_epoch_val': matched_val, 'matched_epoch_test': matched_test,
        'test_internal': dict(te),
        'encoder_init_sha256': init_sha, 'set_conditioner_init_sha256': sc_init_sha,
        'final_effective_rank': epoch_rows[-1].get('rep_effective_rank'),
        'final_mean_pairwise_cosine': epoch_rows[-1].get('rep_mean_pairwise_cosine'),
        'wall_clock_seconds': time.time() - t0,
        'checkpoint': str(ckpt_dir / 'checkpoint.pth'), 'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    print(f"[multipos_choice01] done. {cli.cell}/{cli.arm} best_epoch={best['epoch']} "
          f"test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
