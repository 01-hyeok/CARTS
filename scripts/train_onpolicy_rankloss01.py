#!/usr/bin/env python3
"""TRACK-A-ONPOLICY-RANKLOSS01 -- On-policy prefix fixed, loss varied.

Follow-up to TRACK-A-FACTORIAL-E2E01: that experiment's largest, most
consistent main effect was Prefix (on-policy >> teacher-forcing) at both
ETTh1 horizons, while loss was held fixed at Oracle-Choice CE (R0). This
script fixes Oracle=individual, Prefix=onpolicy, Score=cosine (the
factorial's own best-supported common configuration -- see the pre-run
report) and varies ONLY the training loss:

    R0  Oracle-Choice CE           (train_oracle_choice01.oracle_choice_step_loss, unchanged)
    R1  Weighted Top-K CE (WCE)    (RelationStage1.weighted_topk_listwise_ce, unchanged math,
                                    Oracle Top-K recomputed LIVE from the on-policy state --
                                    the old teacher-cache path is NOT reused, see module note)
    R2  Pairwise utility ranking   (new: softplus hinge over sampled utility-ordered pairs)
    R3  Utility-aware listwise     (new: Top-M-support relevance CE, M is a LOSS support only,
                                    free-running inference stays full-memory direct Top-K)
    R4  Choice-CE + Rank hybrid    (optional, only if compute allows)

Everything else -- dataset, scratch shared encoder init per cell, candidate
mask, K, aggregation (host-fixed), optimizer/lr/wd/batch/epochs/patience,
checkpoint criterion (min val free_running_aggregate_future_mse), Stage-2
evaluation -- is identical to `scripts/train_factorial_e2e01.py` and is
imported from it, not reimplemented.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import weighted_topk_listwise_ce
from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (
    HostScorer, arm_score, encode_raw, free_running_aggregate_future_mse,
    individual_utility, param_displacement, representation_diagnostics,
    run_sequence as run_sequence_choice_ce, state_sha, step_rank_diagnostics)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss

LOSSES = ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise', 'R4_hybrid')


# --------------------------------------------------------------------------
# on-policy sequence, generalized over the loss family
# --------------------------------------------------------------------------
def run_sequence_rankloss(z_q, E, cand_mask, set_conditioner, loss_name,
                          futures, query_future, tau_choice, k, chunk_size,
                          m_listwise, min_gap_pairwise, n_pairs_pairwise,
                          lambda_hybrid, free_running=False):
    """K steps, ON-POLICY throughout (state = model's own prefix, always).
    Supervision is ALWAYS `individual_utility` (future-based Oracle), never
    the model's own choice -- identical discipline to
    `train_factorial_e2e01.run_sequence`'s on-policy branch, generalized to
    the loss families this script adds.

    free_running=True: no Oracle target computed at all (inference/eval).
    """
    bsz, device = z_q.size(0), z_q.device
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
        u_hat = arm_score(h_t, E, None)  # Score fixed = cosine for this experiment
        valid_now = cand_mask & ~selected

        if free_running:
            nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
            picks.append(nxt.squeeze(-1))
            selected = selected.scatter(1, nxt, True)
            continue

        with torch.no_grad():
            u_target = individual_utility(futures, query_future)  # step-invariant Oracle

        loss_t, diag_t = _step_loss(loss_name, u_hat, u_target, valid_now, tau_choice,
                                    m_listwise, min_gap_pairwise, n_pairs_pairwise,
                                    lambda_hybrid)
        losses.append(loss_t)
        diags.append(diag_t)

        # ON-POLICY: the next prefix element is the MODEL's own argmax, NEVER
        # differentiated through, NEVER used as the supervision label above
        # (the label came from u_target, computed before this line).
        nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()

        with torch.no_grad():
            oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True)
            step_records.append({
                'step': t + 1, 'oracle_idx': oracle_next.squeeze(-1),
                'model_idx': nxt.squeeze(-1), 'u_target': u_target,
                'u_hat': u_hat.detach(), 'valid_now': valid_now,
            })

        picks.append(nxt.squeeze(-1))
        selected = selected.scatter(1, nxt, True)

    return losses, diags, torch.stack(picks, dim=1), step_records


def _step_loss(loss_name, u_hat, u_target, valid_now, tau_choice,
               m_listwise, min_gap_pairwise, n_pairs_pairwise, lambda_hybrid):
    if loss_name == 'R0_choice_ce':
        return oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
    if loss_name == 'R1_wce':
        return _wce_step_loss(u_hat, u_target, valid_now, tau_choice)
    if loss_name == 'R2_pairwise':
        return _pairwise_step_loss(u_hat, u_target, valid_now, min_gap_pairwise, n_pairs_pairwise)
    if loss_name == 'R3_listwise':
        return _listwise_step_loss(u_hat, u_target, valid_now, tau_choice, m_listwise)
    if loss_name == 'R4_hybrid':
        l_ce, d_ce = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        l_rank, d_rank = _listwise_step_loss(u_hat, u_target, valid_now, tau_choice, m_listwise)
        loss = l_ce + lambda_hybrid * l_rank
        diag = {**d_ce, **{f'rank_{k}': v for k, v in d_rank.items()}}
        return loss, diag
    raise ValueError(f'unknown loss_name: {loss_name}')


# ---- R1: WCE. Math is `weighted_topk_listwise_ce` UNCHANGED; only the
# Oracle Top-K construction is done here, LIVE, from the current on-policy
# state -- NOT from the project's stale teacher-cache path (audited in the
# pre-run report: `exp_stage1_relation.py::_teacher_batch` reads a
# precomputed, teacher-forced-trajectory cache, which would silently
# reintroduce the exact bias this experiment controls for). ----
def _wce_step_loss(u_hat, u_target, valid_now, tau_teacher, top_k_oracle=10):
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)
    k_eff = min(top_k_oracle, valid_now.sum(dim=-1).min().clamp_min(1).item())
    top_vals, top_idx = masked_target.topk(k_eff, dim=-1)
    oracle_valid = top_vals > (neg_inf / 2)
    oracle_mse = -top_vals  # u = -MSE -> recover MSE distances for the weighting formula
    active_query = valid_now.any(dim=-1)
    log_prob = F.log_softmax(u_hat.masked_fill(~valid_now, neg_inf), dim=-1)
    targets = {'oracle_indices': top_idx, 'oracle_mse': oracle_mse,
              'oracle_valid': oracle_valid, 'active_query': active_query}
    loss, metrics = weighted_topk_listwise_ce(log_prob, targets, tau_teacher)
    return loss, {k: float(v) for k, v in metrics.items()}


# ---- R2: pairwise utility ranking, sampled (no N^2 pairs). ----
def _pairwise_step_loss(u_hat, u_target, valid_now, min_gap, n_pairs, n_top=8, n_hard=8, n_rand=8):
    bsz, n = u_hat.shape
    device = u_hat.device
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)

    total_loss, total_pairs, active_rows = 0.0, 0, 0
    losses_per_row = []
    for b in range(bsz):
        vb = valid_now[b]
        n_valid = int(vb.sum())
        if n_valid < 2:
            continue
        idx = vb.nonzero(as_tuple=True)[0]
        # candidate pools, drawn from the VALID index set only
        top_pool = masked_target[b, idx].topk(min(n_top, n_valid)).indices
        top_idx = idx[top_pool]
        # hard negatives: high student score, but not in the utility top pool
        student_order = u_hat[b, idx].argsort(descending=True)
        hard_idx = idx[student_order[:min(n_hard, n_valid)]]
        rand_perm = torch.randperm(n_valid, device=device)[:min(n_rand, n_valid)]
        rand_idx = idx[rand_perm]
        pool = torch.unique(torch.cat([top_idx, hard_idx, rand_idx]))
        if pool.numel() < 2:
            continue
        u_pool = masked_target[b, pool]
        s_pool = u_hat[b, pool]
        # all pairs within this small pool (pool size <= n_top+n_hard+n_rand, bounded)
        gap = u_pool.unsqueeze(1) - u_pool.unsqueeze(0)          # u_i - u_j
        keep = gap > min_gap
        if not bool(keep.any()):
            continue
        s_diff = s_pool.unsqueeze(1) - s_pool.unsqueeze(0)       # s_i - s_j, want > 0 when kept
        pair_loss = F.softplus(-s_diff)[keep]
        losses_per_row.append(pair_loss.mean())
        total_pairs += int(keep.sum())
        active_rows += 1

    if not losses_per_row:
        zero = u_hat.sum() * 0.0
        return zero, {'pairwise_loss': 0.0, 'n_pairs': 0, 'active_rows': 0}
    loss = torch.stack(losses_per_row).mean()
    return loss, {'pairwise_loss': float(loss.detach()), 'n_pairs': total_pairs,
                  'active_rows': active_rows, 'mean_pairs_per_row': total_pairs / max(active_rows, 1)}


# ---- R3: utility-aware listwise, Top-M SUPPORT for the LOSS only. Never
# touches `run_sequence_rankloss`'s free_running branch, which is always
# full-memory direct Top-K. ----
def _listwise_step_loss(u_hat, u_target, valid_now, tau, m_support):
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)
    n_valid_min = int(valid_now.sum(dim=-1).min())
    m_eff = min(m_support, max(n_valid_min, 1))
    top_vals, top_idx = masked_target.topk(m_eff, dim=-1)
    support_valid = top_vals > (neg_inf / 2)

    teacher = F.softmax(top_vals.masked_fill(~support_valid, neg_inf) / float(tau), dim=-1)
    student_logits = u_hat.gather(1, top_idx)
    student_log_prob = F.log_softmax(
        student_logits.masked_fill(~support_valid, neg_inf), dim=-1)
    active = valid_now.any(dim=-1) & support_valid.any(dim=-1)
    per_row = -(teacher * student_log_prob.masked_fill(~support_valid, 0.0)).sum(dim=-1)
    loss = per_row[active].mean() if bool(active.any()) else u_hat.sum() * 0.0
    with torch.no_grad():
        entropy = -(teacher * (teacher + 1e-8).log()).sum(dim=-1)[active]
    return loss, {'listwise_loss': float(loss.detach()), 'm_support': m_eff,
                  'teacher_entropy_mean': float(entropy.mean()) if active.any() else float('nan')}


# --------------------------------------------------------------------------
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
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]

            losses, diags, picks, steps = run_sequence_rankloss(
                z_q, E, cand_mask, set_conditioner, cli.loss_name,
                futures, query_future, cli.tau_choice, cli.top_k, cli.chunk_size,
                cli.m_listwise, cli.min_gap_pairwise, cli.n_pairs_pairwise,
                cli.lambda_hybrid)

            batch_loss = batch_loss + sum(losses) / cli.top_k
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
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

        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters()))
        gn_n += 1
        optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1),
           'encoder_grad_norm': enc_gn / max(gn_n, 1),
           'score_layer_grad_norm': sc_gn / max(gn_n, 1)}
    out.update({f'train_{k}': v / max(aggn, 1) for k, v in agg.items()})
    return out


@torch.no_grad()
def eval_epoch(exp, args, host, model, set_conditioner, cli, loader, channels, device,
              collect_stepwise=False):
    model.train(False)
    set_conditioner.train(False)
    fr_sum, fr_n = 0.0, 0
    step_sums, step_n = {}, {}
    stepwise_rows = []

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

            _, _, picks, _ = run_sequence_rankloss(
                z_q, E, cand_mask, set_conditioner, cli.loss_name,
                futures, query_future, cli.tau_choice, cli.top_k, cli.chunk_size,
                cli.m_listwise, cli.min_gap_pairwise, cli.n_pairs_pairwise,
                cli.lambda_hybrid, free_running=True)
            fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                   query_future, cli.tau_topk)
            fr_sum += float(fr.sum())
            fr_n += int(fr.numel())

            selected = torch.zeros_like(cand_mask)
            prev_agg = None
            for t in range(cli.top_k):
                valid_now = cand_mask & ~selected
                if t == 0:
                    h_t = z_q
                else:
                    m = E[picks[:, :t]].mean(dim=1)
                    h_t = set_conditioner(z_q, m)
                u_hat = arm_score(h_t, E, None)
                u_target = individual_utility(futures, query_future)
                neg_inf = torch.finfo(u_hat.dtype).min / 4
                oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                model_idx = picks[:, t]
                s = step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx, model_idx,
                                          futures, query_future)
                cur = free_running_aggregate_future_mse(
                    picks[:, :t + 1], host_scores, futures, query_future, cli.tau_topk).mean()
                s['current_aggregate_mse'] = float(cur)
                s['marginal_aggregate_improvement'] = (
                    float(prev_agg - cur) if prev_agg is not None else float('nan'))
                prev_agg = cur
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        key = f't{t + 1}::{kk}'
                        step_sums[key] = step_sums.get(key, 0.0) + vv
                        step_n[key] = step_n.get(key, 0) + 1
                selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    out = {'free_running_aggregate_future_mse': fr_sum / max(fr_n, 1)}
    per_step = {kk: step_sums[kk] / max(step_n[kk], 1) for kk in step_sums}
    if collect_stepwise:
        for t in range(cli.top_k):
            row = {'step': t + 1}
            for kk, vv in per_step.items():
                if kk.startswith(f't{t + 1}::'):
                    row[kk.split('::', 1)[1]] = vv
            stepwise_rows.append(row)
    for name in ('oracle_action_acc', 'expert_rank_median', 'expert_regret_mean',
                 'expert_containment_at_10', 'ndcg_at_10', 'spearman',
                 'selected_individual_future_mse'):
        vals = [per_step[k] for k in per_step if k.endswith('::' + name)]
        out[name] = sum(vals) / max(len(vals), 1) if vals else float('nan')
    t1 = {k.split('::', 1)[1]: v for k, v in per_step.items() if k.startswith('t1::')}
    tk = {k.split('::', 1)[1]: v for k, v in per_step.items() if k.startswith(f't{cli.top_k}::')}
    out['t1_oracle_action_acc'] = t1.get('oracle_action_acc', float('nan'))
    out['t1_ndcg_at_10'] = t1.get('ndcg_at_10', float('nan'))
    out['final_t_oracle_action_acc'] = tk.get('oracle_action_acc', float('nan'))
    out['final_t_expert_regret_mean'] = tk.get('expert_regret_mean', float('nan'))
    return out, stepwise_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--loss_name', choices=LOSSES, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_onpolicy_rankloss01')
    ap.add_argument('--out_dir', default='results/track_a_onpolicy_rankloss01')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--train_epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--probe_size', type=int, default=256)
    ap.add_argument('--test_epochs', default='1,5,10')
    ap.add_argument('--m_listwise', type=int, default=100)
    ap.add_argument('--min_gap_pairwise', type=float, default=1e-4)
    ap.add_argument('--n_pairs_pairwise', type=int, default=0, help='unused, pool-based sampling records actual count')
    ap.add_argument('--lambda_hybrid', type=float, default=0.1)
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_args = dict(torch.load(cli.stage2_host, map_location='cpu')['args'])
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
    hp_source['m_listwise'] = {'value': cli.m_listwise, 'source': 'pre-registered, single value, per spec S4'}
    hp_source['min_gap_pairwise'] = {'value': cli.min_gap_pairwise, 'source': 'pre-registered'}
    hp_source['lambda_hybrid'] = {'value': cli.lambda_hybrid, 'source': 'pre-registered (used only for R4)'}
    hp_source['oracle'] = {'value': 'individual', 'source': 'fixed by design -- see pre-run report'}
    hp_source['prefix'] = {'value': 'onpolicy', 'source': 'fixed by design -- see pre-run report'}
    hp_source['score'] = {'value': 'cosine', 'source': 'fixed by design -- see pre-run report'}

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.loss_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    cli.tau_topk = host.tau_topk
    if int(host.top_k) != int(cli.top_k):
        raise SystemExit(f'[ABORT] host top_k={host.top_k} != Stage-1 top_k={cli.top_k}')

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
        sc_blob_path = str(Path(cli.shared_init_in).with_name(
            Path(cli.shared_init_in).stem + '_setcond.pth'))
        blob = torch.load(sc_blob_path, map_location='cpu')
        set_conditioner.load_state_dict(blob['state_dict'])
        sc_init_sha = state_sha(set_conditioner.state_dict())
        if sc_init_sha != blob['sha256']:
            raise SystemExit('[ABORT] SetConditioner shared-init SHA mismatch')
    elif cli.shared_init_out:
        sc_out = str(Path(cli.shared_init_out).with_name(
            Path(cli.shared_init_out).stem + '_setcond.pth'))
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in set_conditioner.state_dict().items()},
                   'sha256': sc_init_sha}, sc_out)

    params = list(model.parameters()) + list(set_conditioner.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)
    channels = list(range(int(args.enc_in)))
    _, train_loader = exp._get_data(flag='train')
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:cli.probe_size]

    fingerprint = {
        'cell': cli.cell, 'loss_name': cli.loss_name,
        'oracle': 'individual', 'prefix': 'onpolicy', 'score': 'cosine',
        'encoder_init_sha256': init_sha, 'set_conditioner_init_sha256': sc_init_sha,
        'scratch_init': True, 'encoder_trainable': True,
        'stage2_host': cli.stage2_host, 'reference_ckpt': cli.reference_ckpt,
        'seq_len': int(args.seq_len), 'pred_len': int(args.pred_len), 'enc_in': int(args.enc_in),
        'd_model': d_model, 'top_k': cli.top_k, 'tau_topk': cli.tau_topk, 'tau_choice': cli.tau_choice,
        'hyperparameter_provenance': hp_source,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse',
        'aggregation': 'softmax(host_score/tau_topk) over the K picks (host-fixed, exogenous)',
        'loss_formula': {
            'R0_choice_ce': 'CE(u_hat, argmax(u_target)) -- unchanged from factorial',
            'R1_wce': 'weighted_topk_listwise_ce(log p_S, {Top-10 Oracle by u_target LIVE on-policy}, tau=tau_choice)',
            'R2_pairwise': f'softplus(-(s_i-s_j)) over sampled pairs with u_i-u_j > {cli.min_gap_pairwise}, pool=top8+hard8+rand8',
            'R3_listwise': f'CE(teacher=softmax(u_target[TopM]/tau), student=softmax(u_hat[TopM])), M={cli.m_listwise} (LOSS SUPPORT ONLY)',
            'R4_hybrid': f'R0 + {cli.lambda_hybrid} * R3',
        }[cli.loss_name],
    }
    (cell_dir / f'config_fingerprint_{cli.loss_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[onpolicy_rankloss01] {cli.cell}/{cli.loss_name} init_sha={init_sha[:16]} '
          f'sc_init_sha={sc_init_sha[:16]}')

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
            te, _ = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader,
                               channels, device)
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

        print(f"[onpolicy_rankloss01] {cli.loss_name} epoch {epoch} "
              f"train_loss={tr['train_loss']:.5f} "
              f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
              f"eff_rank={rep['effective_rank']:.3f} pair_cos={rep['mean_pairwise_cosine']:.4f} "
              f"enc_gn={tr['encoder_grad_norm']:.5f}")

        if epoch - best['epoch'] >= cli.patience:
            print(f'[onpolicy_rankloss01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te, stepwise_best = eval_epoch(exp, args, host, model, set_conditioner, cli, test_loader,
                                   channels, device, collect_stepwise=True)

    with open(cell_dir / f'epoch_metrics_{cli.loss_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    with open(cell_dir / f'stepwise_metrics_{cli.loss_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in stepwise_best for k in r})
        w = csv.DictWriter(fh, fieldnames=['step'] + [k for k in keys if k != 'step'])
        w.writeheader()
        for r in stepwise_best:
            w.writerow(r)
    with open(cell_dir / f'representation_metrics_{cli.loss_name}.csv', 'w', newline='') as fh:
        keys = ['epoch'] + [k for k in sorted(epoch_rows[0]) if k.startswith('rep_')]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in epoch_rows:
            w.writerow({k: r.get(k) for k in keys})

    matched_val = {f'epoch{e}': next((r.get('val_free_running_aggregate_future_mse')
                                      for r in epoch_rows if r['epoch'] == e), None)
                  for e in sorted(test_at)}
    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None)
                    for e in sorted(test_at)}
    summary = {
        'cell': cli.cell, 'loss_name': cli.loss_name,
        'best_epoch': best['epoch'],
        'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'],
        'final_val_free_running_aggregate_future_mse':
            epoch_rows[-1]['val_free_running_aggregate_future_mse'],
        'matched_epoch_val': matched_val, 'matched_epoch_test': matched_test,
        'test_internal': dict(te),
        'encoder_init_sha256': init_sha, 'set_conditioner_init_sha256': sc_init_sha,
        'final_effective_rank': epoch_rows[-1].get('rep_effective_rank'),
        'final_mean_pairwise_cosine': epoch_rows[-1].get('rep_mean_pairwise_cosine'),
        'final_encoder_param_displacement': epoch_rows[-1].get('rep_encoder_param_displacement'),
        'wall_clock_seconds': time.time() - t0,
        'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.loss_name}.json').write_text(json.dumps(summary, indent=2))
    print(f"[onpolicy_rankloss01] done. {cli.cell}/{cli.loss_name} best_epoch={best['epoch']} "
          f"best_val_fr_agg={best['val']:.6f} test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
