#!/usr/bin/env python3
"""TRACK-A-SET-LOSS-CONTROL01 -- controlled, full-7-channel, matched-init
comparison of Hard Choice CE / Adaptive MultiPos / Soft Regret Mass /
Set-Utility Soft CE for the Greedy Set Oracle under On-policy prefix.

Corrects the two flaws EXP-SET-LOSS01 was found to have: (1) it trained on
`channel = 0` only while Stage-2 evaluates all 7 ETTh1 channels; (2) it had
no Hard-Choice-CE arm run at the SAME seed, so no direct A0-vs-{A1,A2,A3}
comparison was possible. This script is a NEW, separate trainer -- it does
not modify `scripts/train_factorial_e2e01.py`,
`scripts/train_multipos_choice01.py`, or `utils/set_loss_experimental.py`;
it imports the loss/Oracle/diagnostic machinery from all three, unchanged.

t=0 is ALWAYS Hard Choice CE, for every arm including A1/A2/A3 -- t=0's
prefix is empty (Individual-like selection), so arm differences are
deliberately confined to t>=1, the genuinely Set-conditioned steps
(spec section 3.3). On-policy: the prefix at every step is the model's
OWN prior argmax picks (never teacher-forced, never cached across steps);
the Oracle/Set-utility target is recomputed from THAT prefix every step
(spec section 3.4) -- this is exactly `run_sequence`'s existing
`model_next` / prefix-advance logic in `train_factorial_e2e01.py`,
reproduced here (not imported, since that module's `run_sequence` does
not have a per-step arm-dispatch hook) but using every one of that
module's own lower-level functions unchanged.
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

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (HostScorer, arm_score, candidate_weights,
                                           encode_raw, free_running_aggregate_future_mse,
                                           grad_norm, greedy_set_utility, param_displacement,
                                           representation_diagnostics, state_sha,
                                           step_rank_diagnostics)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_multipos_choice01 import ALPHA, multipos_choice_step_loss
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.set_loss_experimental import (compute_oracle_teacher_signal,
                                         set_utility_soft_ce_loss,
                                         soft_regret_mass_loss)

ARMS = ('A0_hard_choice', 'A1_adaptive_multipos', 'A2_srm', 'A3_setutility_softce')
M_TOPM = 10


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _tensor_sha256(t):
    h = hashlib.sha256()
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _shared_teacher_diag(u_hat, u_target, valid_now, tau):
    """Common, arm-independent diagnostics computed the SAME way regardless
    of which loss is optimized (spec section 9's redefined metrics)."""
    with torch.no_grad():
        teacher = compute_oracle_teacher_signal(u_target, valid_now, m=M_TOPM)
        rows = teacher['row_has_valid']
        model_idx = u_hat.masked_fill(~valid_now, torch.finfo(u_hat.dtype).min / 4).argmax(dim=-1)
        u_star = teacher['diag']['u_star']
        chosen_raw_regret = (u_star - u_target.gather(1, model_idx.unsqueeze(-1)).squeeze(-1))
        chosen_norm_regret = teacher['regret_norm'].gather(1, model_idx.unsqueeze(-1)).squeeze(-1)
        chosen_norm_regret = chosen_norm_regret.clamp(max=1e6)
        support_size = teacher['p_mask'].float().sum(dim=-1)
        w = teacher['weight']
        w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pi = w / w_sum
        pi_c = pi.clamp_min(1e-12)
        h_pi = -(pi_c * pi_c.log()).sum(-1)
        eff_pos = h_pi.exp()
        ess = 1.0 / (pi.pow(2).sum(-1).clamp_min(1e-12))
        norm_h_pi = h_pi / torch.log(support_size.clamp_min(2.0))

        logits = u_hat.masked_fill(~valid_now, torch.finfo(u_hat.dtype).min / 4) / float(tau)
        probs = F.softmax(logits, dim=-1)
        probs_c = probs.clamp_min(1e-12)
        student_entropy = -(probs_c * probs_c.log()).sum(-1)
        topm_mass = (probs * teacher['p_mask'].float()).sum(-1)

        def m(x):
            return float(x[rows].mean()) if bool(rows.any()) else float('nan')
        return {
            'chosen_raw_regret': m(chosen_raw_regret), 'chosen_normalized_regret': m(chosen_norm_regret),
            'support_size': m(support_size), 'teacher_entropy': m(h_pi),
            'normalized_teacher_entropy': m(norm_h_pi), 'effective_positive_count': m(eff_pos),
            'teacher_ess': m(ess), 'student_entropy': m(student_entropy),
            'topm_probability_mass': m(topm_mass),
        }


def compute_step_loss(arm, t, u_hat, u_target, valid_now, tau):
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


def run_sequence(z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                 arm, tau_choice, k, chunk_size, free_running=False):
    """On-policy ONLY (scope: no TF this round). At every step: prefix is
    the model's own prior argmax picks (never cached, never teacher-
    forced); Set-Oracle target is recomputed fresh from that prefix
    (`greedy_set_utility`, the unmodified reference). `free_running=True`
    skips loss/Oracle entirely (matches train_factorial_e2e01.py's own
    free-running contract: no future information reaches selection)."""
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
            m_ = E[prefix].mean(dim=1)
            h_t = set_conditioner(z_q, m_)
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

        loss_t, diag_t = compute_step_loss(arm, t, u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)

        model_next = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
        oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True)
        nxt = model_next  # ON-POLICY, unconditionally (scope: no TF)

        with torch.no_grad():
            step_records.append({'step': t, 'oracle_idx': oracle_next.squeeze(-1),
                                 'model_idx': model_next.squeeze(-1), 'u_target': u_target,
                                 'u_hat': u_hat.detach(), 'valid_now': valid_now})
        picks.append(nxt.squeeze(-1).detach())
        selected = selected.scatter(1, nxt, True)

    return losses, diags, torch.stack(picks, dim=1), step_records


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
    agg_ch, aggn_ch = {}, {}  # per-channel FR-Agg-adjacent accumulation done at eval, not here
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0
    ch_grad_seen = {c: False for c in channels}

    for batch_x, batch_y, batch_start_idx in loader:
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
                z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                cli.arm, cli.tau_choice, cli.top_k, cli.chunk_size)
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
        for c in channels:
            pass  # per-channel gradient-existence check is done in the unit tests, not per-batch here
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters()))
        gn_n += 1
        cli.optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1),
           'encoder_grad_norm': enc_gn / max(gn_n, 1), 'score_layer_grad_norm': sc_gn / max(gn_n, 1)}
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
                z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                cli.arm, cli.tau_choice, cli.top_k, cli.chunk_size, free_running=True)
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
    ap.add_argument('--checkpoints', default='checkpoints/track_a_set_loss_control01')
    ap.add_argument('--out_dir', default='results/TRACK-A-SET-LOSS-CONTROL01')
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
    cli = ap.parse_args()

    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_ck = torch.load(cli.stage2_host, map_location='cpu')
    host_args = dict(host_ck['args'])
    cli.tau_choice = 0.1  # spec: student temperature = 0.1 (fixed, not read from ref_args this round)
    cli.tau_topk = float(host_args['tau_topk'])

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    stage2_host_sha256 = _file_sha256(cli.stage2_host)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.seed, 'top_k': cli.top_k, 'tau_topk': cli.tau_choice,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))

    data_split_sha256 = _tensor_sha256(torch.cat([exp.memory_x[:64].flatten(),
                                                   exp.memory_y[:64].flatten()]))
    memory_sha256 = _tensor_sha256(torch.cat([exp.memory_x.flatten()[:200000],
                                              exp.memory_y.flatten()[:200000]]))

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)

    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['model_state_dict'])
        set_conditioner.load_state_dict(blob['sc_state_dict'])
        got_model = state_sha(model.state_dict())
        got_sc = state_sha(set_conditioner.state_dict())
        if got_model != blob['encoder_init_sha256'] or got_sc != blob['set_conditioner_init_sha256']:
            raise SystemExit(f'[ABORT] shared-init SHA mismatch: model {got_model} != '
                             f'{blob["encoder_init_sha256"]} or sc {got_sc} != '
                             f'{blob["set_conditioner_init_sha256"]}')
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
                        'cell': cli.cell}, cli.shared_init_out)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    param_count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in set_conditioner.parameters())

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(set_conditioner.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    torch.manual_seed(cli.seed)
    _, train_loader = exp._get_data(flag='train')
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:256]

    fingerprint = {
        'exp': 'TRACK-A-SET-LOSS-CONTROL01', 'cell': cli.cell, 'arm': cli.arm,
        'axis_oracle': 'greedy_set', 'axis_prefix': 'onpolicy', 'axis_score': 'cosine',
        'loss_name': cli.arm, 'seed': cli.seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'data_split_sha256': data_split_sha256, 'memory_sha256': memory_sha256,
        'stage2_host_sha256': stage2_host_sha256, 'param_count': param_count,
        'top_k': cli.top_k, 'tau_choice': cli.tau_choice, 'tau_topk': cli.tau_topk,
        'learning_rate': cli.learning_rate, 'batch_size': cli.batch_size,
        'epochs': cli.train_epochs, 'patience': cli.patience, 'weight_decay': cli.weight_decay,
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse (all channels)',
        'optimizer': 'Adam', 'reference_ckpt': cli.reference_ckpt, 'stage2_host': cli.stage2_host,
        'alpha_multipos': ALPHA if cli.arm == 'A1_adaptive_multipos' else None,
        'm_topm': M_TOPM,
    }
    (cell_dir / f'config_fingerprint_{cli.arm}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[set_loss_control01] {cli.cell}/{cli.arm} encoder_init_sha={encoder_init_sha256[:16]} '
         f'sc_init_sha={set_conditioner_init_sha256[:16]} channels={channels}')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    stepwise_best = []
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, cli, train_loader, channels, device)
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

        print(f"[set_loss_control01] {cli.arm} epoch {epoch} train_loss={tr['train_loss']:.5f} "
             f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
             f"chosen_regret_tge1={row.get('val_chosen_normalized_regret_tge1', float('nan')):.4f} "
             f"enc_gn={tr['encoder_grad_norm']:.5f}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[set_loss_control01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    te, stepwise_best, test_picks = eval_epoch(exp, args, host, model, set_conditioner, cli,
                                               test_loader, channels, device, collect_stepwise=True)

    # save the best checkpoint's free-running TEST picks for Stage-1/Stage-2
    # pick-agreement verification (section 7's required equality test)
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

    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None)
                    for e in sorted(test_at)}
    summary = {
        'exp': 'TRACK-A-SET-LOSS-CONTROL01', 'cell': cli.cell, 'arm': cli.arm,
        'best_epoch': best['epoch'], 'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'], 'matched_epoch_test': matched_test,
        'test_internal': te, 'encoder_init_sha256': encoder_init_sha256,
        'set_conditioner_init_sha256': set_conditioner_init_sha256,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[set_loss_control01] done. {cli.cell}/{cli.arm} best_epoch={best['epoch']} "
         f"best_val_fr_agg={best['val']:.6f} test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
