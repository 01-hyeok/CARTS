#!/usr/bin/env python3
"""TRACK-A-SET-DIFFICULTY01 -- why does the Greedy Set Oracle look harder to
learn than the Individual Oracle?

DIAGNOSTIC ONLY. No training. Loads TRACK-A-FACTORIAL-E2E01's own best
checkpoints for `individual_onpolicy_cosine` and `set_onpolicy_cosine`
(resolved from those experiments' own `retrieval_metrics_*.json`, never
guessed or retrained) and evaluates four things, all on the model's own
FREE-RUNNING trajectory (never teacher-forced, never fed future information):

  Q1/Q2/Q3 -- step-wise difficulty (`step_rank_diagnostics`, reused
             unmodified from `train_factorial_e2e01.py`)
  Q2       -- near-tie / ambiguity of the Oracle utility distribution
  Q3       -- prefix perturbation sensitivity (Set Oracle only; Individual
             Oracle is the built-in control and MUST be invariant)
  Q4       -- ambiguity-binned accuracy/regret cross-tab

Pre-registered, fixed choices (per spec: decide once, never re-tuned after
seeing results):
  - Perturbation rule = P1 (last-action replacement): the LAST prefix
    element is replaced by the model's own SECOND-BEST valid candidate at
    the step that element was originally chosen. Deterministic.
  - Near-tie threshold = fraction of the per-query (u_1 - u_10) range:
    frac in {0.001, 0.01, 0.05}. Absolute thresholds are not used because
    the utility scale (-MSE) differs by an order of magnitude between H96
    and H720.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (arm_score, encode_raw,
                                           greedy_set_utility, individual_utility,
                                           step_rank_diagnostics)
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights

NEAR_TIE_FRACS = (0.001, 0.01, 0.05)
TOP_K = 10


def resolve_checkpoint(factorial_root, cell, arm):
    p = Path(factorial_root) / cell / f'retrieval_metrics_{arm}.json'
    if not p.exists():
        raise SystemExit(f'[ABORT] cannot resolve checkpoint: {p} does not exist')
    d = json.loads(p.read_text())
    ckpt_path = d['checkpoint']
    if not Path(ckpt_path).exists():
        raise SystemExit(f'[ABORT] resolved checkpoint path does not exist on disk: {ckpt_path}')
    return ckpt_path, d


def load_arm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = SimpleNamespace(**ckpt['args'])
    exp = Exp_Stage1_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()
    return exp, args, model, sc, ckpt


class HostScorer:
    """Same fixed exogenous aggregation weighting as the factorial trainer
    (read-only, frozen host, no gradient)."""
    def __init__(self, stage2_ckpt, device):
        from utils.retrieval_diagnostics import load_stage2
        self.exp, self.args = load_stage2(stage2_ckpt)
        self.exp._ensure_memory()
        self.exp._build_key_bank()
        self.model = self.exp.model.module if hasattr(self.exp.model, 'module') else self.exp.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.score_fn = self.model._retrieval_score_fn()
        self.tau_topk = float(self.args.tau_topk)

    @torch.no_grad()
    def scores(self, batch_x, c, cand_mask):
        z_q = self.model._branch_embedding(batch_x, c, c)
        z_mem = self.model._branch_memory(self.exp.key_bank, c, 0, c, z_q.dtype, self.device)
        s = (self.score_fn(z_q, z_mem) if self.score_fn is not None
             else torch.matmul(z_q, z_mem.transpose(0, 1)))
        return s.masked_fill(~cand_mask, torch.finfo(s.dtype).min / 4)


@torch.no_grad()
def free_running_trajectory(model, set_conditioner, z_q, E, cand_mask, oracle_kind,
                            w_host, futures, query_future, k=TOP_K):
    """Walk the model's OWN free-running prefix (argmax every step, no
    future information anywhere), recording at each step the state needed
    for every downstream diagnostic: u_hat, u_target, valid_now, picks so
    far. Structurally leak-free: nothing here reads `futures`/`query_future`
    for SELECTION, only for building the (oracle-target-only) u_target used
    purely as a diagnostic label, exactly mirroring the factorial trainer's
    own eval_epoch."""
    bsz, device = z_q.size(0), z_q.device
    selected = torch.zeros_like(cand_mask)
    picks = []
    steps = []
    neg_inf = torch.finfo(z_q.dtype).min / 4
    for t in range(k):
        if t == 0:
            h_t = z_q
        else:
            m = E[torch.stack(picks, dim=1)].mean(dim=1)
            h_t = set_conditioner(z_q, m)
        u_hat = arm_score(h_t, E, None)
        valid_now = cand_mask & ~selected
        if oracle_kind == 'individual':
            u_target = individual_utility(futures, query_future)
        else:
            prefix_now = (torch.stack(picks, dim=1) if picks
                          else torch.zeros(bsz, 0, dtype=torch.long, device=device))
            u_target = greedy_set_utility(prefix_now, w_host, futures, query_future, None)
        oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        model_idx = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        steps.append({'t': t + 1, 'u_hat': u_hat, 'u_target': u_target,
                     'valid_now': valid_now, 'oracle_idx': oracle_idx, 'model_idx': model_idx,
                     'prefix_before': (torch.stack(picks, dim=1) if picks
                                       else torch.zeros(bsz, 0, dtype=torch.long, device=device))})
        picks.append(model_idx)
        selected = selected.scatter(1, model_idx.unsqueeze(-1), True)
    return steps, torch.stack(picks, dim=1)


@torch.no_grad()
def ambiguity_diagnostics(u_target, valid_now):
    """gap_1_2, gap_1_5, gap_1_10 (absolute) + near-tie counts at three
    relative fractions of the per-query (u_1 - u_10) range."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked = u_target.masked_fill(~valid_now, neg_inf)
    n_valid = valid_now.sum(dim=-1)
    k_eff = min(10, int(n_valid.min().clamp_min(1)))
    top_vals, _ = masked.topk(k_eff, dim=-1)
    u1 = top_vals[:, 0]
    u2 = top_vals[:, 1] if k_eff >= 2 else top_vals[:, 0]
    u5 = top_vals[:, min(4, k_eff - 1)]
    u10 = top_vals[:, -1]
    gap_1_2 = u1 - u2
    gap_1_5 = u1 - u5
    gap_1_10 = u1 - u10
    out = {'gap_1_2_mean': float(gap_1_2.mean()), 'gap_1_5_mean': float(gap_1_5.mean()),
          'gap_1_10_mean': float(gap_1_10.mean()),
          'gap_1_2_median': float(gap_1_2.median()), 'utility_range_mean': float(gap_1_10.mean())}
    rng = gap_1_10.clamp_min(1e-12)
    for frac in NEAR_TIE_FRACS:
        eps = frac * rng
        near_tie = (masked >= (u1 - eps).unsqueeze(-1)) & valid_now
        out[f'near_tie_count_frac{frac}'] = float(near_tie.float().sum(dim=-1).mean())
    return out, gap_1_2


@torch.no_grad()
def perturb_prefix_p1(prefix_before, u_hat, valid_now, model_idx_at_last_step):
    """P1 -- last-action replacement: the LAST element of `prefix_before ++
    [model_idx_at_last_step]` becomes the model's SECOND-BEST valid
    candidate at that same step's u_hat, instead of its actual (best)
    argmax pick. Deterministic, no randomness."""
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    masked = u_hat.masked_fill(~valid_now, neg_inf)
    top2 = masked.topk(2, dim=-1).indices
    second_best = top2[:, 1]
    # guard: if the "second best" happens to equal the actual pick (can't
    # happen since top-1 IS the actual pick, but keep explicit for safety)
    perturbed_last = torch.where(second_best == model_idx_at_last_step, top2[:, 0], second_best)
    if prefix_before.size(1) == 0:
        return perturbed_last.unsqueeze(-1)
    return torch.cat([prefix_before, perturbed_last.unsqueeze(-1)], dim=-1)


@torch.no_grad()
def set_oracle_ranking(prefix, w_host, futures, query_future, cand_mask):
    u = greedy_set_utility(prefix, w_host, futures, query_future, None)
    return u.masked_fill(~cand_mask, torch.finfo(u.dtype).min / 4)


@torch.no_grad()
def overlap_at_k(order_a, order_b, k):
    a, b = set(order_a[:k].tolist()), set(order_b[:k].tolist())
    return len(a & b) / k


@torch.no_grad()
def spearman_kendall(u_a, u_b, valid):
    a = u_a[valid]
    b = u_b[valid]
    if a.numel() < 3:
        return float('nan'), float('nan')
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    spearman = float((ra * rb).sum() / denom)
    # Kendall tau via a sampled pairwise concordance estimate (full O(n^2) on
    # a valid set of thousands is too slow; sample pairs, fixed seed)
    n = a.numel()
    g = torch.Generator().manual_seed(0)
    n_pairs = min(20000, n * (n - 1) // 2)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    concordant = ((a[i] - a[j]) * (b[i] - b[j])) > 0
    discordant = ((a[i] - a[j]) * (b[i] - b[j])) < 0
    denom_k = (concordant.float().sum() + discordant.float().sum()).clamp_min(1)
    kendall = float((concordant.float().sum() - discordant.float().sum()) / denom_k)
    return spearman, kendall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=('ETTh1_96', 'ETTh1_720'))
    ap.add_argument('--factorial_root', default='results/track_a_factorial_e2e')
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--out_dir', default='results/TRACK-A-SET-DIFFICULTY01')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    ind_ckpt, ind_fp = resolve_checkpoint(cli.factorial_root, cli.cell, 'individual_onpolicy_cosine')
    set_ckpt, set_fp = resolve_checkpoint(cli.factorial_root, cli.cell, 'set_onpolicy_cosine')
    print(f'[set_difficulty01] {cli.cell} individual_ckpt={ind_ckpt} (best_epoch={ind_fp["best_epoch"]})')
    print(f'[set_difficulty01] {cli.cell} set_ckpt={set_ckpt} (best_epoch={set_fp["best_epoch"]})')

    host = HostScorer(cli.stage2_host, device)

    arms = {}
    for oracle_kind, ckpt_path, fp in (('individual', ind_ckpt, ind_fp), ('set', set_ckpt, set_fp)):
        exp, args, model, sc, ckpt = load_arm(ckpt_path, device)
        exp._ensure_memory()
        got_fp = ckpt.get('fingerprint', {})
        mismatches = []
        for key in ('seq_len', 'pred_len', 'enc_in', 'd_model', 'top_k'):
            if str(got_fp.get(key)) != str(fp['fingerprint'].get(key)):
                mismatches.append(key)
        if mismatches:
            raise SystemExit(f'[ABORT] {oracle_kind} checkpoint fingerprint mismatch on {mismatches}')
        arms[oracle_kind] = {'exp': exp, 'args': args, 'model': model, 'sc': sc, 'ckpt': ckpt}
        print(f'[set_difficulty01] {cli.cell}/{oracle_kind} fingerprint OK '
              f'(axis_oracle={got_fp.get("axis_oracle")}, axis_prefix={got_fp.get("axis_prefix")}, '
              f'axis_score={got_fp.get("axis_score")})')

    channels = list(range(int(arms['individual']['args'].enc_in)))
    _, test_loader = arms['individual']['exp']._get_data(flag='test', shuffle=False)

    # ---- accumulators ----
    step_sums = {'individual': {}, 'set': {}}
    step_n = {'individual': {}, 'set': {}}
    amb_sums = {'individual': {}, 'set': {}}
    amb_n = {'individual': {}, 'set': {}}
    gap_bins = {'individual': {'low': [], 'medium': [], 'high': []},
               'set': {'low': [], 'medium': [], 'high': []}}
    bin_acc_regret = {'individual': {'low': [0.0, 0.0, 0], 'medium': [0.0, 0.0, 0], 'high': [0.0, 0.0, 0]},
                      'set': {'low': [0.0, 0.0, 0], 'medium': [0.0, 0.0, 0], 'high': [0.0, 0.0, 0]}}
    sens_sums, sens_n = {}, {}
    control_sums, control_n = {}, {}
    dup_count, invalid_count, rows_total = 0, 0, 0
    nan_flag = False

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(test_loader):
        if cli.limit_batches and bi >= cli.limit_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = arms['individual']['exp']._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= TOP_K

        for oracle_kind in ('individual', 'set'):
            a = arms[oracle_kind]
            model, sc, exp, args = a['model'], a['sc'], a['exp'], a['args']
            for c in channels:
                E = encode_raw(model, exp.memory_x, c)
                z_q = encode_raw(model, batch_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                futures = memory_c + offset_c.view(-1, 1, 1)
                q_future = batch_y[:, :, c]
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)

                steps, picks = free_running_trajectory(
                    model, sc, z_q, E, cand_mask, oracle_kind, w_host, futures, q_future)

                for b in range(picks.size(0)):
                    if not valid_query[b]:
                        continue
                    row = picks[b].tolist()
                    rows_total += 1
                    if len(set(row)) != len(row):
                        dup_count += 1
                    if not bool(cand_mask[b, row].all()):
                        invalid_count += 1

                for st in steps:
                    vq = valid_query
                    diag = step_rank_diagnostics(st['u_hat'][vq], st['u_target'][vq], st['valid_now'][vq],
                                                 st['oracle_idx'][vq], st['model_idx'][vq], futures[vq], q_future[vq])
                    amb, gap12 = ambiguity_diagnostics(st['u_target'][vq], st['valid_now'][vq])
                    t = st['t']
                    for kk, vv in diag.items():
                        key = f't{t}::{kk}'
                        step_sums[oracle_kind][key] = step_sums[oracle_kind].get(key, 0.0) + vv
                        step_n[oracle_kind][key] = step_n[oracle_kind].get(key, 0) + 1
                    for kk, vv in amb.items():
                        key = f't{t}::{kk}'
                        amb_sums[oracle_kind][key] = amb_sums[oracle_kind].get(key, 0.0) + vv
                        amb_n[oracle_kind][key] = amb_n[oracle_kind].get(key, 0) + 1
                    if any(torch.isnan(st['u_hat']).any().item() for _ in [0]) or \
                       torch.isnan(st['u_target'][st['valid_now']]).any().item():
                        nan_flag = True

                    # ambiguity binning: low(gap>=median-ish uses global thirds via
                    # pre-set fixed cutoffs on relative gap = gap12 / range)
                    rng = (amb['utility_range_mean'] if amb['utility_range_mean'] > 1e-12 else 1e-12)
                    rel_gap = (gap12 / rng).clamp(0, 1)
                    acc = (st['model_idx'][vq] == st['oracle_idx'][vq]).float()
                    tgt = st['u_target'].masked_fill(~st['valid_now'], torch.finfo(st['u_target'].dtype).min / 4)
                    regret = (tgt.gather(1, st['oracle_idx'].unsqueeze(-1)).squeeze(-1)
                             - tgt.gather(1, st['model_idx'].unsqueeze(-1)).squeeze(-1))[vq]
                    rel_gap_v = rel_gap[vq]
                    low = rel_gap_v >= 0.10
                    high = rel_gap_v < 0.01
                    med = (~low) & (~high)
                    for name, mask in (('low', low), ('medium', med), ('high', high)):
                        if bool(mask.any()):
                            bin_acc_regret[oracle_kind][name][0] += float(acc[mask].sum())
                            bin_acc_regret[oracle_kind][name][1] += float(regret[mask].sum())
                            bin_acc_regret[oracle_kind][name][2] += int(mask.sum())

                # ---- Set-only (and Individual control) prefix sensitivity, P1, t=2..10 ----
                for st in steps[1:]:
                    t = st['t']
                    perturbed_prefix = perturb_prefix_p1(
                        st['prefix_before'][:, :-1] if st['prefix_before'].size(1) > 0 else st['prefix_before'],
                        steps[t - 2]['u_hat'], steps[t - 2]['valid_now'], steps[t - 2]['model_idx'])
                    # original prefix at this step = prefix_before ++ model_idx_of_prev_step
                    orig_prefix = st['prefix_before']
                    if oracle_kind == 'set':
                        u_orig = set_oracle_ranking(orig_prefix, w_host, futures, q_future, st['valid_now'])
                        u_pert = set_oracle_ranking(perturbed_prefix, w_host, futures, q_future, st['valid_now'])
                    else:
                        u_orig = individual_utility(futures, q_future).masked_fill(~st['valid_now'],
                                                     torch.finfo(futures.dtype).min / 4)
                        u_pert = u_orig  # prefix-invariant by construction; still measured as control

                    vq = valid_query
                    for b_idx in vq.nonzero(as_tuple=True)[0].tolist():
                        vb = st['valid_now'][b_idx]
                        if int(vb.sum()) < 10:
                            continue
                        oa = u_orig[b_idx][vb].argsort(descending=True)
                        pa = u_pert[b_idx][vb].argsort(descending=True)
                        idx_valid = vb.nonzero(as_tuple=True)[0]
                        top1_agree = float(oa[0] == pa[0])
                        top5 = overlap_at_k(oa, pa, min(5, oa.numel()))
                        top10 = overlap_at_k(oa, pa, min(10, oa.numel()))
                        sp, kt = spearman_kendall(u_orig[b_idx], u_pert[b_idx], vb)
                        acc_dict = (sens_sums if oracle_kind == 'set' else control_sums)
                        n_dict = (sens_n if oracle_kind == 'set' else control_n)
                        for kk, vv in (('top1_agreement', top1_agree), ('top5_overlap', top5),
                                      ('top10_overlap', top10), ('spearman', sp), ('kendall', kt)):
                            if vv == vv:
                                key = f't{t}::{kk}'
                                acc_dict[key] = acc_dict.get(key, 0.0) + vv
                                n_dict[key] = n_dict.get(key, 0) + 1

    if nan_flag:
        raise SystemExit('[ABORT] NaN detected in u_hat/u_target during diagnostic run')

    # ---- write stepwise csvs ----
    for oracle_kind in ('individual', 'set'):
        rows = []
        for t in range(1, TOP_K + 1):
            row = {'t': t}
            for kk in step_sums[oracle_kind]:
                if kk.startswith(f't{t}::'):
                    row[kk.split('::', 1)[1]] = step_sums[oracle_kind][kk] / max(step_n[oracle_kind][kk], 1)
            for kk in amb_sums[oracle_kind]:
                if kk.startswith(f't{t}::'):
                    row[kk.split('::', 1)[1]] = amb_sums[oracle_kind][kk] / max(amb_n[oracle_kind][kk], 1)
            rows.append(row)
        with open(out_dir / f'{oracle_kind}_stepwise.csv', 'w', newline='') as fh:
            keys = ['t'] + sorted({k for r in rows for k in r if k != 't'})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # ---- ambiguity summary csv ----
    with open(out_dir / 'ambiguity.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['oracle', 'gap_1_2_mean', 'gap_1_5_mean', 'gap_1_10_mean'] +
                  [f'near_tie_count_frac{f}' for f in NEAR_TIE_FRACS])
        for oracle_kind in ('individual', 'set'):
            g12 = sum(amb_sums[oracle_kind][k] for k in amb_sums[oracle_kind] if k.endswith('::gap_1_2_mean')) / \
                max(sum(amb_n[oracle_kind][k] for k in amb_n[oracle_kind] if k.endswith('::gap_1_2_mean')), 1)
            g15 = sum(amb_sums[oracle_kind][k] for k in amb_sums[oracle_kind] if k.endswith('::gap_1_5_mean')) / \
                max(sum(amb_n[oracle_kind][k] for k in amb_n[oracle_kind] if k.endswith('::gap_1_5_mean')), 1)
            g110 = sum(amb_sums[oracle_kind][k] for k in amb_sums[oracle_kind] if k.endswith('::gap_1_10_mean')) / \
                max(sum(amb_n[oracle_kind][k] for k in amb_n[oracle_kind] if k.endswith('::gap_1_10_mean')), 1)
            row = [oracle_kind, g12, g15, g110]
            for f in NEAR_TIE_FRACS:
                key = f'near_tie_count_frac{f}'
                v = sum(amb_sums[oracle_kind][k] for k in amb_sums[oracle_kind] if k.endswith(f'::{key}')) / \
                    max(sum(amb_n[oracle_kind][k] for k in amb_n[oracle_kind] if k.endswith(f'::{key}')), 1)
                row.append(v)
            w.writerow(row)

    # ---- prefix sensitivity csv (Set = primary, Individual = control) ----
    with open(out_dir / 'prefix_sensitivity.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t', 'set_top1_agreement', 'set_top5_overlap', 'set_top10_overlap',
                   'set_spearman', 'set_kendall', 'control_individual_top1_agreement',
                   'control_individual_top10_overlap', 'control_individual_spearman'])
        for t in range(2, TOP_K + 1):
            def g(d, n, name):
                key = f't{t}::{name}'
                return d.get(key, float('nan')) / max(n.get(key, 1), 1)
            w.writerow([t, g(sens_sums, sens_n, 'top1_agreement'), g(sens_sums, sens_n, 'top5_overlap'),
                       g(sens_sums, sens_n, 'top10_overlap'), g(sens_sums, sens_n, 'spearman'),
                       g(sens_sums, sens_n, 'kendall'), g(control_sums, control_n, 'top1_agreement'),
                       g(control_sums, control_n, 'top10_overlap'), g(control_sums, control_n, 'spearman')])

    # ---- ambiguity-bin cross-tab ----
    bin_rows = []
    for oracle_kind in ('individual', 'set'):
        for name in ('low', 'medium', 'high'):
            acc_sum, reg_sum, n = bin_acc_regret[oracle_kind][name]
            bin_rows.append({'oracle': oracle_kind, 'ambiguity_bin': name,
                            'oracle_action_acc': acc_sum / max(n, 1),
                            'regret_mean': reg_sum / max(n, 1), 'n': n})
    with open(out_dir / 'ambiguity_bins.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['oracle', 'ambiguity_bin', 'oracle_action_acc', 'regret_mean', 'n'])
        w.writeheader()
        for r in bin_rows:
            w.writerow(r)

    summary = {
        'cell': cli.cell,
        'individual_checkpoint': ind_ckpt, 'individual_best_epoch': ind_fp['best_epoch'],
        'set_checkpoint': set_ckpt, 'set_best_epoch': set_fp['best_epoch'],
        'perturbation_rule': 'P1_last_action_replacement',
        'near_tie_fracs': list(NEAR_TIE_FRACS),
        'duplicate_rate': dup_count / max(rows_total, 1),
        'invalid_rate': invalid_count / max(rows_total, 1),
        'rows_total': rows_total,
        'no_new_training': True, 'no_encoder_retrain': True,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[set_difficulty01] done. {cli.cell} dup_rate={summary["duplicate_rate"]} '
          f'invalid_rate={summary["invalid_rate"]}')


if __name__ == '__main__':
    main()
