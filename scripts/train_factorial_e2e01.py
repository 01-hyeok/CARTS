#!/usr/bin/env python3
"""TRACK-A-FACTORIAL-E2E01 -- Stage-1 trainer for one arm of the
Oracle x Prefix x Score factorial, with a SCRATCH, TRAINABLE encoder.

One invocation = one arm. Exactly three things may differ between arms:

    --target        individual | greedy_set        (Axis 1)
    --prefix_policy tf | onpolicy                  (Axis 2)
    --scorer        cosine | asymmetric            (Axis 3)

Everything else -- dataset, split, memory construction, candidate mask,
encoder architecture, INITIAL ENCODER WEIGHTS, selector architecture,
optimizer, LR, weight decay, batch size, epoch budget, early-stopping rule,
K, aggregation function, loss formulation, temperature, checkpoint-selection
rule, seed -- is held fixed and is asserted/recorded in `config_fingerprint`.

Design decisions that are DERIVED FROM THE SPEC, not chosen here
-----------------------------------------------------------------
1. Aggregation function (spec S4 "aggregation function" identical across
   arms; S7 "Individual과 Greedy Set 모두 동일한 aggregation function";
   S16 "Stage-2에서 사용하는 것과 동일한 aggregation").

   Stage-2's forced-selection path (`utils/retrieval_ops.py:90-94`) forms
   `alpha = softmax(host_score(picks)/tau_topk)` where `host_score` is the
   Stage-2 HOST's own frozen retrieval score. Therefore:

     * the Greedy Set Oracle's `Aggregate(S ∪ {i})` uses those same host
       weights, and
     * `free_running_aggregate_future_mse` uses those same host weights.

   The weighting is EXOGENOUS and byte-identical for all 8 arms. Using each
   arm's own score instead would make the Oracle target and the primary
   metric arm-dependent, which violates S4 and reintroduces the
   self-referential artifact recorded in
   `research/AUDIT_ORACLE_RANK_GAIN01.md` section 4.4.

2. Loss (spec S5): a single masked full-memory Oracle-Choice Cross-Entropy,
   `scripts/train_oracle_choice01.py::oracle_choice_step_loss`, reused
   verbatim for every arm. Only the target index `i*` changes.

3. Training hyperparameters are resolved from the STAGE-1 REFERENCE args
   (`--reference_ckpt`), never from the Stage-2 host. Every resolved value is
   written to `config_fingerprint["hyperparameter_provenance"]` together with
   the exact source it came from, and both the Stage-1 and Stage-2 arg
   subsets are stored alongside so the two origins stay distinguishable even
   where the numbers coincide.

   The Stage-2 host contributes exactly one number to this script: the
   aggregation temperature (`tau_aggregation`), because the aggregation is
   the host's. The Choice-CE temperature (`tau_choice`) comes from Stage-1.

Scratch-init discipline
-----------------------
`--reference_ckpt` supplies ARGS ONLY; its weights are never loaded
(`build_experiment` reconstructs the args and `Exp_Stage1_Relation`'s
`_build_model` only calls `Model(args).float()`). The first arm of a cell
writes `--shared_init_out`; the remaining seven load `--shared_init_in` and
the loader asserts exact tensor equality against the stored SHA256.
"""
import argparse
import copy
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

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_diagnostics import load_stage2

ARMS = ('individual', 'greedy_set')
PREFIXES = ('tf', 'onpolicy')
SCORERS = ('cosine', 'asymmetric')


# --------------------------------------------------------------------------
# encoder / scoring
# --------------------------------------------------------------------------
def encode_raw(model, x, c):
    """Un-normalised encoder output. Normalisation happens inside the scorer
    so the cosine and asymmetric families see the identical input."""
    return model.encoder(model._relation_tensor(x, c, c))


def arm_score(z_q, z_k, metric):
    """score(q, k). `metric is None` -> plain cosine (Axis-3 = cosine).
    Otherwise the identity-initialised asymmetric RetrievalMetric."""
    if metric is None:
        return torch.matmul(F.normalize(z_q, dim=-1),
                            F.normalize(z_k, dim=-1).transpose(0, 1))
    return metric.score(z_q, z_k)


def state_sha(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Oracle expert actions
# --------------------------------------------------------------------------
def individual_utility(futures, query_future):
    """u_i = -MSE(y_i, y_q). Step-invariant: the Individual Oracle ignores
    the prefix entirely (spec S6), so this is computed once per query."""
    return -((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)


def greedy_set_utility(prefix_idx, w_host, futures, query_future, chunk_size):
    """u_i^(t) = -MSE(Aggregate(S ∪ {i}), y_q), Aggregate weighted by the
    FIXED host weights (see module docstring, decision 1). `dense_utility`
    is `@torch.no_grad()` internally -- the Oracle is a target, never a
    gradient path."""
    return -dense_utility(prefix_idx, w_host, futures, query_future,
                          chunk_size=chunk_size)


# --------------------------------------------------------------------------
# one K-step sequence
# --------------------------------------------------------------------------
def run_sequence(z_q, E, cand_mask, set_conditioner, metric, w_host,
                 futures, query_future, target, prefix_policy,
                 tau_choice, k, chunk_size, free_running=False):
    """K steps of (state -> logits -> Oracle target -> Choice CE).

    prefix_policy='tf'       : prefix advances along the ORACLE's own argmax
    prefix_policy='onpolicy' : prefix advances along the MODEL's own argmax
    free_running=True        : prefix advances along the MODEL's own argmax
                               AND no Oracle target is computed at all.
                               Used for validation/test inference (spec S15)
                               so no future information can reach selection.

    In every mode the state at t=0 is the empty prefix and the logits are
    `arm_score(z_q, E)` -- identical for TF and on-policy at t=1 (spec S18).

    Returns (losses, diags, picks, step_records).
    """
    is_set = target == 'greedy_set'
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
        u_hat = arm_score(h_t, E, metric)
        valid_now = cand_mask & ~selected

        prefix_now = (torch.stack(picks, dim=1).detach() if picks
                      else torch.zeros(bsz, 0, dtype=torch.long, device=device))

        if free_running:
            nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
            picks.append(nxt.squeeze(-1))
            selected = selected.scatter(1, nxt, True)
            continue

        with torch.no_grad():
            if is_set:
                u_target = greedy_set_utility(prefix_now, w_host, futures,
                                              query_future, chunk_size)
            else:
                u_target = individual_utility(futures, query_future)

        loss_t, diag_t = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)

        oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True)
        model_next = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True).detach()
        nxt = oracle_next if prefix_policy == 'tf' else model_next

        with torch.no_grad():
            step_records.append({
                'step': t + 1,
                'oracle_idx': oracle_next.squeeze(-1),
                'model_idx': model_next.squeeze(-1),
                'u_target': u_target,
                'u_hat': u_hat.detach(),
                'valid_now': valid_now,
            })

        picks.append(nxt.squeeze(-1).detach())
        selected = selected.scatter(1, nxt, True)

    return losses, diags, torch.stack(picks, dim=1), step_records


# --------------------------------------------------------------------------
# primary metric
# --------------------------------------------------------------------------
def free_running_aggregate_future_mse(picks, host_scores, futures, query_future, tau_topk):
    """THE primary retrieval metric (spec S16).

    Identical formula to Stage-2's forced-selection aggregation
    (`utils/retrieval_ops.py:91-94`):
        alpha = softmax(host_score(picks) / tau_topk)
        y_ret = sum_j alpha_j * futures[picks_j]
        mse   = mean_h (y_ret - y_q)^2

    This is numerically the same quantity the existing code calls
    `A_weighted` (`scripts/eval_margutil01_stage2.py:167`); the explicit name
    is used per spec S16 rather than reusing the old one.
    """
    sc = host_scores.gather(1, picks)
    alpha = torch.softmax(sc / float(tau_topk), dim=-1)
    tgt = futures.gather(1, picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
    y_ret = (alpha.unsqueeze(-1) * tgt).sum(1)
    return (y_ret - query_future).pow(2).mean(-1)


# --------------------------------------------------------------------------
# representation diagnostics (spec S13)
# --------------------------------------------------------------------------
@torch.no_grad()
def representation_diagnostics(emb):
    """emb: [M, D] raw (un-normalised) encoder outputs from a FIXED probe set."""
    out = {}
    emb = emb.float()
    out['has_nan'] = bool(torch.isnan(emb).any() or torch.isinf(emb).any())
    if out['has_nan']:
        return {**out, 'effective_rank': float('nan'), 'mean_pairwise_cosine': float('nan'),
                'mean_dim_std': float('nan'), 'dead_dim_ratio': float('nan'),
                'embed_norm_mean': float('nan'), 'embed_norm_std': float('nan'),
                'sv1_fraction': float('nan')}
    centered = emb - emb.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    p = sv / sv.sum().clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    out['effective_rank'] = float(torch.exp(ent))
    out['sv1_fraction'] = float(p[0])
    out['singular_values_top8'] = [float(x) for x in sv[:8]]
    dim_std = emb.std(dim=0)
    out['mean_dim_std'] = float(dim_std.mean())
    out['dead_dim_ratio'] = float((dim_std < 1e-4).float().mean())
    n = F.normalize(emb, dim=-1)
    sim = torch.matmul(n, n.transpose(0, 1))
    off = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    out['mean_pairwise_cosine'] = float(sim[off].mean())
    norms = emb.norm(dim=-1)
    out['embed_norm_mean'] = float(norms.mean())
    out['embed_norm_std'] = float(norms.std())
    return out


@torch.no_grad()
def param_displacement(model, init_state):
    num, den = 0.0, 0.0
    cur = model.state_dict()
    for key, v0 in init_state.items():
        v = cur[key].detach().cpu().float()
        num += float((v - v0.float()).pow(2).sum())
        den += float(v0.float().pow(2).sum())
    return (num ** 0.5) / max(den ** 0.5, 1e-12)


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


# --------------------------------------------------------------------------
# ranking diagnostics on one step
# --------------------------------------------------------------------------
@torch.no_grad()
def step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx, model_idx,
                          futures, query_future, k_list=(1, 5, 10)):
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    logits = u_hat.masked_fill(~valid_now, neg_inf)
    oracle_logit = logits.gather(1, oracle_idx.unsqueeze(-1))
    rank = (logits > oracle_logit).sum(dim=-1).float() + 1.0     # 1-based
    n_valid = valid_now.sum(dim=-1).float().clamp_min(1.0)
    tgt = u_target.masked_fill(~valid_now, neg_inf)
    regret = tgt.gather(1, oracle_idx.unsqueeze(-1)).squeeze(-1) \
        - tgt.gather(1, model_idx.unsqueeze(-1)).squeeze(-1)
    sel_ind_mse = ((futures.gather(1, model_idx.view(-1, 1, 1).expand(-1, 1, futures.size(-1)))
                    .squeeze(1) - query_future) ** 2).mean(-1)
    out = {
        'expert_rank_mean': float(rank.mean()),
        'expert_rank_median': float(rank.median()),
        'expert_rank_fraction': float((rank / n_valid).mean()),
        'expert_regret_mean': float(regret.mean()),
        'oracle_action_acc': float((model_idx == oracle_idx).float().mean()),
        'selected_individual_future_mse': float(sel_ind_mse.mean()),
        'n_valid_mean': float(n_valid.mean()),
    }
    for kk in k_list:
        out[f'expert_containment_at_{kk}'] = float((rank <= kk).float().mean())
    # NDCG@10 with relevance = shifted oracle utility, ranked by the model
    rel = (tgt - tgt.masked_fill(~valid_now, float('inf')).min(dim=-1, keepdim=True).values)
    rel = rel.masked_fill(~valid_now, 0.0)
    order = logits.argsort(dim=-1, descending=True)[:, :10]
    gains = rel.gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(2, 12, device=u_hat.device).float()).unsqueeze(0)
    dcg = (gains * disc).sum(-1)
    ideal = rel.topk(10, dim=-1).values
    idcg = (ideal * disc).sum(-1).clamp_min(1e-12)
    out['ndcg_at_10'] = float((dcg / idcg).mean())
    # Spearman between predicted and true utility over valid candidates (per row)
    sp = []
    for b in range(min(u_hat.size(0), 16)):
        vb = valid_now[b]
        if int(vb.sum()) < 2:
            continue
        a = u_hat[b][vb].argsort().argsort().float()
        c = u_target[b][vb].argsort().argsort().float()
        a = a - a.mean()
        c = c - c.mean()
        sp.append(float((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-12)))
    out['spearman'] = sum(sp) / max(len(sp), 1) if sp else float('nan')
    return out


# --------------------------------------------------------------------------
# epoch loops
# --------------------------------------------------------------------------
def _iter_batches(loader, limit):
    """SMOKE ONLY helper: cap the number of batches. limit<=0 -> full loader."""
    for i, b in enumerate(loader):
        if limit and i >= limit:
            break
        yield b


def train_epoch(exp, args, host, model, set_conditioner, metric, cli,
                loader, channels, device):
    model.train(True)
    set_conditioner.train(True)
    if metric is not None:
        metric.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    tot_loss, nb = 0.0, 0
    agg, aggn = {}, 0
    enc_gn, sc_gn, gn_n = 0.0, 0.0, 0

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
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
                z_q, E, cand_mask, set_conditioner, metric, w_host,
                futures, query_future, cli.target, cli.prefix_policy,
                cli.tau_choice, cli.top_k, cli.chunk_size)

            batch_loss = batch_loss + sum(losses) / cli.top_k
            with torch.no_grad():
                fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                       query_future, cli.tau_topk)
                d0 = steps[0]
                s = step_rank_diagnostics(d0['u_hat'], d0['u_target'], d0['valid_now'],
                                          d0['oracle_idx'], d0['model_idx'],
                                          futures, query_future)
                s['train_prefix_aggregate_future_mse'] = float(fr.mean())
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        agg[kk] = agg.get(kk, 0.0) + vv
                aggn += 1

        batch_loss = batch_loss / len(channels)
        batch_loss.backward()
        enc_gn += grad_norm(params)
        sc_gn += grad_norm(list(set_conditioner.parameters())
                           + (list(metric.parameters()) if metric is not None else []))
        gn_n += 1
        cli.optimizer.step()
        tot_loss += float(batch_loss.detach())
        nb += 1

    out = {'train_choice_ce': tot_loss / max(nb, 1),
           'encoder_grad_norm': enc_gn / max(gn_n, 1),
           'score_layer_grad_norm': sc_gn / max(gn_n, 1)}
    out.update({f'train_{k}': v / max(aggn, 1) for k, v in agg.items()})
    return out


@torch.no_grad()
def eval_epoch(exp, args, host, model, set_conditioner, metric, cli,
               loader, channels, device, collect_stepwise=False):
    """Free-running inference (spec S15) + the teacher-forced-state internal
    diagnostics, computed separately so they are never conflated."""
    model.train(False)
    set_conditioner.train(False)
    if metric is not None:
        metric.train(False)

    fr_sum, fr_n = 0.0, 0
    ce_sum, ce_n = 0.0, 0
    step_sums, step_n = {}, {}
    stepwise_rows = []

    for batch_x, batch_y, batch_start_idx in _iter_batches(loader, getattr(cli, 'limit_batches', 0)):
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

            # ---- PRIMARY: free-running, no oracle information at all ----
            _, _, picks, _ = run_sequence(
                z_q, E, cand_mask, set_conditioner, metric, w_host,
                futures, query_future, cli.target, cli.prefix_policy,
                cli.tau_choice, cli.top_k, cli.chunk_size, free_running=True)
            fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                   query_future, cli.tau_topk)
            fr_sum += float(fr.sum())
            fr_n += int(fr.numel())

            # ---- stepwise metrics along the FREE-RUNNING trajectory ----
            selected = torch.zeros_like(cand_mask)
            prev_agg = None
            for t in range(cli.top_k):
                valid_now = cand_mask & ~selected
                if t == 0:
                    h_t = z_q
                else:
                    m = E[picks[:, :t]].mean(dim=1)
                    h_t = set_conditioner(z_q, m)
                u_hat = arm_score(h_t, E, metric)
                if cli.target == 'greedy_set':
                    u_target = greedy_set_utility(picks[:, :t], w_host, futures,
                                                  query_future, cli.chunk_size)
                else:
                    u_target = individual_utility(futures, query_future)
                neg_inf = torch.finfo(u_hat.dtype).min / 4
                oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                model_idx = picks[:, t]
                s = step_rank_diagnostics(u_hat, u_target, valid_now, oracle_idx,
                                          model_idx, futures, query_future)
                cur = free_running_aggregate_future_mse(
                    picks[:, :t + 1], host_scores, futures, query_future, cli.tau_topk).mean()
                s['current_aggregate_mse'] = float(cur)
                s['marginal_aggregate_improvement'] = (
                    float(prev_agg - cur) if prev_agg is not None else float('nan'))
                prev_agg = cur
                loss_t, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, cli.tau_choice)
                s['choice_ce'] = float(loss_t)
                ce_sum += float(loss_t)
                ce_n += 1
                for kk, vv in s.items():
                    if isinstance(vv, float) and vv == vv:
                        key = f't{t + 1}::{kk}'
                        step_sums[key] = step_sums.get(key, 0.0) + vv
                        step_n[key] = step_n.get(key, 0) + 1
                selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    out = {'free_running_aggregate_future_mse': fr_sum / max(fr_n, 1),
           'val_choice_ce': ce_sum / max(ce_n, 1)}
    per_step = {kk: step_sums[kk] / max(step_n[kk], 1) for kk in step_sums}
    if collect_stepwise:
        for t in range(cli.top_k):
            row = {'step': t + 1}
            for kk, vv in per_step.items():
                if kk.startswith(f't{t + 1}::'):
                    row[kk.split('::', 1)[1]] = vv
            stepwise_rows.append(row)
    # overall (mean over steps) versions of the headline internal metrics
    for name in ('oracle_action_acc', 'expert_rank_median', 'expert_regret_mean',
                 'expert_containment_at_10', 'ndcg_at_10', 'spearman',
                 'selected_individual_future_mse'):
        vals = [per_step[k] for k in per_step if k.endswith('::' + name)]
        out[name] = sum(vals) / max(len(vals), 1) if vals else float('nan')
    return out, stepwise_rows


# --------------------------------------------------------------------------
# frozen Stage-2 host score provider
# --------------------------------------------------------------------------
class HostScorer:
    """The Stage-2 host's own frozen retrieval score, used ONLY as the fixed
    exogenous aggregation weighting (module docstring, decision 1). No host
    parameter is ever updated and no gradient flows through it."""

    def __init__(self, stage2_ckpt, device):
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
        self.top_k = int(self.args.top_k)

    @torch.no_grad()
    def scores(self, batch_x, c, cand_mask):
        z_q = self.model._branch_embedding(batch_x, c, c)
        z_mem = self.model._branch_memory(self.exp.key_bank, c, 0, c, z_q.dtype, self.device)
        s = (self.score_fn(z_q, z_mem) if self.score_fn is not None
             else torch.matmul(z_q, z_mem.transpose(0, 1)))
        return s.masked_fill(~cand_mask, torch.finfo(s.dtype).min / 4)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True,
                    help='EXISTING Stage-1 checkpoint -- ARGS ONLY, weights never loaded')
    ap.add_argument('--stage2_host', required=True,
                    help='Stage-2 host checkpoint; supplies the FIXED aggregation weighting')
    ap.add_argument('--target', choices=ARMS, required=True)
    ap.add_argument('--prefix_policy', choices=PREFIXES, required=True)
    ap.add_argument('--scorer', choices=SCORERS, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True, help='e.g. ETTh1_96')
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_factorial_e2e')
    ap.add_argument('--out_dir', default='results/track_a_factorial_e2e')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    # Training hyperparameters default to None and are RESOLVED FROM THE
    # STAGE-1 REFERENCE ARGS, never from the Stage-2 host. A CLI value
    # overrides and is recorded as such in the fingerprint's provenance block.
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--train_epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--probe_size', type=int, default=256)
    ap.add_argument('--test_epochs', default='1,5,10',
                    help='epochs at which the TEST aggregate MSE is also measured (spec S17)')
    ap.add_argument('--limit_batches', type=int, default=0,
                    help='SMOKE ONLY: cap batches per split. 0 = full run.')
    cli = ap.parse_args()

    # ---- hyperparameter provenance (spec: Stage-1 config is the source) ----
    ref_args = dict(torch.load(cli.reference_ckpt, map_location='cpu')['args'])
    host_ck = torch.load(cli.stage2_host, map_location='cpu')
    host_args = dict(host_ck['args'])
    hp_source = {}

    def resolve(name, ref_key, ref_default, ref_default_note):
        """CLI > Stage-1 reference args > documented Stage-1 default."""
        cli_val = getattr(cli, name)
        if cli_val is not None:
            hp_source[name] = {'value': cli_val, 'source': 'CLI override'}
        elif ref_key in ref_args and ref_args[ref_key] is not None:
            setattr(cli, name, ref_args[ref_key])
            hp_source[name] = {'value': ref_args[ref_key],
                               'source': f'Stage-1 reference args["{ref_key}"]'}
        else:
            setattr(cli, name, ref_default)
            hp_source[name] = {'value': ref_default, 'source': ref_default_note}
        return getattr(cli, name)

    resolve('learning_rate', 'learning_rate', 1e-3, 'Stage-1 default')
    resolve('batch_size', 'batch_size', 32, 'Stage-1 default')
    resolve('train_epochs', 'train_epochs', 10, 'Stage-1 default')
    resolve('patience', 'patience', 5, 'Stage-1 default')
    resolve('seed', 'seed', 0, 'Stage-1 default')
    resolve('top_k', 'top_k', 10, 'Stage-1 default')
    # weight_decay is absent from Stage-1 args: Exp_Stage1_Relation._select_optimizer
    # builds Adam without it, i.e. torch's default 0.0.
    resolve('weight_decay', 'weight_decay', 0.0,
            'Stage-1 exp/exp_stage1_relation.py::_select_optimizer -> Adam default (0.0)')
    hp_source['optimizer'] = {
        'value': 'Adam',
        'source': 'Stage-1 exp/exp_stage1_relation.py::_select_optimizer'}
    # TWO different temperatures with DIFFERENT provenance. They happen to be
    # equal (0.1) in every cell, but they are recorded separately on purpose.
    cli.tau_choice = float(ref_args.get('tau_topk', 0.1))
    cli.tau_topk = float(host_args['tau_topk'])
    hp_source['tau_choice'] = {'value': cli.tau_choice,
                               'source': 'Stage-1 reference args["tau_topk"] (Choice-CE temperature)'}
    hp_source['tau_aggregation'] = {'value': cli.tau_topk,
                                    'source': 'Stage-2 host args["tau_topk"] (aggregation weighting only)'}
    hp_source['lr_schedule'] = {
        'value': 'constant',
        'source': ("DIVERGENCE FROM Stage-1 reference (lradj="
                   f"{ref_args.get('lradj')!r}, applied by Exp_Stage1_Relation.train). "
                   "This experiment keeps LR constant, matching every prior scratch/"
                   "Oracle-Choice script in this project (train_oracle_choice01, "
                   "train_oracle_scratch01, train_oracle_rank_gain01). Identical for "
                   "all 8 arms, so it cannot bias the factorial contrast.")}

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = HostScorer(cli.stage2_host, device)
    if int(host.top_k) != int(cli.top_k):
        raise SystemExit(f'[ABORT] host top_k={host.top_k} != Stage-1 top_k={cli.top_k}')
    if abs(host.tau_topk - cli.tau_choice) > 1e-12:
        print(f'[factorial_e2e01] NOTE: Stage-1 tau_choice={cli.tau_choice} differs from '
              f'host aggregation tau={host.tau_topk}; both recorded separately.')

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': cli.seed,
        'top_k': cli.top_k, 'tau_topk': cli.tau_choice,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)

    # ---- SCRATCH shared initial weights (spec S3) ----
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
    metric = None
    cos_dev = 0.0
    if cli.scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine',
                                 layer_norm=False).to(device)
        cos_dev = float(cosine_init_deviation(metric))
        if cos_dev > 1e-5:
            raise SystemExit(f'[ABORT] asymmetric scorer is not identity at init: '
                             f'cosine_init_deviation={cos_dev:.3e}')

    for p in model.parameters():
        p.requires_grad_(True)
    params = (list(model.parameters()) + list(set_conditioner.parameters())
              + (list(metric.parameters()) if metric is not None else []))
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate,
                                     weight_decay=cli.weight_decay)

    channels = list(range(int(args.enc_in)))
    _, train_loader = exp._get_data(flag='train')
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe_x = exp.memory_x[:cli.probe_size]

    fingerprint = {
        'cell': cli.cell, 'arm': cli.arm_name,
        'axis_oracle': cli.target, 'axis_prefix': cli.prefix_policy, 'axis_score': cli.scorer,
        'encoder_init_sha256': init_sha, 'scratch_init': True, 'encoder_trainable': True,
        'stage2_host': cli.stage2_host, 'reference_ckpt': cli.reference_ckpt,
        'seq_len': int(args.seq_len), 'pred_len': int(args.pred_len), 'enc_in': int(args.enc_in),
        'd_model': d_model, 'top_k': cli.top_k, 'tau_topk': cli.tau_topk,
        'tau_choice': cli.tau_choice, 'candidate_mask': getattr(args, 'candidate_mask', None),
        'relation_value_space': getattr(args, 'relation_value_space', None),
        'optimizer': 'Adam', 'learning_rate': cli.learning_rate,
        'weight_decay': cli.weight_decay, 'batch_size': cli.batch_size,
        'train_epochs': cli.train_epochs, 'patience': cli.patience, 'seed': cli.seed,
        'hyperparameter_provenance': hp_source,
        'stage1_reference_args_subset': {k: ref_args.get(k) for k in (
            'learning_rate', 'batch_size', 'train_epochs', 'patience', 'seed',
            'top_k', 'tau_topk', 'lradj', 'd_model', 'candidate_mask',
            'relation_value_space', 'relation_top_n')},
        'stage2_host_args_subset': {k: host_args.get(k) for k in (
            'learning_rate', 'batch_size', 'train_epochs', 'patience', 'seed',
            'top_k', 'tau_topk', 'fusion_mode', 'gate_mode', 'fixed_lambda',
            'relation_top_n', 'candidate_mask', 'relation_value_space')},
        'stage2_role': 'FROZEN HOST -- supplies the aggregation weighting only; '
                       'never trained, never used as a Stage-1 hyperparameter source',
        'loss': 'oracle_choice_cross_entropy',
        'aggregation': 'softmax(host_score/tau_topk) over the K picks (host-fixed, exogenous)',
        'checkpoint_criterion': 'min val free_running_aggregate_future_mse',
        'cosine_init_deviation': cos_dev,
        'memory_bank_shape': list(exp.memory_x.shape),
        'n_channels': len(channels),
        'limit_batches': cli.limit_batches,
        'is_smoke_run': bool(cli.limit_batches),
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(
        json.dumps(fingerprint, indent=2))
    print(f'[factorial_e2e01] {cli.cell}/{cli.arm_name} init_sha={init_sha[:16]} '
          f'cosine_init_deviation={cos_dev:.3e}')

    test_at = {int(x) for x in cli.test_epochs.split(',') if x}
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    stepwise_best = []
    t0 = time.time()

    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, host, model, set_conditioner, metric, cli,
                         train_loader, channels, device)
        va, _ = eval_epoch(exp, args, host, model, set_conditioner, metric, cli,
                           val_loader, channels, device)
        with torch.no_grad():
            rep = representation_diagnostics(encode_raw(model, probe_x, 0))
        rep['encoder_param_displacement'] = param_displacement(model, init_state)

        row = {'epoch': epoch, **tr,
               'val_free_running_aggregate_future_mse': va['free_running_aggregate_future_mse'],
               **{f'val_{k}': v for k, v in va.items() if k != 'free_running_aggregate_future_mse'},
               **{f'rep_{k}': v for k, v in rep.items() if not isinstance(v, list)}}

        if epoch in test_at:
            te, _ = eval_epoch(exp, args, host, model, set_conditioner, metric, cli,
                               test_loader, channels, device)
            row['test_free_running_aggregate_future_mse'] = te['free_running_aggregate_future_mse']

        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(),
                   'set_conditioner_state_dict': set_conditioner.state_dict(),
                   'metric_state_dict': metric.state_dict() if metric is not None else None,
                   'args': vars(args), 'epoch': epoch, 'fingerprint': fingerprint,
                   'val_free_running_aggregate_future_mse': row['val_free_running_aggregate_future_mse']}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')

        if row['val_free_running_aggregate_future_mse'] < best['val']:
            best = {'val': row['val_free_running_aggregate_future_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')

        print(f"[factorial_e2e01] {cli.arm_name} epoch {epoch} "
              f"train_ce={tr['train_choice_ce']:.5f} "
              f"val_fr_agg={row['val_free_running_aggregate_future_mse']:.6f} "
              f"eff_rank={rep['effective_rank']:.3f} "
              f"pair_cos={rep['mean_pairwise_cosine']:.4f} "
              f"enc_gn={tr['encoder_grad_norm']:.5f}")

        if epoch - best['epoch'] >= cli.patience:
            print(f'[factorial_e2e01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    # ---- best checkpoint: full test evaluation + stepwise ----
    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    set_conditioner.load_state_dict(bl['set_conditioner_state_dict'])
    if metric is not None:
        metric.load_state_dict(bl['metric_state_dict'])
    te, stepwise_best = eval_epoch(exp, args, host, model, set_conditioner, metric, cli,
                                   test_loader, channels, device, collect_stepwise=True)

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
    with open(cell_dir / f'representation_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = ['epoch'] + [k for k in sorted(epoch_rows[0]) if k.startswith('rep_')]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in epoch_rows:
            w.writerow({k: r.get(k) for k in keys})

    matched = {f'epoch{e}': next((r.get('val_free_running_aggregate_future_mse')
                                  for r in epoch_rows if r['epoch'] == e), None)
               for e in sorted(test_at)}
    matched_test = {f'epoch{e}': next((r.get('test_free_running_aggregate_future_mse')
                                       for r in epoch_rows if r['epoch'] == e), None)
                    for e in sorted(test_at)}
    summary = {
        'cell': cli.cell, 'arm': cli.arm_name,
        'oracle': cli.target, 'prefix': cli.prefix_policy, 'score': cli.scorer,
        'best_epoch': best['epoch'],
        'best_val_free_running_aggregate_future_mse': best['val'],
        'best_ckpt_test_free_running_aggregate_future_mse': te['free_running_aggregate_future_mse'],
        'final_epoch': epoch_rows[-1]['epoch'],
        'final_val_free_running_aggregate_future_mse':
            epoch_rows[-1]['val_free_running_aggregate_future_mse'],
        'matched_epoch_val': matched, 'matched_epoch_test': matched_test,
        'test_internal': {k: v for k, v in te.items()},
        'encoder_init_sha256': init_sha, 'cosine_init_deviation': cos_dev,
        'final_effective_rank': epoch_rows[-1].get('rep_effective_rank'),
        'final_mean_pairwise_cosine': epoch_rows[-1].get('rep_mean_pairwise_cosine'),
        'final_encoder_param_displacement': epoch_rows[-1].get('rep_encoder_param_displacement'),
        'wall_clock_seconds': time.time() - t0,
        'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
    print(f"[factorial_e2e01] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
          f"best_val_fr_agg={best['val']:.6f} "
          f"test_fr_agg={te['free_running_aggregate_future_mse']:.6f}")


if __name__ == '__main__':
    main()
