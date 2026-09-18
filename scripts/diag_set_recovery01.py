#!/usr/bin/env python3
"""EXP-SET-RECOVERY01 -- Prefix Oracle Intervention diagnostic.

For a trained Stage-1 checkpoint (Set or Individual, any loss), builds a
HYBRID free-running trajectory: for step t < m the next candidate is the
Greedy Set Oracle's argmax (recomputed from the ACTUAL hybrid prefix so
far, never a cached/original-trajectory value); for t >= m it is the
trained retriever's own argmax. Both branches mask out already-selected
candidates. `m=0` must be index-identical to a plain free-running rollout;
`m=K` (K=top_k) must be index-identical to a full-Oracle rollout with the
SAME on-policy-style "recompute oracle from the actual prefix every step"
semantics `train_factorial_e2e01.py::run_sequence`'s `prefix_policy='tf'`
already uses (oracle_next is that function's own oracle-based next-pick,
unmodified).

Reuses, unmodified: `encode_raw`, `arm_score`, `greedy_set_utility`,
`individual_utility`, `HostScorer`, `candidate_weights`,
`free_running_aggregate_future_mse` (all from `train_factorial_e2e01.py`).
Does not reuse `run_sequence` directly -- its API has no notion of a
per-step oracle/student switch at an arbitrary `m` -- but every
oracle/student decision inside this new function calls the SAME
utility/score functions `run_sequence` calls, so this is not a duplicated
reimplementation of the Oracle or student score, only new step-selection
control flow.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score, greedy_set_utility, individual_utility

M_VALUES = (0, 1, 3, 5, 8, 10)


@torch.no_grad()
def hybrid_prefix_intervention(z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                               m, top_k, chunk_size, target='greedy_set'):
    """Returns dict with 'picks' [B,K], and per-step diagnostics list of
    dicts (one per step): selector ('oracle'|'student'), u_hat, u_target,
    picked idx, oracle top-1 idx, valid_now mask.

    `target='greedy_set'` uses `greedy_set_utility` (prefix-dependent,
    recomputed every step from the REAL hybrid prefix); `target='individual'`
    uses `individual_utility` (prefix-invariant, recomputed each step for
    symmetry/testability even though its value never actually changes).
    """
    bsz = z_q.size(0)
    device = z_q.device
    selected = torch.zeros_like(cand_mask)
    picks = []
    neg_inf = torch.finfo(z_q.dtype).min / 4
    steps = []

    for t in range(top_k):
        if t == 0:
            h_t = z_q
        else:
            prefix = torch.stack(picks, dim=1)
            h_t = set_conditioner(z_q, E[prefix].mean(dim=1))
        u_hat = arm_score(h_t, E, None)
        valid_now = cand_mask & ~selected

        prefix_now = (torch.stack(picks, dim=1) if picks
                     else torch.zeros(bsz, 0, dtype=torch.long, device=device))
        if target == 'greedy_set':
            u_target = greedy_set_utility(prefix_now, w_host, futures, query_future, chunk_size)
        else:
            u_target = individual_utility(futures, query_future)

        oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        student_idx = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        use_oracle = t < m
        a_t = oracle_idx if use_oracle else student_idx

        steps.append({'step': t, 'selector': 'oracle' if use_oracle else 'student',
                      'u_hat': u_hat.detach(), 'u_target': u_target.detach(),
                      'picked_idx': a_t.detach(), 'oracle_idx': oracle_idx.detach(),
                      'student_idx': student_idx.detach(), 'valid_now': valid_now})
        picks.append(a_t)
        selected = selected.scatter(1, a_t.unsqueeze(-1), True)

    return {'picks': torch.stack(picks, dim=1), 'steps': steps}


@torch.no_grad()
def single_step_correction(z_q, E, cand_mask, set_conditioner, w_host, futures, query_future,
                           correct_step, top_k, chunk_size, target='greedy_set'):
    """Section 3: Oracle correction at EXACTLY `correct_step` (0..K-1),
    learned retriever everywhere else. Prefix after the correction
    includes the ACTUAL corrected candidate (recomputed every step, never
    substituted post-hoc)."""
    bsz = z_q.size(0)
    device = z_q.device
    selected = torch.zeros_like(cand_mask)
    picks = []
    neg_inf = torch.finfo(z_q.dtype).min / 4
    steps = []

    for t in range(top_k):
        if t == 0:
            h_t = z_q
        else:
            prefix = torch.stack(picks, dim=1)
            h_t = set_conditioner(z_q, E[prefix].mean(dim=1))
        u_hat = arm_score(h_t, E, None)
        valid_now = cand_mask & ~selected

        prefix_now = (torch.stack(picks, dim=1) if picks
                     else torch.zeros(bsz, 0, dtype=torch.long, device=device))
        if target == 'greedy_set':
            u_target = greedy_set_utility(prefix_now, w_host, futures, query_future, chunk_size)
        else:
            u_target = individual_utility(futures, query_future)

        oracle_idx = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        student_idx = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
        use_oracle = (t == correct_step)
        a_t = oracle_idx if use_oracle else student_idx

        steps.append({'step': t, 'selector': 'oracle' if use_oracle else 'student',
                      'u_hat': u_hat.detach(), 'u_target': u_target.detach(),
                      'picked_idx': a_t.detach(), 'oracle_idx': oracle_idx.detach(),
                      'student_idx': student_idx.detach(), 'valid_now': valid_now})
        picks.append(a_t)
        selected = selected.scatter(1, a_t.unsqueeze(-1), True)

    return {'picks': torch.stack(picks, dim=1), 'steps': steps}


def free_running_aggregate_future_mse(picks, host_scores, futures, query_future, tau_topk):
    """Re-export for convenience -- IDENTICAL to
    train_factorial_e2e01.free_running_aggregate_future_mse (imported
    there, not reimplemented; kept as a thin alias so callers of this
    module don't need a second import)."""
    from scripts.train_factorial_e2e01 import \
        free_running_aggregate_future_mse as _fr
    return _fr(picks, host_scores, futures, query_future, tau_topk)


def recovery_rate(e_m0, e_m, e_m10):
    """Recovery(m) = (E_{m=0} - E_m) / (E_{m=0} - E_{m=10}).
    Returns (value, issue) -- `issue` is a string reason when the
    denominator is near-zero or negative (spec section 5: never force a
    number out of a degenerate denominator)."""
    denom = e_m0 - e_m10
    if abs(denom) < 1e-9:
        return None, f'[ISSUE] denominator E_m0-E_m10={denom:.3e} ~ 0 -- recovery undefined'
    if denom < 0:
        return None, f'[ISSUE] denominator E_m0-E_m10={denom:.3e} < 0 -- m=10 worse than m=0, recovery undefined'
    return (e_m0 - e_m) / denom, None


def combination_redundancy(picked_futures):
    """picked_futures: [B, K, pred_len] (the K selected candidates' own
    future values, delta or absolute space consistently). Returns
    mean/max pairwise cosine similarity and mean pairwise MSE, per query,
    averaged over the batch."""
    B, K, H = picked_futures.shape
    flat = picked_futures.reshape(B, K, H)
    norm = torch.nn.functional.normalize(flat, dim=-1)
    cos = torch.matmul(norm, norm.transpose(1, 2))  # [B,K,K]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=flat.device).unsqueeze(0).expand(B, -1, -1)
    mean_cos = cos[off_diag].view(B, -1).mean(dim=-1)
    max_cos = cos.masked_fill(~off_diag, -2.0).view(B, -1).max(dim=-1).values
    diff = flat.unsqueeze(2) - flat.unsqueeze(1)  # [B,K,K,H]
    mse = (diff ** 2).mean(dim=-1)  # [B,K,K]
    mean_mse = mse[off_diag].view(B, -1).mean(dim=-1)
    return {'mean_pairwise_cosine': mean_cos, 'max_pairwise_cosine': max_cos,
           'mean_pairwise_mse': mean_mse}
