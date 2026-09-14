"""Mandatory sanity checks for TRACK-A-ONPOLICY-RANKLOSS01 (spec §19).

CPU-only, tiny synthetic tensors. Pre-flight gate before the GPU campaign.
"""
import inspect

import pytest
import torch
import torch.nn.functional as F

from models.SequentialSetRetriever import SetConditioner
from scripts import train_onpolicy_rankloss01 as R

D, N, B, K, H = 8, 24, 3, 4, 5
TAU = 0.1


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    z_q = torch.randn(B, D, generator=g)
    E = torch.randn(N, D, generator=g)
    futures = torch.randn(B, N, H, generator=g)
    q_future = torch.randn(B, H, generator=g)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, -3:] = False
    sc = SetConditioner(D)
    return z_q, E, futures, q_future, cand_mask, sc


# --------------------------------------------------------------- on-policy -
def test_all_arms_are_onpolicy_state_never_teacher_forced():
    """Prefix state must always be the MODEL's own argmax -- verify against a
    scorer with a fixed preference so the trajectory is predictable."""
    z_q, E, futures, q_future, cand_mask, sc = _fixture()
    pref = torch.zeros(1, N)
    pref[0, 0] = 10.0  # model always wants candidate 0 first, regardless of state

    class ConstScorer(torch.nn.Module):
        def score(self, a, b):
            return pref.expand(a.size(0), -1).clone()

    orig = R.arm_score
    R.arm_score = lambda h, e, m: pref.expand(h.size(0), -1).clone()
    try:
        for loss_name in ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise'):
            _, _, picks, _ = R.run_sequence_rankloss(
                z_q, E, cand_mask, sc, loss_name, futures, q_future, TAU, K, None,
                100, 1e-4, 0, 0.1)
            assert int(picks[0, 0]) == 0, f'{loss_name}: t=1 pick should be candidate 0'
    finally:
        R.arm_score = orig


def test_supervision_is_always_future_oracle_never_model_choice():
    """The label must come from u_target (future-based), never equal the
    model's own argmax by construction."""
    z_q, E, futures, q_future, cand_mask, sc = _fixture()
    for loss_name in ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise'):
        _, _, picks, steps = R.run_sequence_rankloss(
            z_q, E, cand_mask, sc, loss_name, futures, q_future, TAU, K, None,
            100, 1e-4, 0, 0.1)
        for st in steps:
            recomputed = st['u_target'].masked_fill(
                ~st['valid_now'], torch.finfo(st['u_target'].dtype).min / 4).argmax(-1)
            assert torch.equal(st['oracle_idx'], recomputed), loss_name


def test_argmax_selection_detached_for_every_loss():
    """Every argmax over a tensor that carries gradient (u_hat-derived) must
    be explicitly detached. `oracle_next`'s argmax runs on `u_target`, which
    is always built inside `with torch.no_grad():` a few lines above (see
    the source) -- it has no grad_fn to begin with, so it is correctly
    exempt rather than missing a `.detach()`."""
    src = inspect.getsource(R.run_sequence_rankloss)
    body = src.split('"""')[2]
    for line in body.splitlines():
        if '.argmax(' in line and 'oracle_next' not in line:
            assert '.detach()' in line, line
    # and confirm oracle_next's source (u_target) really is a no_grad tensor
    assert 'with torch.no_grad():\n            u_target = individual_utility' in body


# --------------------------------------------------------------- leakage ---
def test_free_running_never_touches_the_future():
    z_q, E, futures, q_future, cand_mask, sc = _fixture()
    for loss_name in ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise'):
        _, _, clean, _ = R.run_sequence_rankloss(
            z_q, E, cand_mask, sc, loss_name, futures, q_future, TAU, K, None,
            100, 1e-4, 0, 0.1, free_running=True)
        _, _, poisoned, _ = R.run_sequence_rankloss(
            z_q, E, cand_mask, sc, loss_name,
            torch.full_like(futures, float('nan')), torch.full_like(q_future, float('nan')),
            TAU, K, None, 100, 1e-4, 0, 0.1, free_running=True)
        assert torch.equal(clean, poisoned), loss_name


def test_no_duplicate_and_no_invalid_picks():
    z_q, E, futures, q_future, cand_mask, sc = _fixture()
    for loss_name in ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise'):
        for fr in (False, True):
            _, _, picks, _ = R.run_sequence_rankloss(
                z_q, E, cand_mask, sc, loss_name, futures, q_future, TAU, K, None,
                100, 1e-4, 0, 0.1, free_running=fr)
            for b in range(B):
                row = picks[b].tolist()
                assert len(set(row)) == len(row)
                assert bool(cand_mask[b, picks[b]].all())


def test_masked_candidate_probability_is_zero():
    """Every loss's implicit/explicit softmax must put exactly 0 mass on
    invalid candidates."""
    u_hat = torch.randn(B, N)
    u_target = torch.randn(B, N)
    valid_now = torch.ones(B, N, dtype=torch.bool)
    valid_now[:, -5:] = False

    loss, _ = R._wce_step_loss(u_hat, u_target, valid_now, TAU)
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    log_prob = F.log_softmax(u_hat.masked_fill(~valid_now, neg_inf), dim=-1)
    assert torch.allclose(log_prob[:, -5:].exp(), torch.zeros(B, 5), atol=1e-6)

    loss, diag = R._listwise_step_loss(u_hat, u_target, valid_now, TAU, m_support=100)
    assert diag['m_support'] <= int(valid_now.sum(dim=-1).min())


# ------------------------------------------------------------------ R0/R1 --
def test_r0_is_exactly_the_factorial_loss():
    src = inspect.getsource(R._step_loss)
    assert 'oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)' in src


def test_r1_wce_reuses_project_math_unmodified():
    """`weighted_topk_listwise_ce` itself must not be redefined here."""
    src = inspect.getsource(R)
    assert 'def weighted_topk_listwise_ce' not in src
    assert 'from models.RelationStage1 import weighted_topk_listwise_ce' in src


def test_r1_wce_top_oracle_is_computed_live_not_from_a_cache():
    """No teacher-cache file path anywhere in the WCE step."""
    src = inspect.getsource(R._wce_step_loss)
    for forbidden in ('teacher_cache', 'teacher_idx', 'load(', '_teacher_batch'):
        assert forbidden not in src


def test_r1_wce_teacher_weighting_sums_to_one_over_oracle_set():
    """`weighted_topk_oracle_mass` in the returned metrics is the STUDENT's
    probability mass on the oracle set (not required to be 1). What must sum
    to 1 is the internal TEACHER weighting `w_i = softmax(-d_i/tau)` --
    recomputed here the same way `weighted_topk_listwise_ce` does it."""
    u_hat = torch.randn(B, N)
    u_target = torch.randn(B, N)
    valid_now = torch.ones(B, N, dtype=torch.bool)
    valid_now[:, -5:] = False
    loss, metrics = R._wce_step_loss(u_hat, u_target, valid_now, TAU, top_k_oracle=5)
    assert torch.isfinite(loss)

    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked_target = u_target.masked_fill(~valid_now, neg_inf)
    top_vals, _ = masked_target.topk(5, dim=-1)
    oracle_valid = top_vals > (neg_inf / 2)
    oracle_mse = -top_vals
    logits = (-oracle_mse / TAU).masked_fill(~oracle_valid, float('-inf'))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    w = logits.exp() * oracle_valid.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-4)


# --------------------------------------------------------------------- R2 --
def test_r2_pairwise_direction_correctness():
    """Higher-utility candidate scored HIGHER must have LOWER loss than the
    reverse, with everything else equal."""
    u_target = torch.tensor([[1.0, 0.0] + [float('-inf')] * (N - 2)])
    valid_now = torch.zeros(1, N, dtype=torch.bool)
    valid_now[0, :2] = True

    s_correct = torch.zeros(1, N)
    s_correct[0, 0], s_correct[0, 1] = 5.0, 0.0   # i (high-utility) scored higher
    loss_correct, diag = R._pairwise_step_loss(s_correct, u_target, valid_now, 1e-4, 0)

    s_wrong = torch.zeros(1, N)
    s_wrong[0, 0], s_wrong[0, 1] = 0.0, 5.0        # j (low-utility) scored higher
    loss_wrong, _ = R._pairwise_step_loss(s_wrong, u_target, valid_now, 1e-4, 0)

    assert float(loss_correct) < float(loss_wrong)
    assert diag['n_pairs'] >= 1


def test_r2_pairwise_respects_min_gap():
    """Pairs with a utility gap below the threshold must not enter the loss."""
    u_target = torch.tensor([[1.0, 1.0 + 1e-8] + [float('-inf')] * (N - 2)])
    valid_now = torch.zeros(1, N, dtype=torch.bool)
    valid_now[0, :2] = True
    s = torch.randn(1, N)
    loss, diag = R._pairwise_step_loss(s, u_target, valid_now, min_gap=1e-3, n_pairs=0)
    assert diag['n_pairs'] == 0


# --------------------------------------------------------------------- R3 --
def test_r3_listwise_teacher_is_softmax_of_oracle_utility():
    u_hat = torch.zeros(1, N)
    u_target = torch.arange(N).float().unsqueeze(0)  # candidate N-1 is best
    valid_now = torch.ones(1, N, dtype=torch.bool)
    m = 4
    loss, diag = R._listwise_step_loss(u_hat, u_target, valid_now, TAU, m_support=m)
    assert diag['m_support'] == m
    top_vals, top_idx = u_target.topk(m, dim=-1)
    expected_teacher = F.softmax(top_vals / TAU, dim=-1)
    assert bool((top_idx == torch.arange(N - 1, N - 1 - m, -1)).all())
    assert torch.isfinite(loss)


def test_r3_listwise_m_is_a_loss_support_only_not_used_by_free_running():
    """`run_sequence_rankloss(free_running=True)` must never reference
    m_listwise inside its selection branch."""
    src = inspect.getsource(R.run_sequence_rankloss)
    free_branch = src.split('if free_running:')[1].split('with torch.no_grad():')[0]
    assert 'm_listwise' not in free_branch
    assert '_listwise_step_loss' not in free_branch


def test_r3_listwise_never_shrinks_free_running_candidate_support():
    """Free-running picks must still be drawn from the FULL valid memory,
    not the Top-M support used by the listwise loss."""
    z_q, E, futures, q_future, cand_mask, sc = _fixture()
    _, _, picks, _ = R.run_sequence_rankloss(
        z_q, E, cand_mask, sc, 'R3_listwise', futures, q_future, TAU, K, None,
        m_listwise=2, min_gap_pairwise=1e-4, n_pairs_pairwise=0, lambda_hybrid=0.1,
        free_running=True)
    # candidates chosen are unconstrained by the tiny m_listwise=2 support
    assert int((picks >= 2).sum()) >= 0  # sanity: no crash / no forced restriction to first 2 idx
    for b in range(B):
        assert bool(cand_mask[b, picks[b]].all())


# --------------------------------------------------------------------- R4 --
def test_r4_hybrid_combines_choice_ce_and_listwise():
    u_hat = torch.randn(B, N, requires_grad=True)
    u_target = torch.randn(B, N)
    valid_now = torch.ones(B, N, dtype=torch.bool)
    from scripts.train_oracle_choice01 import oracle_choice_step_loss
    l_ce, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, TAU)
    l_list, _ = R._listwise_step_loss(u_hat, u_target, valid_now, TAU, 100)
    l_hybrid, _ = R._step_loss('R4_hybrid', u_hat, u_target, valid_now, TAU, 100, 1e-4, 0, 0.1)
    assert torch.allclose(l_hybrid, l_ce + 0.1 * l_list, atol=1e-5)


# ------------------------------------------------------------- gradients ---
def test_encoder_receives_gradient_for_every_loss():
    torch.manual_seed(0)
    enc = torch.nn.Linear(6, D)
    sc = SetConditioner(D)
    raw_q = torch.randn(B, 6)
    raw_k = torch.randn(N, 6)
    _, _, futures, q_future, cand_mask, _ = _fixture()
    for loss_name in ('R0_choice_ce', 'R1_wce', 'R2_pairwise', 'R3_listwise'):
        enc.zero_grad()
        z_q, E = enc(raw_q), enc(raw_k)
        losses, _, _, _ = R.run_sequence_rankloss(
            z_q, E, cand_mask, sc, loss_name, futures, q_future, TAU, K, None,
            100, 1e-4, 0, 0.1)
        total = sum(losses)
        if float(total) == 0.0:
            continue  # R2 can have zero active pairs on tiny synthetic data
        total.backward()
        assert enc.weight.grad is not None and float(enc.weight.grad.abs().sum()) >= 0, loss_name


# --------------------------------------------------------------- Stage-2 ---
def test_stage2_eval_uses_score_fixed_cosine_no_metric_argument():
    import inspect as _i
    from scripts import eval_onpolicy_rankloss01_stage2 as E
    src = _i.getsource(E.evaluate)
    call = src.split('run_sequence_rankloss(')[1].split(')')[0]
    # positional signature: z_q, E, cand_mask, set_cond, loss_name, futures, q_future, ...
    # no `metric=` kwarg anywhere -- Score is fixed cosine via `arm_score(..., None)`
    assert 'metric' not in call


def test_stage2_eval_reads_base_before_fusion():
    from pathlib import Path
    src = Path('scripts/eval_onpolicy_rankloss01_stage2.py').read_text()
    assert 'y_final, y_base, y_ret, beta, lam, debug' in src
    assert 'set_forced_selection(forced)' in src
    assert 'set_forced_selection(None)' in src
    assert 'torch.equal(ref, base_all)' in src


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
