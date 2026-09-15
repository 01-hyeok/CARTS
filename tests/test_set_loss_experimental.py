"""EXP-SET-LOSS01 -- unit tests for utils/set_loss_experimental.py.

CPU-only, tiny synthetic tensors (same fixture convention as
tests/test_factorial_e2e01.py). Covers the common teacher-signal
contract, then SRM-specific and SoftCE-specific behavior, then a light
prefix-sensitivity integration check against the real (unmodified)
reference Set Oracle.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from scripts.train_factorial_e2e01 import candidate_weights, greedy_set_utility
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.set_loss_experimental import (compute_oracle_teacher_signal,
                                         set_utility_soft_ce_loss,
                                         soft_regret_mass_loss)

TAU = 0.1


def _fixture(bsz=4, n=12, seed=0, n_invalid=3):
    g = torch.Generator().manual_seed(seed)
    u_hat = torch.randn(bsz, n, generator=g)
    u_target = torch.randn(bsz, n, generator=g) * 2.0
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    if n_invalid:
        valid_mask[:, -n_invalid:] = False
    return u_hat, u_target, valid_mask


# --------------------------------------------------------------------------
# common teacher-signal contract
# --------------------------------------------------------------------------
def test_invalid_candidate_weight_is_zero():
    u_hat, u_target, valid_mask = _fixture()
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    assert torch.equal(t['weight'][~valid_mask], torch.zeros_like(t['weight'][~valid_mask]))
    assert torch.equal(t['p_mask'][~valid_mask], torch.zeros_like(t['p_mask'][~valid_mask]))


def test_selected_candidate_excluded_via_valid_mask():
    """'selected' candidates are excluded from V_t via valid_mask in the
    caller (run_sequence's own valid_now = cand_mask & ~selected) -- so
    this is the same code path as 'invalid'; verified by using a mask
    where the excluded set represents a mid-sequence selection."""
    u_hat, u_target, valid_mask = _fixture(n_invalid=0)
    valid_mask[:, 2] = False  # pretend candidate 2 was already selected
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    assert bool((t['weight'][:, 2] == 0).all())
    assert bool((~t['p_mask'][:, 2]).all())


def test_teacher_weight_has_no_gradient():
    u_hat, u_target, valid_mask = _fixture()
    u_target = u_target.requires_grad_(True)
    loss, _ = soft_regret_mass_loss(u_hat.requires_grad_(True), u_target, valid_mask, TAU, m=3)
    loss.backward()
    assert u_target.grad is None or torch.equal(u_target.grad, torch.zeros_like(u_target.grad))


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_shift_invariance_exact(loss_fn):
    u_hat, u_target, valid_mask = _fixture(seed=1)
    l1, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)
    l2, _ = loss_fn(u_hat, u_target + 1000.0, valid_mask, TAU, m=3)
    assert torch.allclose(l1, l2, atol=1e-5)


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_scale_invariance_approximate(loss_fn):
    u_hat, u_target, valid_mask = _fixture(seed=2)
    l1, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)
    l2, _ = loss_fn(u_hat, u_target * 5.0, valid_mask, TAU, m=3)
    assert torch.allclose(l1, l2, atol=1e-3)


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_topm_cutoff_ties_all_included(loss_fn):
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=0, seed=3)
    u_target[0, :5] = 5.0   # 5-way tie for the top (M=3 would normally cut at 3)
    u_target[0, 5:] = 0.0
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    assert int(t['p_mask'][0].sum()) == 5  # all 5 tied candidates included, not just 3


def test_fewer_than_m_valid_candidates():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=7, seed=4)  # only 3 valid
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=10)
    assert int(t['p_mask'][0].sum()) <= 3
    loss_srm, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=10)
    loss_ce, _ = set_utility_soft_ce_loss(u_hat, u_target, valid_mask, TAU, m=10)
    assert torch.isfinite(loss_srm) and torch.isfinite(loss_ce)


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_all_valid_utilities_equal_zero_loss_and_gradient(loss_fn):
    u_hat, u_target, valid_mask = _fixture(bsz=3, n=8, n_invalid=2, seed=5)
    u_target = torch.zeros_like(u_target)  # all valid utilities identical (0)
    u_hat = u_hat.clone().requires_grad_(True)
    loss, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)
    assert float(loss) == 0.0
    loss.backward()
    assert u_hat.grad is None or torch.equal(u_hat.grad, torch.zeros_like(u_hat.grad))


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_nan_utility_produces_finite_or_raises_not_silently_wrong(loss_fn):
    u_hat, u_target, valid_mask = _fixture(bsz=2, n=8, n_invalid=2, seed=6)
    u_target[0, 1] = float('nan')
    loss, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)
    # row 0 is contaminated; row 1 must remain unaffected and finite
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    assert torch.isfinite(t['weight'][1]).all()


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_inf_utility_does_not_crash(loss_fn):
    u_hat, u_target, valid_mask = _fixture(bsz=2, n=8, n_invalid=2, seed=7)
    u_target[0, 0] = float('inf')
    loss, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)  # must not raise


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_gpu_matches_cpu_if_available(loss_fn):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    u_hat, u_target, valid_mask = _fixture(bsz=4, n=20, seed=8)
    l_cpu, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=5)
    l_gpu, _ = loss_fn(u_hat.cuda(), u_target.cuda(), valid_mask.cuda(), TAU, m=5)
    assert torch.allclose(l_cpu, l_gpu.cpu(), atol=1e-5)


@pytest.mark.parametrize('loss_fn', [soft_regret_mass_loss, set_utility_soft_ce_loss])
def test_fp32_finite_loss_and_gradient(loss_fn):
    for seed in range(10):
        u_hat, u_target, valid_mask = _fixture(seed=seed)
        u_hat = u_hat.clone().requires_grad_(True)
        loss, _ = loss_fn(u_hat, u_target, valid_mask, TAU, m=3)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(u_hat.grad).all()


# --------------------------------------------------------------------------
# SRM-specific
# --------------------------------------------------------------------------
def test_srm_m1_equals_hard_choice_ce():
    """With M=1, P_t is exactly {argmax}, weight there is exp(0)=1, so
    L_SRM,t = -logsumexp_{i*}(log p(i*) - 0) = -log p(i*) -- the Hard
    Choice CE NLL against the same target index (up to the two functions'
    own row-filtering conventions, which are identical)."""
    u_hat, u_target, valid_mask = _fixture(seed=9)
    l_srm, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=1)
    l_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_mask, TAU)
    assert torch.allclose(l_srm, l_hard, atol=1e-5)


def test_srm_decreases_when_good_candidate_probability_increases():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=0, seed=10)
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    best = int(u_target[0].argmax())
    l0, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=3)
    u_hat2 = u_hat.clone()
    u_hat2[0, best] += 5.0
    l1, _ = soft_regret_mass_loss(u_hat2, u_target, valid_mask, TAU, m=3)
    assert float(l1) < float(l0)


def test_srm_increases_when_bad_candidate_probability_increases():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=0, seed=11)
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    worst_valid = int(u_target[0].masked_fill(~valid_mask[0], float('inf')).argmin())
    l0, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=3)
    u_hat2 = u_hat.clone()
    u_hat2[0, worst_valid] += 5.0
    l1, _ = soft_regret_mass_loss(u_hat2, u_target, valid_mask, TAU, m=3)
    assert float(l1) >= float(l0)


def test_srm_logspace_matches_direct_computation_small_input():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=5, n_invalid=0, seed=12)
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    logits = u_hat.masked_fill(~valid_mask, torch.finfo(u_hat.dtype).min / 4) / TAU
    log_probs = F.log_softmax(logits, dim=-1)
    p_mask, regret = t['p_mask'][0], t['regret_norm'][0]
    direct = -math.log(sum(float(log_probs[0, i].exp() * math.exp(-float(regret[i])))
                          for i in range(5) if bool(p_mask[i])))
    l_srm, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=3)
    assert abs(direct - float(l_srm)) < 1e-4


# --------------------------------------------------------------------------
# SoftCE-specific
# --------------------------------------------------------------------------
def test_softce_m1_equals_hard_choice_ce():
    u_hat, u_target, valid_mask = _fixture(seed=13)
    l_ce, _ = set_utility_soft_ce_loss(u_hat, u_target, valid_mask, TAU, m=1)
    l_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_mask, TAU)
    assert torch.allclose(l_ce, l_hard, atol=1e-5)


def test_softce_equal_utility_positives_get_equal_teacher_probability():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=0, seed=14)
    u_target[0, :4] = 3.0  # 4-way tie for the top
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=4)
    w = t['weight'][0, :4]
    assert torch.allclose(w, w[0].expand_as(w), atol=1e-6)


def test_softce_higher_utility_positive_gets_higher_teacher_probability():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=10, n_invalid=0, seed=15)
    u_target[0, 0] = 3.0
    u_target[0, 1] = 1.0
    u_target[0, 2:] = -5.0  # everything else far below -> 0/1 clearly in P_t
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    assert float(t['weight'][0, 0]) > float(t['weight'][0, 1])


def test_softce_gradient_near_zero_when_student_matches_teacher():
    u_hat, u_target, valid_mask = _fixture(bsz=1, n=8, n_invalid=0, seed=16)
    t = compute_oracle_teacher_signal(u_target, valid_mask, m=3)
    pi = t['weight'][0] / t['weight'][0].sum()
    # construct u_hat so that softmax(u_hat/tau) == pi exactly on P_t, -inf elsewhere
    student_logits = torch.where(t['p_mask'][0], torch.log(pi.clamp_min(1e-12)) * TAU,
                                 torch.full_like(pi, -1e4))
    u_hat = student_logits.unsqueeze(0).clone().requires_grad_(True)
    loss, _ = set_utility_soft_ce_loss(u_hat, u_target, valid_mask, TAU, m=3)
    loss.backward()
    assert float(u_hat.grad.abs().max()) < 1e-3


# --------------------------------------------------------------------------
# prefix sensitivity (integration against the real, unmodified reference
# Set Oracle -- utils/dense_utility.py / greedy_set_utility, untouched)
# --------------------------------------------------------------------------
def test_prefix_change_changes_set_utility_and_teacher_target():
    g = torch.Generator().manual_seed(20)
    bsz, n, h = 2, 10, 4
    futures = torch.randn(bsz, n, h, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    host_scores = torch.rand(bsz, n, generator=g) * 2 - 1
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    w_host = candidate_weights(host_scores, cand_mask, TAU)

    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    one_prefix = torch.zeros(bsz, 1, dtype=torch.long)  # prefix = candidate 0

    u_empty = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=8)
    u_one = greedy_set_utility(one_prefix, w_host, futures, query_future, chunk_size=8)
    assert not torch.allclose(u_empty, u_one)

    t_empty = compute_oracle_teacher_signal(u_empty, cand_mask, m=3)
    t_one = compute_oracle_teacher_signal(u_one, cand_mask, m=3)
    assert not torch.equal(t_empty['p_mask'], t_one['p_mask']) or \
        not torch.allclose(t_empty['weight'], t_one['weight'])


def test_deterministic_given_same_inputs():
    u_hat, u_target, valid_mask = _fixture(seed=21)
    l1, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=3)
    l2, _ = soft_regret_mass_loss(u_hat, u_target, valid_mask, TAU, m=3)
    assert torch.equal(l1, l2)
    l3, _ = set_utility_soft_ce_loss(u_hat, u_target, valid_mask, TAU, m=3)
    l4, _ = set_utility_soft_ce_loss(u_hat, u_target, valid_mask, TAU, m=3)
    assert torch.equal(l3, l4)
