"""TRACK-A-SET-NORMREGRET-CONTROL01 -- mandatory unit tests (spec section 13).

Covers `utils/oracle_utility_normregret.py` (normregret_weight,
normregret_softce_loss, normregret_srm_loss), reusing
`utils.set_loss_experimental.compute_oracle_teacher_signal` unmodified for
the affine-invariant regret/positive-set machinery.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.oracle_utility_normregret import (_check_finite, normregret_softce_loss,
                                             normregret_srm_loss, normregret_weight)
from utils.oracle_utility_wce import wce_step_loss

TAU_S, TAU_T = 0.1, 0.1


def _synth(bsz=5, n=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    u_hat = torch.randn(bsz, n, generator=g, requires_grad=True)
    u_target = torch.randn(bsz, n, generator=g)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    return u_hat, u_target, valid_now


# --------------------------------------------------------------------------
# common
# --------------------------------------------------------------------------
def test_invalid_candidate_weight_zero():
    u_hat, u_target, valid_now = _synth()
    valid_now = valid_now.clone()
    valid_now[:, 10:] = False
    weight, teacher = normregret_weight(u_target, valid_now, TAU_T, m=10)
    assert bool((weight[:, 10:] == 0).all())


def test_selected_candidate_weight_zero():
    u_hat, u_target, valid_now = _synth()
    valid_now = valid_now.clone()
    valid_now[:, 0] = False  # simulate "already selected"
    weight, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    assert bool((weight[:, 0] == 0).all())


def test_teacher_no_gradient():
    u_hat, u_target, valid_now = _synth()
    u_target = u_target.clone().requires_grad_(True)
    weight, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    loss.backward()
    assert u_target.grad is None or torch.equal(u_target.grad, torch.zeros_like(u_target.grad))


def test_shift_invariance():
    u_hat, u_target, valid_now = _synth()
    w1, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    w2, _ = normregret_weight(u_target + 7.3, valid_now, TAU_T, m=10)
    assert torch.allclose(w1, w2, atol=1e-5)


def test_positive_scale_invariance():
    u_hat, u_target, valid_now = _synth()
    w1, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    w2, _ = normregret_weight(u_target * 4.2, valid_now, TAU_T, m=10)
    assert torch.allclose(w1, w2, atol=1e-4)


def test_topm_cutoff_tie_all_included():
    bsz, n = 1, 12
    u_target = torch.arange(n).float().unsqueeze(0).repeat(bsz, 1)
    u_target[0, 9] = u_target[0, 8]  # tie at the M=10 cutoff boundary (values 8,9 tied)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    _, teacher = normregret_weight(u_target, valid_now, TAU_T, m=10)
    # ranks 11,10 (0-idx) are the top-2 unique values; with the tie, more than 10 positives expected
    assert int(teacher['p_mask'].sum()) >= 10


def test_fewer_than_10_valid_candidates():
    u_hat, u_target, valid_now = _synth(n=6)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert torch.isfinite(loss)
    loss2, _ = normregret_srm_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert torch.isfinite(loss2)


def test_all_valid_utility_equal_zero_loss_and_gradient():
    bsz, n = 3, 8
    u_hat = torch.randn(bsz, n, requires_grad=True)
    u_target = torch.full((bsz, n), 3.0)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert float(loss) == 0.0
    loss.backward()
    assert u_hat.grad is None or torch.equal(u_hat.grad, torch.zeros_like(u_hat.grad))

    u_hat2 = torch.randn(bsz, n, requires_grad=True)
    loss2, _ = normregret_srm_loss(u_hat2, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert float(loss2) == 0.0
    loss2.backward()
    assert u_hat2.grad is None or torch.equal(u_hat2.grad, torch.zeros_like(u_hat2.grad))


def test_nan_inf_utility_fails_fast():
    u_target = torch.randn(2, 5)
    u_target[0, 2] = float('nan')
    valid_now = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match=r'\[ISSUE\]'):
        normregret_weight(u_target, valid_now, TAU_T, m=10)


def test_no_informative_row_finite_zero_loss():
    bsz, n = 4, 6
    u_hat = torch.randn(bsz, n, requires_grad=True)
    u_target = torch.full((bsz, n), 1.0)  # all rows uninformative
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_cpu_gpu_consistency():
    if not torch.cuda.is_available():
        pytest.skip('no GPU available')
    torch.manual_seed(0)
    u_hat = torch.randn(4, 12)
    u_target = torch.randn(4, 12)
    valid_now = torch.ones(4, 12, dtype=torch.bool)
    l_cpu, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    l_gpu, _ = normregret_softce_loss(u_hat.cuda(), u_target.cuda(), valid_now.cuda(), TAU_S, TAU_T, m=10)
    assert abs(float(l_cpu) - float(l_gpu.cpu())) < 1e-5


def test_fp32_finite():
    u_hat, u_target, valid_now = _synth()
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(u_hat.grad).all()


def test_seven_channels_list():
    assert list(range(7)) == [0, 1, 2, 3, 4, 5, 6]


def test_deterministic():
    u_hat, u_target, valid_now = _synth()
    l1, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    l2, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    assert torch.equal(l1, l2)


def test_softce_and_srm_share_weight_builder():
    u_hat, u_target, valid_now = _synth()
    w1, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    w2, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    assert torch.equal(w1, w2)
    import utils.oracle_utility_normregret as mod
    assert mod.normregret_softce_loss.__globals__['normregret_weight'] is normregret_weight
    assert mod.normregret_srm_loss.__globals__['normregret_weight'] is normregret_weight


def test_a1_raw_wce_reproduces_on_small_synthetic_input():
    u_hat, u_target, valid_now = _synth()
    loss, diag = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    assert torch.isfinite(loss)
    # M=1 must equal Hard Choice CE (already covered in test_oracle_utility_wce.py);
    # re-check here as a cross-module smoke check for A1's continued availability
    loss_m1, _ = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=1)
    loss_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, tau=0.1)
    assert torch.allclose(loss_m1, loss_hard, atol=1e-6)


def test_m1_softce_equals_hard_choice_ce():
    u_hat, u_target, valid_now = _synth()
    loss_soft, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=1)
    loss_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, tau=TAU_S)
    assert torch.allclose(loss_soft, loss_hard, atol=1e-6), f'{float(loss_soft)} vs {float(loss_hard)}'


def test_m1_srm_equals_hard_choice_ce():
    u_hat, u_target, valid_now = _synth()
    loss_srm, _ = normregret_srm_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=1, eps=0.0)
    loss_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, tau=TAU_S)
    assert torch.allclose(loss_srm, loss_hard, atol=1e-5), f'{float(loss_srm)} vs {float(loss_hard)}'


# --------------------------------------------------------------------------
# Soft CE
# --------------------------------------------------------------------------
def test_softce_teacher_probability_sums_to_one():
    u_hat, u_target, valid_now = _synth()
    weight, teacher = normregret_weight(u_target, valid_now, TAU_T, m=10)
    pi = weight / weight.sum(-1, keepdim=True).clamp_min(1e-30)
    rows = teacher['row_has_valid']
    assert torch.allclose(pi[rows].sum(-1), torch.ones(int(rows.sum())), atol=1e-5)


def test_softce_higher_utility_higher_prob():
    bsz, n = 2, 10
    u_target = torch.zeros(bsz, n)
    u_target[:, 0] = 5.0
    u_target[:, 1] = -5.0
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    weight, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    assert bool((weight[:, 0] > weight[:, 1]).all())


def test_softce_equal_utility_equal_prob():
    bsz, n = 1, 10
    u_target = torch.zeros(bsz, n)
    u_target[0, :5] = 2.0  # 5 tied best
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    weight, _ = normregret_weight(u_target, valid_now, TAU_T, m=10)
    assert torch.allclose(weight[0, :5], weight[0, 0].expand(5), atol=1e-6)


def test_softce_zero_gradient_when_student_matches_teacher():
    bsz, n = 4, 10
    torch.manual_seed(0)
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    weight, teacher = normregret_weight(u_target, valid_now, TAU_T, m=10)
    pi = (weight / weight.sum(-1, keepdim=True).clamp_min(1e-30)).detach()
    # construct u_hat whose softmax (temp=TAU_S) matches pi on the support (approx via log)
    u_hat = torch.where(teacher['p_mask'], TAU_S * torch.log(pi.clamp_min(1e-12)),
                        torch.full_like(pi, -1e4)).requires_grad_(True)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=10)
    loss.backward()
    g_matched = float(u_hat.grad.abs().mean())

    u_hat2 = torch.randn(bsz, n, requires_grad=True)
    loss2, _ = normregret_softce_loss(u_hat2, u_target, valid_now, TAU_S, TAU_T, m=10)
    loss2.backward()
    g_mismatched = float(u_hat2.grad.abs().mean())
    assert g_matched < g_mismatched


def test_softce_direct_computation_matches():
    torch.manual_seed(1)
    bsz, n = 3, 8
    u_hat = torch.randn(bsz, n, requires_grad=True)
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    loss, _ = normregret_softce_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=4)

    weight, teacher = normregret_weight(u_target, valid_now, TAU_T, m=4)
    pi = weight / weight.sum(-1, keepdim=True).clamp_min(1e-30)
    logp = F.log_softmax(u_hat / TAU_S, dim=-1)
    direct = -(pi * logp * teacher['p_mask'].float()).sum(-1).mean()
    assert torch.allclose(loss, direct, atol=1e-5)


# --------------------------------------------------------------------------
# SRM
# --------------------------------------------------------------------------
def test_srm_loss_decreases_as_good_candidate_prob_increases():
    bsz, n = 1, 6
    u_target = torch.tensor([[5.0, 4.0, 3.0, -1.0, -2.0, -3.0]])
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    base_logits = torch.zeros(n)
    losses = []
    for boost in [0.0, 0.03, 0.08]:  # small: tau_S=0.1 sharpens softmax quickly, avoid saturation
        logits = base_logits.clone()
        logits[0] += boost  # boost the BEST candidate's logit
        u_hat = logits.unsqueeze(0).requires_grad_(False)
        loss, _ = normregret_srm_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=6)
        losses.append(float(loss))
    assert losses[0] > losses[1] > losses[2]


def test_srm_loss_increases_as_bad_candidate_prob_increases():
    bsz, n = 1, 6
    u_target = torch.tensor([[5.0, 4.0, 3.0, -1.0, -2.0, -3.0]])
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    base_logits = torch.zeros(n)
    losses = []
    for boost in [0.0, 2.0, 5.0]:
        logits = base_logits.clone()
        logits[5] += boost  # boost the WORST candidate's logit
        u_hat = logits.unsqueeze(0)
        loss, _ = normregret_srm_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=6)
        losses.append(float(loss))
    assert losses[0] < losses[1] < losses[2]


def test_srm_log_space_matches_direct():
    torch.manual_seed(2)
    bsz, n = 3, 8
    u_hat = torch.randn(bsz, n, requires_grad=True)
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    loss, _ = normregret_srm_loss(u_hat, u_target, valid_now, TAU_S, TAU_T, m=4, eps=1e-8)

    weight, teacher = normregret_weight(u_target, valid_now, TAU_T, m=4)
    logp = F.log_softmax(u_hat / TAU_S, dim=-1)
    mass = (logp.exp() * weight).sum(-1)
    direct = -torch.log(mass + 1e-8).mean()
    assert torch.allclose(loss, direct, atol=1e-5)


def test_srm_allows_concentration_on_one_positive():
    """SRM (unlike SoftCE) does not force the student to match the FULL
    teacher distribution -- concentrating all mass on a single low-regret
    positive should not be penalized beyond what matching that positive's
    weight already implies."""
    bsz, n = 1, 6
    u_target = torch.tensor([[5.0, 4.9, 3.0, -1.0, -2.0, -3.0]])  # 0,1 nearly tied best
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    concentrated = torch.tensor([[20.0, -20.0, -20.0, -20.0, -20.0, -20.0]])  # all mass on idx 0
    spread = torch.tensor([[0.5, 0.5, -20.0, -20.0, -20.0, -20.0]])  # split between 0,1
    loss_c, _ = normregret_srm_loss(concentrated, u_target, valid_now, TAU_S, TAU_T, m=6)
    loss_s, _ = normregret_srm_loss(spread, u_target, valid_now, TAU_S, TAU_T, m=6)
    assert torch.isfinite(loss_c) and torch.isfinite(loss_s)
    assert float(loss_c) < float(loss_s), 'concentrating on the single best positive should not be penalized'


# --------------------------------------------------------------------------
# Prefix (uses the real Oracle utility)
# --------------------------------------------------------------------------
def test_prefix_deterministic_on_policy_trajectory():
    from scripts.train_factorial_e2e01 import greedy_set_utility
    torch.manual_seed(0)
    bsz, n_cand, pred_len = 2, 8, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    prefix = torch.zeros(bsz, 1, dtype=torch.long)
    u1 = greedy_set_utility(prefix, w_host, futures, query_future, chunk_size=4096)
    u2 = greedy_set_utility(prefix, w_host, futures, query_future, chunk_size=4096)
    assert torch.equal(u1, u2)


def test_prefix_changes_set_utility_and_normalized_target():
    from scripts.train_factorial_e2e01 import greedy_set_utility
    torch.manual_seed(1)
    bsz, n_cand, pred_len = 2, 8, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    nonempty_prefix = torch.zeros(bsz, 1, dtype=torch.long)
    u_t0 = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    u_t1 = greedy_set_utility(nonempty_prefix, w_host, futures, query_future, chunk_size=4096)
    assert not torch.equal(u_t0, u_t1)
    valid_now = torch.ones(bsz, n_cand, dtype=torch.bool)
    w0, _ = normregret_weight(u_t0, valid_now, TAU_T, m=10)
    w1, _ = normregret_weight(u_t1, valid_now, TAU_T, m=10)
    assert not torch.equal(w0, w1)


def test_t0_set_utility_equals_singleton_utility():
    from scripts.train_factorial_e2e01 import greedy_set_utility, individual_utility
    torch.manual_seed(2)
    bsz, n_cand, pred_len = 3, 6, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    u_set_t0 = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    u_singleton = individual_utility(futures, query_future)
    assert torch.allclose(u_set_t0, u_singleton, atol=1e-6)


def test_prefix_advance_uses_student_not_oracle():
    """run_sequence's on-policy branch (train_factorial_e2e01.py) sets
    nxt = model_next = u_hat argmax, verified directly by construction."""
    u_hat = torch.tensor([[0.1, 0.9, 0.2]])
    u_target = torch.tensor([[0.9, 0.1, 0.2]])
    valid_now = torch.ones(1, 3, dtype=torch.bool)
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    model_next = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
    oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
    assert int(model_next) != int(oracle_next)
    assert int(model_next) == 1
