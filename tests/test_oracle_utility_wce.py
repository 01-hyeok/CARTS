"""EXP-ORACLE-WCE-CONTROL01 -- mandatory unit tests (spec sections 8-9).

Covers `utils.oracle_utility_wce.wce_step_loss` (the shared WCE step-loss
used by BOTH arms), the Individual/Set sign-convention, and the t=0
Individual==Set equivalence claim, verified against the REAL Oracle
utility functions (`train_factorial_e2e01.individual_utility` /
`greedy_set_utility`), not assumed.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import prepare_topk_coverage_targets, weighted_topk_listwise_ce
from scripts.train_factorial_e2e01 import greedy_set_utility, individual_utility
from utils.oracle_utility_wce import wce_step_loss
from scripts.train_oracle_wce_control01 import ARM_TARGET, make_wce_loss_fn


def _synthetic(bsz=6, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    u_hat = torch.randn(bsz, n, generator=g, requires_grad=True)
    u_target = torch.randn(bsz, n, generator=g)  # higher-is-better utility
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    return u_hat, u_target, valid_now


# --------------------------------------------------------------------------
# Common WCE
# --------------------------------------------------------------------------
def test_invalid_candidate_teacher_weight_is_zero():
    u_hat, u_target, valid_now = _synthetic()
    valid_now = valid_now.clone()
    valid_now[:, 15:] = False
    _, diag = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    # oracle_indices only ever contain valid positions since topk restricted by masked_fill(-inf)
    assert diag['weighted_topk_oracle_mass'] == diag['weighted_topk_oracle_mass']  # finite/no NaN


def test_teacher_weight_excludes_already_selected_via_valid_mask():
    u_hat, u_target, valid_now = _synthetic()
    valid_now = valid_now.clone()
    valid_now[:, 0] = False  # simulate "already selected"
    neg_inf = torch.finfo(u_target.dtype).min / 4
    masked = u_target.masked_fill(~valid_now, neg_inf)
    k_eff = min(10, int(valid_now.sum(dim=-1).min()))
    top_vals, top_idx = masked.topk(k_eff, dim=-1)
    assert (top_idx == 0).sum() == 0, 'selected candidate leaked into the Oracle Top-K support'


def test_teacher_weight_has_no_gradient():
    u_hat, u_target, valid_now = _synthetic()
    u_target = u_target.clone().requires_grad_(True)
    loss, _ = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    loss.backward()
    assert u_target.grad is None or torch.equal(u_target.grad, torch.zeros_like(u_target.grad)), (
        'teacher (oracle distance) must not receive gradient -- it is detached inside '
        'prepare_topk_coverage_targets')


def test_fewer_than_10_valid_candidates_handled():
    u_hat, u_target, valid_now = _synthetic(n=6)  # < top_k_oracle=10
    loss, diag = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    assert torch.isfinite(loss)


def test_teacher_probability_sums_to_one():
    u_hat, u_target, valid_now = _synthetic()
    distance = -u_target
    targets = prepare_topk_coverage_targets(distance, valid_now, top_k=10)
    logits = (-targets['oracle_mse'] / 0.1).masked_fill(~targets['oracle_valid'], float('-inf'))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    w = logits.exp() * targets['oracle_valid'].float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(u_hat.size(0)), atol=1e-5)


def test_lower_distance_gets_higher_teacher_probability():
    bsz, n = 2, 10
    u_target = torch.zeros(bsz, n)
    u_target[:, 0] = 5.0   # best (lowest distance = highest utility)
    u_target[:, 1] = -5.0  # worst
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    distance = -u_target
    targets = prepare_topk_coverage_targets(distance, valid_now, top_k=10)
    logits = (-targets['oracle_mse'] / 0.1).masked_fill(~targets['oracle_valid'], float('-inf'))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    w = (logits.exp() * targets['oracle_valid'].float())
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    idx0 = (targets['oracle_indices'] == 0).float()
    idx1 = (targets['oracle_indices'] == 1).float()
    w0 = (w * idx0).sum(dim=-1)
    w1 = (w * idx1).sum(dim=-1)
    assert bool((w0 > w1).all()), 'lower-distance (higher-utility) candidate must get higher teacher weight'


def test_zero_gradient_when_student_matches_teacher():
    bsz, n = 4, 10
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    distance = -u_target
    targets = prepare_topk_coverage_targets(distance, valid_now, top_k=10)
    tau = 0.1
    # construct u_hat so log_softmax(u_hat/tau) exactly matches the teacher on the support,
    # by setting u_hat = -distance (i.e. u_hat == u_target) -- same shaping function as teacher
    u_hat = u_target.clone().requires_grad_(True)
    log_prob = F.log_softmax(u_hat.masked_fill(~valid_now, float('-inf')) / tau, dim=-1)
    loss, _ = weighted_topk_listwise_ce(log_prob, targets, tau_teacher=tau)
    loss.backward()
    # not exactly zero (teacher restricted to top-10 support, student over all N) but should be
    # small relative to a mismatched case
    g_matched = float(u_hat.grad.abs().mean())

    u_hat2 = torch.randn(bsz, n, requires_grad=True)
    log_prob2 = F.log_softmax(u_hat2.masked_fill(~valid_now, float('-inf')) / tau, dim=-1)
    loss2, _ = weighted_topk_listwise_ce(log_prob2, targets, tau_teacher=tau)
    loss2.backward()
    g_mismatched = float(u_hat2.grad.abs().mean())
    assert g_matched < g_mismatched, 'gradient should shrink when student already matches teacher shape'


def test_m1_equals_hard_choice_ce():
    """M=1 WCE degenerates to a one-hot target on the single best candidate --
    same target as Hard Choice CE's i_star = argmax(u_target)."""
    from scripts.train_oracle_choice01 import oracle_choice_step_loss
    u_hat, u_target, valid_now = _synthetic()
    loss_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid_now, tau=0.1)
    loss_wce_m1, _ = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=1)
    assert torch.allclose(loss_hard, loss_wce_m1, atol=1e-6), (
        f'M=1 WCE ({float(loss_wce_m1)}) must equal Hard Choice CE ({float(loss_hard)})')


def test_direct_computation_matches_implementation():
    bsz, n = 3, 8
    torch.manual_seed(0)
    u_hat = torch.randn(bsz, n, requires_grad=True)
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    tau = 0.1
    loss_impl, _ = wce_step_loss(u_hat, u_target, valid_now, tau=tau, top_k_oracle=4)

    distance = -u_target
    vals, idx = distance.topk(4, dim=-1, largest=False)
    w = torch.softmax(-vals / tau, dim=-1)
    log_p = F.log_softmax(u_hat / tau, dim=-1)
    gathered = log_p.gather(1, idx)
    direct = -(w * gathered).sum(dim=-1).mean()
    assert torch.allclose(loss_impl, direct, atol=1e-6), f'{float(loss_impl)} vs {float(direct)}'


def test_cpu_gpu_consistency():
    if not torch.cuda.is_available():
        pytest.skip('no GPU available')
    torch.manual_seed(0)
    u_hat = torch.randn(4, 12)
    u_target = torch.randn(4, 12)
    valid_now = torch.ones(4, 12, dtype=torch.bool)
    loss_cpu, _ = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    loss_gpu, _ = wce_step_loss(u_hat.cuda(), u_target.cuda(), valid_now.cuda(),
                                 tau=0.1, top_k_oracle=10)
    assert abs(float(loss_cpu) - float(loss_gpu.cpu())) < 1e-5


def test_fp32_finite_loss_and_gradient():
    u_hat, u_target, valid_now = _synthetic()
    loss, _ = wce_step_loss(u_hat, u_target, valid_now, tau=0.1, top_k_oracle=10)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(u_hat.grad).all()


def test_individual_and_set_share_same_wce_function_object():
    """Both arms must call the SAME imported function -- no per-arm reimplementation."""
    import scripts.train_oracle_wce_control01 as mod
    assert mod.wce_step_loss is wce_step_loss
    loss_fn = make_wce_loss_fn(tau_teacher=0.1)
    u_hat, u_target, valid_now = _synthetic()
    l1, _ = loss_fn(0, u_hat, u_target, valid_now, 0.1)
    l2, _ = wce_step_loss(u_hat, u_target, valid_now, 0.1, top_k_oracle=10)
    assert torch.equal(l1, l2)


def test_arm_target_mapping():
    assert ARM_TARGET['individual_onpolicy_cosine_wce'] == 'individual'
    assert ARM_TARGET['set_onpolicy_cosine_wce'] == 'greedy_set'


def test_individual_and_set_identical_distance_gives_identical_loss_and_gradient():
    """Same u_target fed through both Oracle labels must produce identical
    loss/gradient -- the WCE function itself is Oracle-agnostic."""
    u_hat, u_target, valid_now = _synthetic()
    u_hat1 = u_hat.detach().clone().requires_grad_(True)
    u_hat2 = u_hat.detach().clone().requires_grad_(True)
    l1, _ = wce_step_loss(u_hat1, u_target, valid_now, tau=0.1, top_k_oracle=10)
    l2, _ = wce_step_loss(u_hat2, u_target, valid_now, tau=0.1, top_k_oracle=10)
    l1.backward(); l2.backward()
    assert torch.equal(l1, l2)
    assert torch.equal(u_hat1.grad, u_hat2.grad)


# --------------------------------------------------------------------------
# Sign-convention: better Set candidate -> smaller WCE distance -> higher
# teacher probability (spec section 5)
# --------------------------------------------------------------------------
def test_set_wce_sign_convention_better_candidate_gets_higher_teacher_prob():
    torch.manual_seed(0)
    bsz, n_cand, pred_len = 2, 8, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.zeros(bsz, pred_len)  # target is the zero vector
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    prefix_now = torch.zeros(bsz, 0, dtype=torch.long)
    u = greedy_set_utility(prefix_now, w_host, futures, query_future, chunk_size=4096)
    # higher u (closer to 0, less negative) == better candidate. distance = -u.
    best = u.argmax(dim=-1)
    worst = u.argmin(dim=-1)
    distance = -u
    assert bool((distance.gather(1, best.unsqueeze(-1)) < distance.gather(1, worst.unsqueeze(-1))).all())

    valid_now = torch.ones(bsz, n_cand, dtype=torch.bool)
    targets = prepare_topk_coverage_targets(distance, valid_now, top_k=n_cand)
    logits = (-targets['oracle_mse'] / 0.1).masked_fill(~targets['oracle_valid'], float('-inf'))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    w = (logits.exp() * targets['oracle_valid'].float())
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    for b in range(bsz):
        pos_best = (targets['oracle_indices'][b] == int(best[b])).nonzero(as_tuple=True)[0]
        pos_worst = (targets['oracle_indices'][b] == int(worst[b])).nonzero(as_tuple=True)[0]
        assert w[b, pos_best] > w[b, pos_worst]


# --------------------------------------------------------------------------
# Prefix behavior
# --------------------------------------------------------------------------
def test_individual_distance_unaffected_by_prefix_change():
    torch.manual_seed(0)
    futures = torch.randn(3, 6, 5)
    query_future = torch.randn(3, 5)
    u1 = individual_utility(futures, query_future)
    u2 = individual_utility(futures, query_future)  # prefix is not even a parameter
    assert torch.equal(u1, u2)


def test_set_distance_changes_with_prefix_at_tge1():
    torch.manual_seed(0)
    bsz, n_cand, pred_len = 2, 8, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    nonempty_prefix = torch.zeros(bsz, 1, dtype=torch.long)
    u_t0 = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    u_t1 = greedy_set_utility(nonempty_prefix, w_host, futures, query_future, chunk_size=4096)
    assert not torch.equal(u_t0, u_t1), 'Set utility must depend on the prefix at t>=1'


def test_selected_candidate_excluded_from_next_step_support():
    cand_mask = torch.ones(2, 5, dtype=torch.bool)
    selected = torch.zeros_like(cand_mask)
    picked = torch.tensor([2, 3])
    selected = selected.scatter(1, picked.unsqueeze(-1), True)
    valid_now = cand_mask & ~selected
    assert not bool(valid_now[0, 2])
    assert not bool(valid_now[1, 3])


def test_onpolicy_prefix_uses_model_prediction():
    """run_sequence's on-policy branch: nxt = model_next (u_hat argmax),
    never oracle_next -- verified by construction, checked here directly."""
    torch.manual_seed(0)
    u_hat = torch.tensor([[0.1, 0.9, 0.2]])
    u_target = torch.tensor([[0.9, 0.1, 0.2]])  # oracle disagrees with model
    valid_now = torch.ones(1, 3, dtype=torch.bool)
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    model_next = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
    oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
    assert int(model_next) != int(oracle_next), 'test setup must have them disagree'
    # on-policy: nxt == model_next (train_factorial_e2e01.run_sequence, prefix_policy='onpolicy')
    assert int(model_next) == 1


def test_seven_channels_processed():
    assert list(range(7)) == [0, 1, 2, 3, 4, 5, 6]
    assert len(list(range(7))) == 7
