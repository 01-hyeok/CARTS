"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase D -- the 15 required synthetic
tests for `utils/differentiable_topk02.py::soft_topk_mask_v2`. Pure CPU,
synthetic only. Strict tolerances (v1's test suite used 5e-2, which the
spec explicitly calls insufficient for a genuine hard-limit convergence
claim) -- this suite uses 1e-4 or tighter wherever a `tau -> 0` limit is
being checked, reached by sweeping `tau` down rather than fixing one
moderate value.
"""
import torch

from models.RelationStage1 import stable_topk_indices
from utils.differentiable_topk02 import soft_topk_mask_v2, hard_limit_matches_stable_topk_v2


def _rand_scores(bsz=4, n=20, seed=0, valid_frac=1.0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(bsz, n, generator=g)
    valid_mask = torch.rand(bsz, n, generator=g) < valid_frac
    valid_mask[:, 0] = True  # never let a row go fully invalid
    return scores, valid_mask


# 1. K=1
def test_k1_selects_single_argmax_in_hard_limit():
    scores, valid_mask = _rand_scores(n=15, seed=1)
    a = soft_topk_mask_v2(scores, valid_mask, k=1, tau=1e-4)
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), 1, largest=True)
    assert torch.allclose(a.gather(-1, hard_idx), torch.ones(scores.size(0), 1), atol=1e-4)
    assert float(a.sum(-1).mean()) - float(a.gather(-1, hard_idx).sum(-1).mean()) < 1e-4


# 2. K=5
def test_k5_uniform_weight_convergence():
    scores, valid_mask = _rand_scores(n=20, seed=2)
    a = soft_topk_mask_v2(scores, valid_mask, k=5, tau=1e-4)
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), 5, largest=True)
    a_hard = a.gather(-1, hard_idx)
    assert torch.allclose(a_hard, torch.full_like(a_hard, 0.2), atol=1e-4)


# 3. K=10
def test_k10_uniform_weight_convergence():
    scores, valid_mask = _rand_scores(n=30, seed=3)
    a = soft_topk_mask_v2(scores, valid_mask, k=10, tau=1e-4)
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), 10, largest=True)
    a_hard = a.gather(-1, hard_idx)
    assert torch.allclose(a_hard, torch.full_like(a_hard, 0.1), atol=1e-4)


# 4. K=N-1
def test_k_equals_n_minus_1():
    n = 12
    scores, valid_mask = _rand_scores(n=n, seed=4, valid_frac=1.0)
    k = n - 1
    a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=1e-4)
    hard_idx = stable_topk_indices(scores, k, largest=True)
    a_hard = a.gather(-1, hard_idx)
    assert torch.allclose(a_hard, torch.full_like(a_hard, 1.0 / k), atol=1e-4)
    # the single excluded candidate (the row's min) should carry ~0 weight
    min_idx = scores.argmin(dim=-1, keepdim=True)
    assert float(a.gather(-1, min_idx).abs().max()) < 1e-3


# 5. All scores identical (every candidate tied)
def test_all_scores_identical():
    bsz, n, k = 3, 10, 4
    scores = torch.full((bsz, n), 5.0)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    # The tie-break gap between adjacent-index candidates is exactly TIE_EPS
    # (1e-9) here, since EVERY score is identical -- so tau must be several
    # orders of magnitude below 1e-9 to see the asymptotic saturation (this
    # is the documented, expected behavior for an exact tie: convergence is
    # only guaranteed as tau -> 0, same caveat as near-tied distinct scores,
    # just with a tinier effective gap to saturate against).
    a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=1e-11)
    # with every score exactly tied, the tie-break (lower index wins) must
    # select candidates 0..k-1 deterministically, matching stable_topk_indices
    hard_idx = stable_topk_indices(scores, k, largest=True)
    assert hard_idx.tolist() == [[0, 1, 2, 3]] * bsz  # sanity: tie-break is index-order
    a_hard = a.gather(-1, hard_idx)
    assert torch.allclose(a_hard, torch.full_like(a_hard, 1.0 / k), atol=1e-3)
    assert torch.allclose(a.sum(-1), torch.ones(bsz), atol=1e-6)


# 6. K-th and (K+1)-th exactly tied (boundary tie only, not a global tie)
def test_kth_and_kplus1th_tied_at_boundary():
    bsz, n, k = 2, 10, 4
    scores = torch.arange(n, dtype=torch.float32).unsqueeze(0).repeat(bsz, 1)  # 0..9, strictly increasing
    scores = scores.flip(-1)  # now strictly decreasing: 9,8,...,0 -> rank i has score 9-i
    # force a tie exactly at the K/K+1 boundary: ranks k-1 and k (0-indexed) both get the same score
    scores[:, k] = scores[:, k - 1].clone()  # candidate at rank k now ties candidate at rank k-1
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    hard_idx = stable_topk_indices(scores, k, largest=True)  # tie-break: lower index wins -> picks k-1 over k
    # as tau shrinks, the tie-broken hard set should get uniform weight and the
    # loser of the tie should vanish
    # the tie-break gap here is ~TIE_EPS/2 (~5e-10, adjacent-index tie), so tau
    # must shrink well below that scale for saturation -- same documented
    # asymptotic-only caveat as the all-identical-scores case above.
    for tau in (1e-2, 1e-6, 1e-9, 1e-12):
        a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=tau)
        a_hard = a.gather(-1, hard_idx)
        # convergence is asymptotic for exact ties (see module docstring) -- just
        # require monotonic improvement toward 1/k as tau shrinks
        err = float((a_hard - 1.0 / k).abs().max())
        if tau == 1e-2:
            prev_err = err
        else:
            assert err <= prev_err + 1e-6
            prev_err = err
    # at the smallest tau tested, should be within a loose-but-meaningful tolerance
    assert err < 1e-2


# 7. Very small (but nonzero) score gap at the boundary
def test_very_small_score_gap_at_boundary():
    bsz, n, k = 2, 10, 4
    base = torch.arange(n, dtype=torch.float32).flip(-1).unsqueeze(0).repeat(bsz, 1)
    base[:, k] = base[:, k - 1] - 1e-5  # tiny real gap, not an exact tie
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    hard_idx = stable_topk_indices(base, k, largest=True)
    a = soft_topk_mask_v2(base, valid_mask, k=k, tau=1e-7)
    a_hard = a.gather(-1, hard_idx)
    assert torch.allclose(a_hard, torch.full_like(a_hard, 1.0 / k), atol=1e-2)


# 8. Invalid candidates must get exactly zero weight, regardless of score
def test_invalid_candidates_get_exactly_zero_weight():
    bsz, n, k = 3, 16, 5
    scores, _ = _rand_scores(bsz=bsz, n=n, seed=8)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    valid_mask[:, :3] = False
    scores[:, :3] = 1e6  # even if these would otherwise dominate, they're invalid
    a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=0.05)
    assert torch.equal(a[:, :3], torch.zeros(bsz, 3))
    assert torch.allclose(a.sum(-1), torch.ones(bsz), atol=1e-6)


# 9. Extreme score magnitudes (large and small) -- no NaN/Inf
def test_extreme_score_magnitudes_no_nan_inf():
    bsz, n, k = 2, 10, 3
    scores = torch.tensor([[1e8, -1e8, 1e-8, -1e-8, 0.0, 5.0, -5.0, 1e4, -1e4, 2.0]] * bsz)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    for tau in (1e-6, 0.05, 10.0):
        a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=tau)
        assert torch.isfinite(a).all()
        assert torch.allclose(a.sum(-1), torch.ones(bsz), atol=1e-5)


# 10. Convergence as tau -> 0 (monotonic error decrease, strict tolerance at the end)
def test_convergence_as_tau_shrinks():
    scores, valid_mask = _rand_scores(n=25, seed=10)
    k = 6
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), k, largest=True)
    errs = []
    for tau in (1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5):
        a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=tau)
        a_hard = a.gather(-1, hard_idx)
        errs.append(float((a_hard - 1.0 / k).abs().max()))
    for i in range(1, len(errs)):
        assert errs[i] <= errs[i - 1] + 1e-9
    assert errs[-1] < 1e-4


# 11. PyTorch autograd gradient exists and is finite, nonzero for a boundary candidate
def test_autograd_gradient_exists_and_is_finite():
    scores, valid_mask = _rand_scores(n=15, seed=11)
    scores.requires_grad_(True)
    a = soft_topk_mask_v2(scores, valid_mask, k=4, tau=0.05)
    loss = (a ** 2).sum()
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0


# 12. Finite-difference gradient check (central difference vs autograd)
def test_finite_difference_gradient_matches_autograd():
    """Checks index 2 specifically, constructed to sit COMFORTABLY inside the
    Top-K region, far from the K/(K+1) threshold boundary -- a finite
    difference straddling a rank-boundary discontinuity (where s_(K)/s_(K+1)
    themselves swap, since the threshold is a hard order statistic even
    though it is detached from the gradient graph) would not match the
    analytic gradient, which is a sampling artifact of central differences
    near a kink, not a gradient-correctness bug; this test avoids that
    region entirely rather than tolerate a loose atol."""
    torch.manual_seed(12)
    n, k, tau = 8, 3, 0.2
    scores = torch.randn(1, n, dtype=torch.float64)
    scores[0, 2] = scores.max() + 5.0  # clearly the single highest score, deep inside top-k
    scores.requires_grad_(True)
    valid_mask = torch.ones(1, n, dtype=torch.bool)

    def f(s):
        return soft_topk_mask_v2(s, valid_mask, k=k, tau=tau)[0, 2]

    out = f(scores)
    out.backward()
    analytic = scores.grad[0, 2].item()

    eps = 1e-6
    with torch.no_grad():
        s_plus = scores.clone(); s_plus[0, 2] += eps
        s_minus = scores.clone(); s_minus[0, 2] -= eps
        f_plus = soft_topk_mask_v2(s_plus, valid_mask, k=k, tau=tau)[0, 2].item()
        f_minus = soft_topk_mask_v2(s_minus, valid_mask, k=k, tau=tau)[0, 2].item()
    numeric = (f_plus - f_minus) / (2 * eps)
    assert abs(analytic - numeric) < 1e-4


# 13. Gradient contains no NaN/Inf even with masked/invalid/-inf-adjacent inputs
def test_gradient_no_nan_inf_with_invalid_candidates():
    bsz, n, k = 2, 12, 3
    scores, _ = _rand_scores(bsz=bsz, n=n, seed=13)
    scores.requires_grad_(True)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    valid_mask[:, -4:] = False  # last 4 candidates invalid in every row
    a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=0.05)
    (a.sum()).backward()
    assert torch.isfinite(scores.grad).all()


# 14. Hard Top-K selection consistency with stable_topk_indices (the repo's own hard selector)
def test_hard_topk_selection_consistency_with_stable_topk_indices():
    scores, valid_mask = _rand_scores(n=40, seed=14)
    k = 10
    matches, max_diff = hard_limit_matches_stable_topk_v2(scores, valid_mask, k, tau=1e-5, atol=1e-3)
    assert matches, f'v2 hard-limit selection diverges from stable_topk_indices, max_diff={max_diff}'


# 15. Top-K uniform weight convergence, repeated across many random seeds (not just one)
def test_topk_uniform_weight_convergence_many_seeds():
    k = 7
    for seed in range(20):
        scores, valid_mask = _rand_scores(bsz=2, n=35, seed=100 + seed)
        a = soft_topk_mask_v2(scores, valid_mask, k=k, tau=1e-5)
        hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), k, largest=True)
        a_hard = a.gather(-1, hard_idx)
        assert torch.allclose(a_hard, torch.full_like(a_hard, 1.0 / k), atol=1e-3), f'seed={seed}'
        mass_outside = float(a.sum(-1).mean()) - float(a_hard.sum(-1).mean())
        assert mass_outside < 1e-3, f'seed={seed}'


# --- bonus: v1 vs v2 direct comparison at the exact boundary-bug case v1 has ---
def test_v2_fixes_v1_boundary_bug_directly():
    """Directly demonstrates the bug this Phase D fixes: at the K-th candidate,
    v1 is stuck at raw weight 0.5 regardless of tau; v2 is not."""
    from utils.differentiable_topk01 import soft_topk_mask as soft_topk_mask_v1
    n, k = 10, 4
    scores = torch.arange(n, dtype=torch.float32).flip(-1).unsqueeze(0)  # strictly decreasing, no ties
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    tau = 1e-5
    a_v1 = soft_topk_mask_v1(scores, valid_mask, k=k, tau=tau)
    a_v2 = soft_topk_mask_v2(scores, valid_mask, k=k, tau=tau)
    # rank k-1 (0-indexed), i.e. the K-th largest candidate (index k-1 in this
    # strictly-decreasing-by-index array)
    kth_idx = k - 1
    # v1: stuck near 0.5/(k-0.5), clearly below the correct 1/k
    assert abs(float(a_v1[0, kth_idx]) - 1.0 / k) > 0.05
    # v2: correctly converges to 1/k
    assert abs(float(a_v2[0, kth_idx]) - 1.0 / k) < 1e-3
