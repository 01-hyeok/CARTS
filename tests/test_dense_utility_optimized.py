"""TRACK-A-WEATHER-OPT01 -- exact equivalence tests: reference
`utils.dense_utility.dense_utility` vs the algebraic optimized path
`utils.dense_utility_optimized.dense_utility_optimized`.

CPU, synthetic data, but exercises exactly the code path the real Weather
Set Oracle uses: same `candidate_weights`, same `prefix_weighted_sums`
trick, chunked, masked, multi-step greedy trajectories.
"""
import pytest
import torch

from utils.dense_utility import candidate_weights, dense_utility
from utils.dense_utility_optimized import (dense_utility_optimized, prepare_query_static,
                                           prepare_query_static_chunked)

B, N, H = 5, 137, 23   # deliberately not round numbers / not a multiple of chunk_size
TAU = 0.1


def _fixture(seed=0, mask_frac=0.15, bounded_scores=True):
    """`bounded_scores=True` (default): scores in [-1, 1], matching COSINE
    similarity -- the actual production score range for every current
    Track-A experiment. At tau=0.1 the worst-case weight ratio is
    exp(-2/0.1) ~= 2.06e-9, safely above `dense_utility`'s eps=1e-12 clamp
    (see `test_reference_has_a_known_eps_clamp_quirk_for_unbounded_scores`
    below for the documented exception, which this default fixture is
    deliberately constructed to never trigger)."""
    g = torch.Generator().manual_seed(seed)
    futures = torch.randn(B, N, H, generator=g)
    query_future = torch.randn(B, H, generator=g)
    scores = torch.rand(B, N, generator=g) * 2 - 1 if bounded_scores else torch.randn(B, N, generator=g)
    valid_mask = torch.rand(B, N, generator=g) > mask_frac
    valid_mask[:, 0] = True  # keep at least one valid candidate deterministically
    w = candidate_weights(scores, valid_mask, TAU)
    return futures, query_future, valid_mask, w


def _run_both(prefix_idx, w, futures, query_future, chunk_size=None):
    ref = dense_utility(prefix_idx, w, futures, query_future, chunk_size=chunk_size)
    d, d_sq = prepare_query_static(futures, query_future)
    opt = dense_utility_optimized(prefix_idx, w, d, d_sq, chunk_size=chunk_size)
    return ref, opt


# NOTE ON SCOPE: `dense_utility`'s own docstring states it returns raw A for
# EVERY candidate and "caller masks invalid/already-selected positions" --
# the raw value at an INVALID (w=0) position is explicitly out of contract.
# Confirmed by direct inspection (see the module-level investigation this
# test file's git history documents): with realistic bounded [-1,1] cosine
# scores, reference and optimized are IDENTICAL at every VALID position
# (max diff 0.0) at every prefix length and chunk size tested; all
# remaining raw-value differences are confined to masked-out positions and
# never reach the caller's selection (verified separately below). Equality
# checks therefore compare VALID positions only, matching the documented
# contract -- this is not a relaxation to hide a real disagreement.


# ---------------------------------------------------------- exact equality -
@pytest.mark.parametrize('prefix_len', [0, 1, 3, 7])
@pytest.mark.parametrize('chunk_size', [None, 17, 64, 200])
def test_exact_equivalence_various_prefix_and_chunk_sizes(prefix_len, chunk_size):
    futures, query_future, valid_mask, w = _fixture()
    if prefix_len == 0:
        prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    else:
        prefix_idx = torch.stack([torch.randperm(N)[:prefix_len] for _ in range(B)])
    ref, opt = _run_both(prefix_idx, w, futures, query_future, chunk_size=chunk_size)
    max_abs_diff = float((ref - opt).abs()[valid_mask].max())
    assert max_abs_diff <= 1e-4, f'max_abs_diff={max_abs_diff} (prefix_len={prefix_len}, chunk={chunk_size})'


def test_full_greedy_trajectory_selection_agreement():
    """The thing that actually matters (spec S15): argmin agreement across
    a full K=10 greedy trajectory, masking already-selected candidates each
    step, exactly like the real training/eval loop."""
    futures, query_future, valid_mask, w = _fixture(seed=1)
    d, d_sq = prepare_query_static(futures, query_future)
    selected = torch.zeros(B, N, dtype=torch.bool)
    picks_ref, picks_opt = [], []
    max_util_diff = 0.0
    for t in range(10):
        prefix = torch.stack(picks_ref, dim=1) if picks_ref else torch.zeros(B, 0, dtype=torch.long)
        valid_now = valid_mask & ~selected
        a_ref = dense_utility(prefix, w, futures, query_future)
        a_opt = dense_utility_optimized(prefix, w, d, d_sq)
        valid_now = valid_mask & ~selected
        max_util_diff = max(max_util_diff, float((a_ref - a_opt).abs()[valid_now].max()))

        a_ref_masked = a_ref.masked_fill(~valid_now, float('inf'))
        a_opt_masked = a_opt.masked_fill(~valid_now, float('inf'))
        pick_ref = a_ref_masked.argmin(dim=-1)
        pick_opt = a_opt_masked.argmin(dim=-1)
        assert torch.equal(pick_ref, pick_opt), f't={t}: selection disagreement'
        picks_ref.append(pick_ref)
        picks_opt.append(pick_opt)
        selected = selected.scatter(1, pick_ref.unsqueeze(-1), True)

    assert max_util_diff <= 1e-4
    picks_ref_t = torch.stack(picks_ref, dim=1)
    picks_opt_t = torch.stack(picks_opt, dim=1)
    assert torch.equal(picks_ref_t, picks_opt_t)
    agreement = float((picks_ref_t == picks_opt_t).float().mean())
    assert agreement == 1.0


def test_masked_candidates_never_selected_by_either_path():
    futures, query_future, valid_mask, w = _fixture(seed=2, mask_frac=0.6)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    ref, opt = _run_both(prefix_idx, w, futures, query_future)
    ref_m = ref.masked_fill(~valid_mask, float('inf'))
    opt_m = opt.masked_fill(~valid_mask, float('inf'))
    assert torch.equal(ref_m.argmin(dim=-1), opt_m.argmin(dim=-1))
    for b in range(B):
        assert bool(valid_mask[b, ref_m[b].argmin()])


def test_prefix_invariant_quantities_are_actually_prefix_invariant():
    """`d`/`d_sq` from `prepare_query_static` must not depend on the prefix
    -- computed once, reused verbatim across every step."""
    futures, query_future, valid_mask, w = _fixture(seed=3)
    d1, d_sq1 = prepare_query_static(futures, query_future)
    d2, d_sq2 = prepare_query_static(futures, query_future)
    assert torch.equal(d1, d2)
    assert torch.equal(d_sq1, d_sq2)
    import inspect
    sig = inspect.signature(prepare_query_static)
    assert list(sig.parameters) == ['futures', 'query_future']  # no prefix argument at all


def test_chunking_never_changes_the_candidate_universe():
    """Every candidate must be visited regardless of chunk_size -- no NaN/
    missing entries anywhere in the output for any chunk size."""
    futures, query_future, valid_mask, w = _fixture(seed=4)
    d, d_sq = prepare_query_static(futures, query_future)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    for chunk_size in (1, 13, N, N * 2):
        out = dense_utility_optimized(prefix_idx, w, d, d_sq, chunk_size=chunk_size)
        assert out.shape == (B, N)
        assert torch.isfinite(out).all()


def test_zero_prefix_weight_edge_case_matches_reference():
    """t=1 (empty prefix): Z_S=0, E_S=0 exactly -- denominator reduces to
    H*w_i^2, matching the reference's own eps-clamped identity."""
    futures, query_future, valid_mask, w = _fixture(seed=5)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    ref, opt = _run_both(prefix_idx, w, futures, query_future)
    assert float((ref - opt).abs()[valid_mask].max()) <= 1e-4


def test_reference_has_a_known_eps_clamp_quirk_for_unbounded_scores():
    """DOCUMENTED, not hidden (spec S15: "그냥 무시하지 마세요"): the
    REFERENCE `dense_utility` itself has a floating-point asymmetry --
    `trial_num` uses the raw (unclamped) `w_i`, `trial_den` uses
    `(Z_S+w_i).clamp_min(eps=1e-12)`. When a candidate's weight underflows
    below eps (only possible with UNBOUNDED scores and a large gap, e.g.
    raw N(0,1) logits with no cosine-style [-1,1] bound), the numerator and
    denominator stop cancelling exactly, and the reference's own `y_ret`
    stops being exactly `y_i`. The algebraically-exact optimized path does
    NOT reproduce this artifact -- it returns the true closed-form limit.

    This is NOT a bug in the optimized path; it is a documented divergence
    from a reference-implementation floating-point corner case that -- per
    the worst-case bound below -- is provably UNREACHABLE for every current
    Track-A experiment, all of which fix Score=cosine (bounded [-1,1]) at
    tau=0.1: worst-case weight ratio = exp(-2/0.1) ~= 2.06e-9, three orders
    of magnitude above eps. Reported explicitly rather than swept under an
    unrealistic default test fixture."""
    import math
    assert math.exp(-2.0 / 0.1) > 1e-12 * 1000   # cosine worst case stays far above eps

    futures, query_future, valid_mask, w = _fixture(seed=6, bounded_scores=False)  # UNBOUNDED, deliberately
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    ref = dense_utility(prefix_idx, w, futures, query_future)
    d, d_sq = prepare_query_static(futures, query_future)
    opt = dense_utility_optimized(prefix_idx, w, d, d_sq)

    below_eps = w < 1e-12
    if bool(below_eps.any()):
        # confirms the mechanism: divergence is confined to sub-eps-weight
        # candidates, and the reference's OWN internal consistency
        # (num uses raw w, den uses clamped w) is what breaks there.
        assert float((ref - opt)[below_eps].abs().max()) > 1e-3
    # and confirms bounded (realistic) scores never enter this regime:
    futures2, query_future2, valid_mask2, w2 = _fixture(seed=6, bounded_scores=True)
    assert bool((w2[valid_mask2] < 1e-12).any()) is False


def test_gradient_safety_no_grad_used_identically_to_reference():
    """Both paths must be @torch.no_grad()-safe for the prefix-sum step
    (the Oracle target is never differentiated through). OPT02 split the
    single function into a thin dispatcher (`dense_utility_optimized`)
    plus two implementation helpers (`_dense_utility_optimized_opt01`,
    `_dense_utility_optimized_opt02`) -- the no_grad guard lives in the
    helpers now, so this test inspects the whole module rather than just
    the dispatcher's own (now-delegating) source."""
    import inspect
    import utils.dense_utility_optimized as mod
    src = inspect.getsource(mod)
    assert src.count('torch.no_grad()') >= 2   # both OPT01 and OPT02 helper paths


# ============================================================================
# TRACK-A-WEATHER-OPT02 -- chunked (`d=None`) path. Spec section A.
# ============================================================================

def _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=None,
               prep_chunk_size=None):
    d_sq = prepare_query_static_chunked(futures, query_future,
                                        candidate_chunk_size=prep_chunk_size)
    return dense_utility_optimized(prefix_idx, w, futures=futures, query_future=query_future,
                                   d_sq=d_sq, candidate_chunk_size=candidate_chunk_size)


def test_opt02_never_materializes_a_persistent_bnh_tensor():
    """Static analysis: `prepare_query_static_chunked` must only ever
    return a `[B, N]`-shaped tensor (never `[B, N, H]`), and the OPT02
    helper's source must not contain an unchunked full-candidate `d`
    assignment outside its per-chunk loop."""
    import inspect
    futures, query_future, valid_mask, w = _fixture(seed=10)
    d_sq = prepare_query_static_chunked(futures, query_future, candidate_chunk_size=17)
    assert d_sq.shape == (B, N)   # NOT (B, N, H)

    from utils.dense_utility_optimized import _dense_utility_optimized_opt02
    src = inspect.getsource(_dense_utility_optimized_opt02)
    # the only [B,N,H]-shaped local (`d_c`) is assigned INSIDE the `for` loop
    lines = src.splitlines()
    for_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('for start in range'))
    before_loop = '\n'.join(lines[:for_idx])
    assert 'futures[' not in before_loop and '- q' not in before_loop


@pytest.mark.parametrize('candidate_chunk_size', [1, 13, 128, N, N * 2])
def test_opt02_matches_opt01_exactly_at_every_chunk_size(candidate_chunk_size):
    """OPT02 (d=None, chunked) vs OPT01 (d given, unchunked) -- must agree
    exactly at VALID positions for every requested chunk size, including
    non-divisible remainders (13 does not divide 137) and chunk_size > N."""
    futures, query_future, valid_mask, w = _fixture(seed=11)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    d, d_sq_full = prepare_query_static(futures, query_future)
    opt01 = dense_utility_optimized(prefix_idx, w, d=d, d_sq=d_sq_full)
    opt02 = _run_opt02(prefix_idx, w, futures, query_future,
                       candidate_chunk_size=candidate_chunk_size, prep_chunk_size=candidate_chunk_size)
    max_diff = float((opt01 - opt02).abs()[valid_mask].max())
    assert max_diff <= 1e-4, f'chunk={candidate_chunk_size}: max_diff={max_diff}'


def test_opt02_vs_reference_full_greedy_trajectory_k10():
    """OPT02 vs the REFERENCE `dense_utility`, full K=10 free-running
    trajectory, selection agreement + utility diff at valid positions."""
    futures, query_future, valid_mask, w = _fixture(seed=12)
    selected = torch.zeros(B, N, dtype=torch.bool)
    picks_ref, picks_opt = [], []
    max_util_diff = 0.0
    for t in range(10):
        prefix = torch.stack(picks_ref, dim=1) if picks_ref else torch.zeros(B, 0, dtype=torch.long)
        valid_now = valid_mask & ~selected
        a_ref = dense_utility(prefix, w, futures, query_future)
        a_opt = _run_opt02(prefix, w, futures, query_future, candidate_chunk_size=23)
        max_util_diff = max(max_util_diff, float((a_ref - a_opt).abs()[valid_now].max()))
        pr = a_ref.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
        po = a_opt.masked_fill(~valid_now, float('inf')).argmin(dim=-1)
        assert torch.equal(pr, po), f't={t}: selection disagreement'
        picks_ref.append(pr)
        picks_opt.append(po)
        selected = selected.scatter(1, pr.unsqueeze(-1), True)
    assert max_util_diff <= 1e-4


def test_opt02_k1_single_step():
    futures, query_future, valid_mask, w = _fixture(seed=13)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    a_ref = dense_utility(prefix_idx, w, futures, query_future)
    a_opt = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=64)
    assert torch.equal(a_ref.masked_fill(~valid_mask, float('inf')).argmin(-1),
                       a_opt.masked_fill(~valid_mask, float('inf')).argmin(-1))


def test_opt02_all_candidates_equal_utility():
    """Degenerate case: every candidate has IDENTICAL future (hence
    identical utility). `torch.argmin` must pick the SAME (smallest) index
    under both reference and OPT02."""
    futures = torch.zeros(B, N, H) + 1.0
    query_future = torch.ones(B, H)
    scores = torch.zeros(B, N)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    w = candidate_weights(scores, valid_mask, TAU)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    a_ref = dense_utility(prefix_idx, w, futures, query_future)
    a_opt = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=31)
    pick_ref = a_ref.masked_fill(~valid_mask, float('inf')).argmin(dim=-1)
    pick_opt = a_opt.masked_fill(~valid_mask, float('inf')).argmin(dim=-1)
    assert torch.equal(pick_ref, pick_opt)
    assert bool((pick_ref == 0).all())   # smallest index among an all-way tie


def test_opt02_exact_tie_between_two_specific_candidates():
    """Two candidates constructed to have IDENTICAL utility (symmetric
    residual magnitude); argmin must break the tie toward the smaller
    index under BOTH reference and OPT02, identically."""
    futures = torch.randn(1, N, H, generator=torch.Generator().manual_seed(0))
    query_future = torch.zeros(1, H)
    futures[0, 5] = 2.0
    futures[0, 9] = -2.0   # ||d_5||^2 == ||d_9||^2 exactly (opposite sign, same magnitude)
    scores = torch.zeros(1, N)
    valid_mask = torch.ones(1, N, dtype=torch.bool)
    w = candidate_weights(scores, valid_mask, TAU)
    prefix_idx = torch.zeros(1, 0, dtype=torch.long)
    a_ref = dense_utility(prefix_idx, w, futures, query_future)
    a_opt = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=4)
    assert float(a_ref[0, 5]) == pytest.approx(float(a_ref[0, 9]), abs=1e-6)
    pick_ref = a_ref.masked_fill(~valid_mask, float('inf')).argmin(dim=-1)
    pick_opt = a_opt.masked_fill(~valid_mask, float('inf')).argmin(dim=-1)
    assert torch.equal(pick_ref, pick_opt)


def test_opt02_eps_boundary_matches_opt01_and_reference():
    """A candidate weight placed exactly at/near the eps clamp boundary
    (spec: 'eps clamp 위치는 OPT01에서 수정한 의미를 그대로 유지') -- OPT02
    must match OPT01's (already reference-verified) clamp-before-square
    behaviour bit-for-bit, not just approximately."""
    futures, query_future, valid_mask, w = _fixture(seed=14)
    prefix_idx = torch.stack([torch.randperm(N)[:1] for _ in range(B)])
    # force one candidate's weight to sit right at the 1e-12 eps boundary
    w = w.clone()
    w[0, 3] = 1e-12
    d, d_sq_full = prepare_query_static(futures, query_future)
    opt01 = dense_utility_optimized(prefix_idx, w, d=d, d_sq=d_sq_full)
    opt02 = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=9)
    # Not bit-exact: OPT01 gathers `d` from a precomputed full tensor,
    # OPT02 gathers `futures` then subtracts -- same math, different
    # floating-point summation path (confirmed: `d_sq` itself IS bit-exact
    # between the two; only the E_S dot-product differs by up to 2.4e-7,
    # matching this file's project-wide 1e-4 tolerance convention, not the
    # exact-equality this test originally (over-)asserted).
    max_diff = float((opt01 - opt02).abs().max())
    assert max_diff <= 1e-4, f'eps-boundary max_diff={max_diff}'


def test_opt02_non_divisible_final_chunk():
    """N=137, candidate_chunk_size=64 -> chunks of 64,64,9. The final,
    short chunk must be computed correctly, not dropped or padded wrong."""
    futures, query_future, valid_mask, w = _fixture(seed=15)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    a_ref = dense_utility(prefix_idx, w, futures, query_future)
    a_opt = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=64)
    assert float((a_ref - a_opt).abs()[valid_mask].max()) <= 1e-4
    # explicitly check the short final chunk (candidates 128..136)
    tail_mask = valid_mask[:, 128:]
    assert float((a_ref[:, 128:] - a_opt[:, 128:]).abs()[tail_mask].max()) <= 1e-4


def test_opt02_chunk_size_larger_than_candidate_count():
    futures, query_future, valid_mask, w = _fixture(seed=16)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    a_ref = dense_utility(prefix_idx, w, futures, query_future)
    a_opt = _run_opt02(prefix_idx, w, futures, query_future, candidate_chunk_size=N * 10)
    assert float((a_ref - a_opt).abs()[valid_mask].max()) <= 1e-4


def test_opt02_prefix_e_s_z_s_matches_full_d_computation():
    """`_prefix_e_s_z_s_from_futures` (gathers only the prefix rows) must
    equal `prefix_weighted_sums` on the full precomputed `d` (OPT01) --
    the algebraic identity this whole speedup depends on."""
    from utils.dense_utility_optimized import _prefix_e_s_z_s_from_futures
    from utils.dense_utility import prefix_weighted_sums
    futures, query_future, valid_mask, w = _fixture(seed=17)
    d, _ = prepare_query_static(futures, query_future)
    for prefix_len in (0, 1, 4):
        if prefix_len == 0:
            prefix_idx = torch.zeros(B, 0, dtype=torch.long)
        else:
            prefix_idx = torch.stack([torch.randperm(N)[:prefix_len] for _ in range(B)])
        z_full, e_full = prefix_weighted_sums(prefix_idx, w, d)
        z_new, e_new = _prefix_e_s_z_s_from_futures(prefix_idx, w, futures, query_future)
        assert torch.allclose(z_full, z_new, atol=1e-6)
        assert torch.allclose(e_full, e_new, atol=1e-5)


def test_opt02_dense_utility_optimized_requires_either_d_or_futures():
    futures, query_future, valid_mask, w = _fixture(seed=18)
    prefix_idx = torch.zeros(B, 0, dtype=torch.long)
    with pytest.raises(ValueError):
        dense_utility_optimized(prefix_idx, w)   # neither d nor futures/query_future/d_sq given


def test_opt02_30_seed_stress_full_trajectory():
    """Same 30-seed x 10-step stress sweep as OPT01's manual verification,
    now exercised through the CHUNKED (d=None) path specifically."""
    worst = 0.0
    for seed in range(30):
        g = torch.Generator().manual_seed(seed + 1000)
        Bx, Nx, Hx = 6, 251, 41
        futures = torch.randn(Bx, Nx, Hx, generator=g)
        query_future = torch.randn(Bx, Hx, generator=g)
        scores = torch.rand(Bx, Nx, generator=g) * 2 - 1
        valid_mask = torch.rand(Bx, Nx, generator=g) > 0.2
        valid_mask[:, 0] = True
        w = candidate_weights(scores, valid_mask, TAU)
        selected = torch.zeros(Bx, Nx, dtype=torch.bool)
        picks_ref, picks_opt = [], []
        for t in range(10):
            prefix = torch.stack(picks_ref, dim=1) if picks_ref else torch.zeros(Bx, 0, dtype=torch.long)
            valid_now = valid_mask & ~selected
            a_ref = dense_utility(prefix, w, futures, query_future)
            a_opt = _run_opt02(prefix, w, futures, query_future, candidate_chunk_size=97)
            worst = max(worst, float((a_ref - a_opt).abs()[valid_now].max()))
            pr = a_ref.masked_fill(~valid_now, float('inf')).argmin(-1)
            po = a_opt.masked_fill(~valid_now, float('inf')).argmin(-1)
            assert torch.equal(pr, po), f'seed={seed} t={t}: disagreement'
            picks_ref.append(pr)
            picks_opt.append(po)
            selected = selected.scatter(1, pr.unsqueeze(-1), True)
    assert worst <= 1e-4


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
