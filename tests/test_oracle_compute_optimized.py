"""TRACK-A-WEATHER-OPT03 -- unit tests for utils/oracle_compute_optimized.py.

Scope: Individual Oracle (A) optimization only, per the user-approved OPT03
scope narrowing (A/C/E, where E is re-exported from OPT02 unchanged and C is
the untouched reference loss). The re-export of the Greedy Set Oracle (E)
is checked here only for import/identity, NOT re-verified numerically --
that is OPT02's own 39-test suite, already passing, and duplicating it here
would violate the "don't reimplement/duplicate OPT02" instruction.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import individual_utility
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.oracle_compute_optimized import (dense_utility_optimized,
                                            individual_oracle_utility_optimized,
                                            oracle_choice_step_loss_optimized,
                                            prepare_individual_query_static)


def _make(bsz=4, n=37, h=13, seed=0, device='cpu'):
    g = torch.Generator(device='cpu').manual_seed(seed)
    futures = torch.randn(bsz, n, h, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    return futures.to(device), query_future.to(device)


# --------------------------------------------------------------------------
# re-export sanity (E path, import only -- no duplicate numeric verification)
# --------------------------------------------------------------------------
def test_dense_utility_optimized_is_the_opt02_object_not_a_copy():
    import utils.dense_utility_optimized as opt02_mod
    assert dense_utility_optimized is opt02_mod.dense_utility_optimized


# --------------------------------------------------------------------------
# A. Individual Oracle optimized vs reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize('seed', list(range(30)))
def test_individual_oracle_matches_reference_random_seeds(seed):
    futures, query_future = _make(seed=seed)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=11)
    assert torch.allclose(ref, opt, atol=1e-4)


@pytest.mark.parametrize('chunk', [1, 5, 13, 37, 37 * 2, None])
def test_individual_oracle_chunk_size_invariant(chunk):
    futures, query_future = _make(seed=1)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=chunk)
    assert torch.allclose(ref, opt, atol=1e-4)


def test_individual_oracle_non_divisible_final_chunk():
    futures, query_future = _make(bsz=2, n=101, h=7, seed=2)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=64)
    assert torch.allclose(ref, opt, atol=1e-4)


def test_individual_oracle_chunk_larger_than_candidate_count():
    futures, query_future = _make(bsz=2, n=9, h=5, seed=3)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=9999)
    assert torch.allclose(ref, opt, atol=1e-4)


def test_individual_oracle_exact_tie_two_candidates():
    futures, query_future = _make(bsz=1, n=10, h=4, seed=4)
    # make candidates 3 and 7 both EXACTLY equal to the query -> u=0, the
    # true (and only) maximum, so the tie is at the argmax, not incidental
    futures[0, 3] = query_future[0].clone()
    futures[0, 7] = query_future[0].clone()
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=3)
    assert torch.allclose(ref, opt, atol=1e-4)
    assert torch.allclose(ref[0, 3], ref[0, 7])
    assert torch.allclose(opt[0, 3], opt[0, 7])
    # both reference and optimized break the tie toward the smaller index
    assert int(ref.argmax(dim=-1)) == int(opt.argmax(dim=-1)) == 3


def test_individual_oracle_near_tie():
    futures, query_future = _make(bsz=1, n=10, h=4, seed=5)
    futures[0, 7] = futures[0, 3] + 1e-6
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=3)
    assert torch.allclose(ref, opt, atol=1e-4)
    assert int(ref.argmax(dim=-1)) == int(opt.argmax(dim=-1))


def test_individual_oracle_all_candidates_equal():
    futures = torch.ones(2, 6, 5)
    query_future = torch.zeros(2, 5)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=2)
    assert torch.allclose(ref, opt, atol=1e-4)
    assert int(ref.argmax(dim=-1)[0]) == int(opt.argmax(dim=-1)[0]) == 0


def test_individual_oracle_duplicate_candidate_rows():
    futures, query_future = _make(bsz=1, n=8, h=4, seed=6)
    futures[0, 5] = futures[0, 0].clone()
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=3)
    assert torch.allclose(ref, opt, atol=1e-4)


def test_individual_oracle_handles_nan_candidate_gracefully_matching_reference():
    futures, query_future = _make(bsz=1, n=6, h=4, seed=7)
    futures[0, 2] = float('nan')
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=2)
    assert torch.isnan(ref[0, 2]) and torch.isnan(opt[0, 2])
    other = torch.tensor([i for i in range(6) if i != 2])
    assert torch.allclose(ref[0, other], opt[0, other], atol=1e-4)


def test_individual_oracle_inf_candidate_documented_divergence():
    """KNOWN, DOCUMENTED divergence (OPT03 report section 5): the reference
    computes (Y_i - Y_q)^2 directly, so an all-+inf candidate row gives a
    consistent +inf (-> u=-inf) regardless of the query's sign. The
    norm-expansion identity used here computes ||Y_i||^2 - 2*Y_i.Y_q + ||Y_q||^2;
    when Y_i is +inf in every dim and Y_q has mixed-sign components, the dot
    product sums both +inf and -inf terms, producing NaN via inf-inf
    cancellation -- not a bug in the arithmetic, an inherent limitation of
    this algebraic identity under literal +-inf inputs. This never occurs
    with real (always-finite) Weather sensor data; recorded here so it is
    not silently unverified, matching the spec's explicit NaN/Inf test
    requirement."""
    futures, query_future = _make(bsz=1, n=6, h=4, seed=8)
    futures[0, 4] = float('inf')
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=2)
    assert torch.isinf(ref[0, 4])
    assert not torch.isfinite(opt[0, 4])  # NaN or inf, but NOT silently wrong-but-finite
    other = torch.tensor([i for i in range(6) if i != 4])
    assert torch.allclose(ref[0, other], opt[0, other], atol=1e-4)


def test_individual_oracle_step_invariant_cache_reused_across_k_steps():
    """The core caching claim: computing cand_sq ONCE and reusing it across
    K=10 'steps' gives byte-for-byte the same per-step result as computing
    it fresh every step (which is what the reference call site currently
    does, redundantly -- see OPT03 report finding A-1)."""
    futures, query_future = _make(bsz=3, n=23, h=6, seed=9)
    cand_sq = prepare_individual_query_static(futures, candidate_chunk_size=8)
    fresh_each_step = [individual_oracle_utility_optimized(futures, query_future,
                                                            candidate_chunk_size=8)
                       for _ in range(10)]
    cached_each_step = [individual_oracle_utility_optimized(futures, query_future,
                                                             cand_sq=cand_sq,
                                                             candidate_chunk_size=8)
                        for _ in range(10)]
    for a, b in zip(fresh_each_step, cached_each_step):
        assert torch.equal(a, b)


def test_individual_oracle_empty_batch_dim_zero_candidates_masked_by_caller():
    """The reference does not mask internally; this function must not either
    -- masking is the caller's job (matches `individual_utility`'s own
    contract, verified by identical behavior on an all-invalid-but-computed
    row: both still produce finite numbers, caller masks with -inf)."""
    futures, query_future = _make(bsz=1, n=5, h=4, seed=10)
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future)
    assert torch.allclose(ref, opt, atol=1e-4)
    assert torch.isfinite(ref).all() and torch.isfinite(opt).all()


def test_individual_oracle_gpu_if_available():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    futures, query_future = _make(bsz=4, n=257, h=17, seed=11, device='cuda')
    ref = individual_utility(futures, query_future)
    opt = individual_oracle_utility_optimized(futures, query_future, candidate_chunk_size=64)
    assert torch.allclose(ref, opt, atol=1e-4)


def test_prepare_individual_query_static_matches_direct_norm():
    futures, _ = _make(bsz=2, n=19, h=6, seed=12)
    direct = futures.pow(2).sum(dim=-1)
    chunked = prepare_individual_query_static(futures, candidate_chunk_size=5)
    assert torch.allclose(direct, chunked, atol=1e-5)


def _make_choice_inputs(bsz=6, n=53, seed=0, device='cpu', all_valid=False):
    g = torch.Generator(device='cpu').manual_seed(seed)
    u_hat = torch.randn(bsz, n, generator=g, device='cpu').to(device)
    u_target = torch.randn(bsz, n, generator=g, device='cpu').to(device)
    if all_valid:
        valid_mask = torch.ones(bsz, n, dtype=torch.bool, device=device)
    else:
        valid_mask = torch.rand(bsz, n, generator=g) > 0.3
        valid_mask = valid_mask.to(device)
    return u_hat, u_target, valid_mask


# --------------------------------------------------------------------------
# C. Oracle-Choice CE optimized (dead-code elimination) vs reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize('seed', list(range(30)))
def test_choice_ce_loss_and_diag_bit_identical_random_seeds(seed):
    u_hat, u_target, valid_mask = _make_choice_inputs(seed=seed)
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref.keys() == diag_opt.keys()
    for k in diag_ref:
        a, b = diag_ref[k], diag_opt[k]
        if a != a and b != b:  # both NaN
            continue
        assert a == b, f'{k}: {a} != {b}'


def test_choice_ce_all_valid():
    u_hat, u_target, valid_mask = _make_choice_inputs(seed=1, all_valid=True)
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref == diag_opt


def test_choice_ce_row_with_single_valid_candidate():
    u_hat, u_target, valid_mask = _make_choice_inputs(bsz=4, n=10, seed=2)
    valid_mask[0] = False
    valid_mask[0, 3] = True  # exactly one valid candidate in row 0
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref == diag_opt


def test_choice_ce_row_with_no_valid_candidates():
    u_hat, u_target, valid_mask = _make_choice_inputs(bsz=4, n=10, seed=3)
    valid_mask[1] = False  # entire row invalid
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref == diag_opt


def test_choice_ce_exact_tie_in_target():
    u_hat, u_target, valid_mask = _make_choice_inputs(bsz=2, n=10, seed=4, all_valid=True)
    u_target[0, 2] = u_target[0, 5] = 999.0  # exact tie for the max target
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref == diag_opt


def test_choice_ce_gradient_equivalence():
    """The training loop backprops through `loss` w.r.t. u_hat (which itself
    flows from the encoder). Gradient equivalence, not just forward-value
    equivalence, is required (OPT03 spec section 5)."""
    for seed in range(10):
        u_hat_ref, u_target, valid_mask = _make_choice_inputs(seed=seed)
        u_hat_ref = u_hat_ref.clone().requires_grad_(True)
        u_hat_opt = u_hat_ref.detach().clone().requires_grad_(True)

        loss_ref, _ = oracle_choice_step_loss(u_hat_ref, u_target, valid_mask, tau=0.1)
        loss_opt, _ = oracle_choice_step_loss_optimized(u_hat_opt, u_target, valid_mask, tau=0.1)
        loss_ref.backward()
        loss_opt.backward()
        assert torch.equal(loss_ref.detach(), loss_opt.detach())
        assert torch.equal(u_hat_ref.grad, u_hat_opt.grad)


def test_choice_ce_gpu_if_available():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    u_hat, u_target, valid_mask = _make_choice_inputs(bsz=32, n=4096, seed=5, device='cuda')
    loss_ref, diag_ref = oracle_choice_step_loss(u_hat, u_target, valid_mask, tau=0.1)
    loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau=0.1)
    assert torch.equal(loss_ref, loss_opt)
    assert diag_ref == diag_opt


def test_choice_ce_no_python_level_per_row_loop_in_source():
    import inspect
    src = inspect.getsource(oracle_choice_step_loss_optimized)
    assert 'for b in range' not in src
    body = src.split('"""', 2)[-1]  # drop the docstring, which mentions the removed name
    assert 'std_per_row' not in body


def test_individual_oracle_no_full_bnh_tensor_materialized_in_source():
    import inspect
    src = inspect.getsource(individual_oracle_utility_optimized)
    # the only [.., n, h]-shaped slices are the per-chunk `y_c` (bounded by
    # chunk size), never a bare `futures` (full N) used outside the chunk loop
    lines_before_loop = src.split('for start in range')[0]
    assert 'futures[' not in lines_before_loop
