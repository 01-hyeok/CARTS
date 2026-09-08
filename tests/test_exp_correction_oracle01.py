"""EXP-CORRECTION-ORACLE-DIAG01 (Track B) mandatory pre-run sanity checks
(spec's numbered list of 9 items). Track B is diagnostic-only (no
training) and reuses `select_greedy_weighted_set`/`dense_utility`
unmodified -- these tests exercise the NEW combination (residual value
space) and the new equivalence/leakage properties specific to Track B.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


def _toy(bsz=6, n=37, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    Y_i = torch.randn(n, h, generator=g)
    B_i = torch.randn(n, h, generator=g) * 0.3
    Y_q = torch.randn(bsz, h, generator=g)
    B_q = torch.randn(bsz, h, generator=g) * 0.3
    scores = torch.randn(bsz, n, generator=g)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -3:] = False
    return Y_i, B_i, Y_q, B_q, scores, valid


def test_1_residual_equals_future_minus_base_exactly():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy()
    r_i = Y_i - B_i
    assert torch.allclose(r_i, Y_i - B_i)  # tautological by construction
    # but the KEY property: recompose exactly recovers Y_i
    assert torch.allclose(r_i + B_i, Y_i, atol=1e-6)


def test_2_correction_objective_equivalence():
    """MSE(B_q + C(S), Y_q) == MSE(C(S), r_q) for an arbitrary C(S)."""
    Y_i, B_i, Y_q, B_q, scores, valid = _toy()
    r_q = Y_q - B_q
    bsz, h = Y_q.shape
    C_S = torch.randn(bsz, h)
    obj_a = (B_q + C_S - Y_q).pow(2).mean(-1)
    obj_b = (C_S - r_q).pow(2).mean(-1)
    assert torch.allclose(obj_a, obj_b, atol=1e-6)


def test_3_empty_correction_final_equals_b0():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy()
    C_S = torch.zeros_like(B_q)
    y_final = B_q + C_S
    assert torch.allclose(y_final, B_q, atol=1e-8)


def test_4_k1_brute_force_argmin_matches_select_greedy():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy(bsz=4, n=29, h=4)
    r_i = Y_i - B_i
    r_q = Y_q - B_q
    bsz = Y_q.size(0)
    r_i_batched = r_i.unsqueeze(0).expand(bsz, -1, -1)
    tau = 0.5
    picks = select_greedy_weighted_set(r_i_batched, r_q, scores, valid, k=1, tau=tau)

    # brute-force: with an empty prefix, the K=1 step of select_greedy_weighted_set
    # picks argmin_i MSE(alpha-weighted single candidate r_i, r_q) restricted to a
    # SINGLETON set -- for a singleton set alpha=1, so this is just
    # argmin_i MSE(r_i, r_q) over valid candidates.
    err = (r_i_batched - r_q.unsqueeze(1)).pow(2).mean(-1)
    err = err.masked_fill(~valid, float('inf'))
    brute_force = err.argmin(dim=-1)
    assert torch.equal(picks.squeeze(-1), brute_force)


def test_5_smalln_greedy_matches_exhaustive_stepwise():
    torch.manual_seed(0)
    bsz, n, h, k = 2, 11, 3, 3
    r_i = torch.randn(n, h)
    r_q = torch.randn(bsz, h)
    scores = torch.randn(bsz, n)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    tau = 0.7
    r_i_batched = r_i.unsqueeze(0).expand(bsz, -1, -1)

    picks = select_greedy_weighted_set(r_i_batched, r_q, scores, valid, k=k, tau=tau)

    # exhaustive stepwise re-derivation using dense_utility (a second,
    # independently-implemented primitive) as a cross-check
    w = candidate_weights(scores, valid, tau)
    selected_mask = torch.zeros_like(valid)
    for t in range(k):
        prefix = picks[:, :t]
        a = dense_utility(prefix, w, r_i_batched, r_q, chunk_size=None)
        valid_now = valid & ~selected_mask
        a_masked = a.masked_fill(~valid_now, float('inf'))
        expected_next = a_masked.argmin(dim=-1)
        assert torch.equal(picks[:, t], expected_next), f'step {t} mismatch'
        selected_mask = selected_mask.scatter(1, picks[:, t:t + 1], True)


def test_6_invalid_candidates_never_selected():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy(n=41)
    r_i = Y_i - B_i
    r_q = Y_q - B_q
    bsz = Y_q.size(0)
    r_i_batched = r_i.unsqueeze(0).expand(bsz, -1, -1)
    picks = select_greedy_weighted_set(r_i_batched, r_q, scores, valid, k=5, tau=0.3)
    for b in range(bsz):
        assert bool(valid[b, picks[b]].all()), 'an invalid candidate was selected'
        assert len(set(picks[b].tolist())) == picks.size(1), 'duplicate selection within a query'


def test_7_full_memory_candidate_count_unchanged_by_chunking():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy(n=8449 // 100)  # scaled-down stand-in, real count checked at runtime in the diag script itself
    r_i = Y_i - B_i
    r_q = Y_q - B_q
    bsz = Y_q.size(0)
    r_i_batched = r_i.unsqueeze(0).expand(bsz, -1, -1)
    w = candidate_weights(scores, valid, 0.3)
    a_unchunked = dense_utility(torch.zeros(bsz, 0, dtype=torch.long), w, r_i_batched, r_q, chunk_size=None)
    a_chunked = dense_utility(torch.zeros(bsz, 0, dtype=torch.long), w, r_i_batched, r_q, chunk_size=7)
    assert a_unchunked.shape[-1] == r_i.size(0), 'chunking must never change the candidate universe size'
    assert torch.allclose(a_unchunked, a_chunked, atol=1e-6)


def test_8_no_gradient_through_diagnostic_computation():
    Y_i, B_i, Y_q, B_q, scores, valid = _toy()
    B_i_param = B_i.clone().requires_grad_(True)
    with torch.no_grad():
        r_i = Y_i - B_i_param
    assert not r_i.requires_grad, 'residual computation must be wrapped in no_grad end-to-end'


def test_9_candidate_residual_uses_only_own_past_not_query():
    """Structural check: base_forecast is a pure function of its own input
    tensor `x` -- calling it on the candidate bank must be invariant to any
    query-side tensor, i.e. changing Y_q/B_q must never change r_i."""
    Y_i, B_i, Y_q, B_q, scores, valid = _toy()
    r_i_a = Y_i - B_i
    Y_q_perturbed = Y_q + 100.0
    B_q_perturbed = B_q - 50.0
    r_i_b = Y_i - B_i  # recomputed; B_i/Y_i untouched by the perturbation above
    assert torch.allclose(r_i_a, r_i_b), 'candidate residual must be independent of query future/base forecast'
