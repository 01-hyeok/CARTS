"""EXP-CONTINUATION-DIAG: exhaustive t=2 continuation diagnostic invariants.

No checkpoint/GPU needed -- toy-data structural/math tests, mirroring
tests/test_exp_firstanchor_diag.py's style.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_firstanchor_diag import a_weighted_prefix
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


def _toy(bsz=3, n=20, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    futures = torch.randn(bsz, n, h, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    scores = torch.randn(bsz, n, generator=g) * 0.5
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -2:] = False
    return futures, query_future, scores, valid


def test_dense_utility_a_of_s1_plus_i_matches_a_weighted_prefix_two_element():
    """Sanity #1/#3: A(S1+{i}) via the exhaustive closed form must equal
    A({i1,i}) via the prefix-softmax definition every other arm in this
    project uses -- two independent code paths, same number."""
    futures, query_future, scores, valid = _toy()
    tau = 0.1
    i1 = torch.tensor([0, 1, 2])
    w = candidate_weights(scores, valid, tau)
    a2_all = dense_utility(i1.unsqueeze(1), w, futures, query_future)
    for cand in (3, 4, 5):
        cand_t = torch.full((3,), cand, dtype=torch.long)
        prefix2 = torch.stack([i1, cand_t], dim=1)
        expected = a_weighted_prefix(prefix2, scores, futures, query_future, tau)
        got = a2_all.gather(1, cand_t.unsqueeze(-1)).squeeze(1)
        assert torch.allclose(got, expected, atol=1e-5), f'mismatch at candidate {cand}'


def test_i2_oracle_is_exhaustive_argmin_over_remaining_valid():
    """Sanity #3/#5: i2_oracle must be the TRUE minimum over every valid
    candidate excluding i1 itself, verified against a brute-force loop."""
    futures, query_future, scores, valid = _toy(bsz=2, n=15)
    tau = 0.1
    i1 = torch.tensor([2, 5])
    w = candidate_weights(scores, valid, tau)
    a2_all = dense_utility(i1.unsqueeze(1), w, futures, query_future)
    selected = torch.zeros_like(valid).scatter(1, i1.unsqueeze(1), True)
    remaining = valid & ~selected
    a2_masked = a2_all.masked_fill(~remaining, float('inf'))
    i2_oracle = a2_masked.argmin(dim=-1)

    for b in range(2):
        best_i, best_a = None, float('inf')
        for i in range(15):
            if not remaining[b, i]:
                continue
            prefix2 = torch.tensor([[i1[b], i]])
            a = float(a_weighted_prefix(prefix2, scores[b:b + 1], futures[b:b + 1],
                                         query_future[b:b + 1], tau))
            if a < best_a:
                best_a, best_i = a, i
        assert int(i2_oracle[b]) == best_i, f'row {b}: brute force found {best_i}, got {int(i2_oracle[b])}'


def test_i1_excluded_from_second_candidate_pool():
    """Sanity #5: the first candidate must never be re-selectable as i2."""
    futures, query_future, scores, valid = _toy(bsz=4, n=12)
    tau = 0.1
    i1 = torch.tensor([3, 7, 0, 11])
    w = candidate_weights(scores, valid, tau)
    a2_all = dense_utility(i1.unsqueeze(1), w, futures, query_future)
    selected = torch.zeros_like(valid).scatter(1, i1.unsqueeze(1), True)
    remaining = valid & ~selected
    a2_masked = a2_all.masked_fill(~remaining, float('inf'))
    i2 = a2_masked.argmin(dim=-1)
    assert torch.all(i2 != i1)


def test_true_gain_sign_convention_matches_higher_is_better():
    """true_gain(i) = A1 - A(S1+{i}); larger gain must correspond to a
    SMALLER (better) aggregate A, matching the Dense model's own u_hat sign
    convention (u = -A, higher = better) so the Spearman comparison in the
    main script is not silently sign-flipped."""
    futures, query_future, scores, valid = _toy(bsz=1, n=10)
    tau = 0.1
    i1 = torch.tensor([0])
    w = candidate_weights(scores, valid, tau)
    a1 = a_weighted_prefix(i1.unsqueeze(1), scores, futures, query_future, tau)
    a2_all = dense_utility(i1.unsqueeze(1), w, futures, query_future)
    true_gain = a1.unsqueeze(-1) - a2_all
    # the candidate with the smallest A2 must have the LARGEST true_gain
    selected = torch.zeros_like(valid).scatter(1, i1.unsqueeze(1), True)
    remaining = valid[0] & ~selected[0]
    a2_masked = a2_all[0].masked_fill(~remaining, float('inf'))
    best_by_a2 = int(a2_masked.argmin())
    best_by_gain = int(true_gain[0].masked_fill(~remaining, float('-inf')).argmax())
    assert best_by_a2 == best_by_gain


def test_oracle_first_i1_matches_greedy_oracle_first_pick():
    """Sanity #2: oracle_first's t=1 must match the cached Weighted Set
    Oracle's own first pick -- same identity EXP-FIRSTANCHOR-DIAG already
    established, re-verified at this script's own call site."""
    futures, query_future, scores, valid = _toy(n=25)
    tau = 0.1
    oracle_seq = select_greedy_weighted_set(futures, query_future, scores, valid, k=2, tau=tau)
    dist = (futures - query_future.unsqueeze(1)).pow(2).mean(-1).masked_fill(~valid, float('inf'))
    i1_oracle = dist.argmin(dim=-1)
    assert torch.equal(i1_oracle, oracle_seq[:, 0])
