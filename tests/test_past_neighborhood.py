"""The ambiguity estimator has to be able to detect information when it exists.

A null result from this diagnostic is only believable if the same code reports a
strong result on data where the past provably determines the residual. Both
directions are tested here on synthetic windows.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.past_neighborhood import (
    bucket_widths, finalize, merge_sums, neighborhood_statistics, pair_mse, znorm,
)

FRACTIONS = (0.0, 0.01, 0.05, 0.10, 1.0)
METRICS = ('past_distance', 'residual_pair_mse', 'future_pair_mse',
           'past_tail_pair_mse', 'residual_cosine', 'knn_residual_mse',
           'shuffled_residual_pair_mse')
L, T, N, B = 16, 8, 400, 40


def _run(query_past, memory_past, query_residual, memory_residual, **kwargs):
    valid = torch.ones(query_past.size(0), memory_past.size(0), dtype=torch.bool)
    sums = neighborhood_statistics(
        query_past, memory_past, query_residual, memory_residual,
        query_residual, memory_residual, valid, FRACTIONS,
        shuffled_residual=memory_residual[torch.randperm(memory_residual.size(0))],
        **kwargs,
    )
    return finalize(merge_sums({}, sums), FRACTIONS, METRICS)


def test_detects_a_past_determined_residual():
    """Positive case: residual is a function of the past, so it must collapse."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = torch.randn(B, L)
    to_residual = lambda past: past[:, :T] * 2.0
    rows = _run(query_past, memory_past, to_residual(query_past),
                to_residual(memory_past))
    assert rows[0.0]['residual_pair_mse_ratio'] < 0.5
    assert rows[0.01]['residual_pair_mse_ratio'] < 1.0
    # The neighborhood mean recovers most of the residual.
    residual_power = float(to_residual(query_past).square().mean())
    assert rows[0.01]['knn_residual_mse'] < 0.5 * residual_power


def test_reports_ambiguity_when_the_past_says_nothing():
    """Null case: residual independent of the past, so nothing may collapse."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = torch.randn(B, L)
    rows = _run(query_past, memory_past, torch.randn(B, T), torch.randn(N, T))
    assert 0.8 < rows[0.01]['residual_pair_mse_ratio'] < 1.25
    assert rows[0.01]['residual_cosine_dispersion'] > 0.8


def test_positive_control_collapses_in_both_cases():
    """The past's own tail must collapse whatever the residual does."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = memory_past[:B] + 0.01 * torch.randn(B, L)
    rows = _run(query_past, memory_past, torch.randn(B, T), torch.randn(N, T))
    # Only the nearest-1 bucket is a genuine near-duplicate; a 1% percentile of
    # 400 random windows is still four mostly-unrelated neighbors.
    assert rows[0.0]['past_tail_pair_mse_ratio'] < 0.1
    assert rows[0.0]['past_distance_ratio'] < 0.1
    assert rows[0.01]['past_tail_pair_mse_ratio'] < rows[1.0]['past_tail_pair_mse_ratio']


def test_negative_control_tracks_the_global_level():
    """Permuted residuals must not collapse; that would mean small-group bias."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = torch.randn(B, L)
    to_residual = lambda past: past[:, :T] * 2.0
    rows = _run(query_past, memory_past, to_residual(query_past),
                to_residual(memory_past))
    assert rows[0.01]['shuffled_residual_pair_mse_ratio'] > 0.8


def test_best_candidate_entropy_separates_agreement_from_disagreement():
    """Entropy is scored on a fixed-size random sample of each bucket, so the
    numbers stay comparable between a 1% and a 100% neighborhood."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = torch.randn(B, L)
    residual_q, residual_m = torch.randn(B, T), torch.randn(N, T)
    valid = torch.ones(B, N, dtype=torch.bool)

    everyone_agrees = torch.zeros(N, dtype=torch.long)
    everyone_differs = torch.arange(N) % 50
    agree = finalize(merge_sums({}, neighborhood_statistics(
        query_past, memory_past, residual_q, residual_m, residual_q, residual_m,
        valid, FRACTIONS, best_id=everyone_agrees, num_identities=50,
        entropy_sample=8,
    )), FRACTIONS, ('best_candidate_entropy',))
    differ = finalize(merge_sums({}, neighborhood_statistics(
        query_past, memory_past, residual_q, residual_m, residual_q, residual_m,
        valid, FRACTIONS, best_id=everyone_differs, num_identities=50,
        entropy_sample=8,
    )), FRACTIONS, ('best_candidate_entropy',))
    assert agree[0.10]['best_candidate_entropy'] == 0.0
    assert differ[0.10]['best_candidate_entropy'] > 0.7
    # A bucket narrower than the sample size is reported as missing rather than
    # scored on fewer members, which would not be comparable.
    import math as _math
    assert _math.isnan(agree[0.01]['best_candidate_entropy'])


def test_invalid_neighbors_are_never_used():
    """The temporal validity mask still gates the neighborhood."""
    torch.manual_seed(0)
    memory_past = torch.randn(N, L)
    query_past = memory_past[:B].clone()
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, :B] = False                       # the exact matches are excluded
    residual_m = torch.randn(N, T)
    residual_m[:B] = 1000.0                    # poisoned, and nearest by past
    sums = neighborhood_statistics(
        query_past, memory_past, torch.randn(B, T), residual_m,
        torch.randn(B, T), residual_m, valid, (0.01, 1.0),
    )
    assert sums[(0.01, 'residual_pair_mse')] / sums[(0.01, 'residual_pair_mse__count')] < 100.0


def test_bucket_widths_and_znorm():
    counts = torch.tensor([100, 7, 1])
    widths = bucket_widths(counts, (0.01, 0.10))
    assert widths[0.01].tolist() == [1, 1, 1]
    assert widths[0.10].tolist() == [10, 1, 1]
    x = torch.randn(5, 20) * 3.0 + 7.0
    normalised = znorm(x)
    assert torch.allclose(normalised.mean(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(normalised.std(-1), torch.ones(5), atol=1e-2)
    assert pair_mse(x[:2], x[:2]).diagonal().abs().max() < 1e-4
