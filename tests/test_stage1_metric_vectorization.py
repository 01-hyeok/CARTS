"""Pin the vectorized Stage-1 metrics to the per-query reference.

The per-query loops were 50x slower than the tensor form, but they are the
definition of what these numbers mean. Every fast path here is checked against
its reference on the cases that actually break naive vectorization:

  * queries with no valid candidate  (the reference drops them from the mean)
  * queries with fewer valid candidates than k  (effective_k varies per query)
  * heavy ties  (topk order is not unique)
  * a single valid candidate  (entropy and Spearman are defined as 0)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import (
    _student_retrieval_metrics,
    _student_retrieval_metrics_reference,
)

TOL = dict(rtol=1e-5, atol=1e-6)


def _inputs(bsz, num_cand, valid, seed=0, tie=False):
    generator = torch.Generator().manual_seed(seed)
    if tie:
        scores = torch.randint(0, 3, (bsz, num_cand), generator=generator).float()
        distances = torch.randint(0, 3, (bsz, num_cand), generator=generator).float()
    else:
        scores = torch.randn(bsz, num_cand, generator=generator)
        distances = torch.rand(bsz, num_cand, generator=generator) + 0.05
    prob = torch.softmax(scores.masked_fill(~valid, float('-inf')), dim=-1)
    prob = torch.nan_to_num(prob, nan=0.0)
    return scores, prob, distances


def _compare(scores, prob, distances, valid):
    reference = _student_retrieval_metrics_reference(scores, prob, distances, valid)
    fast = _student_retrieval_metrics(scores, prob, distances, valid)
    assert set(fast) == set(reference), (
        f'key mismatch: only-fast={sorted(set(fast) - set(reference))} '
        f'only-ref={sorted(set(reference) - set(fast))}'
    )
    mismatched = {
        key: (float(reference[key]), float(fast[key]))
        for key in reference
        if not torch.allclose(fast[key].float(), reference[key].float(), **TOL)
    }
    assert not mismatched, f'vectorized metrics drifted from the reference: {mismatched}'


def test_dense_batch_matches_reference():
    valid = torch.ones(8, 200, dtype=torch.bool)
    _compare(*_inputs(8, 200, valid, seed=1), valid)


def test_partially_masked_batch_matches_reference():
    generator = torch.Generator().manual_seed(2)
    valid = torch.rand(8, 200, generator=generator) > 0.3
    valid[:, 0] = True
    _compare(*_inputs(8, 200, valid, seed=2), valid)


def test_queries_with_no_valid_candidate_are_dropped_from_the_mean():
    """The trap: averaging over all rows instead of the active ones."""
    valid = torch.ones(8, 60, dtype=torch.bool)
    valid[0] = False
    valid[5] = False
    _compare(*_inputs(8, 60, valid, seed=3), valid)


def test_fewer_valid_candidates_than_k():
    """effective_k = min(k, count) is per query, not per batch."""
    valid = torch.zeros(6, 80, dtype=torch.bool)
    for row, count in enumerate([1, 2, 3, 7, 9, 40]):
        valid[row, :count] = True
    _compare(*_inputs(6, 80, valid, seed=4), valid)


def test_single_valid_candidate_row():
    valid = torch.zeros(4, 50, dtype=torch.bool)
    valid[:, 0] = True
    valid[3, :12] = True
    _compare(*_inputs(4, 50, valid, seed=5), valid)


def test_heavy_ties():
    valid = torch.ones(8, 120, dtype=torch.bool)
    _compare(*_inputs(8, 120, valid, seed=6, tie=True), valid)


def test_heavy_ties_with_mask():
    generator = torch.Generator().manual_seed(7)
    valid = torch.rand(8, 120, generator=generator) > 0.4
    valid[:, :15] = True
    _compare(*_inputs(8, 120, valid, seed=7, tie=True), valid)


def test_all_queries_invalid_returns_zeros():
    valid = torch.zeros(4, 30, dtype=torch.bool)
    scores, prob, distances = _inputs(4, 30, valid, seed=8)
    reference = _student_retrieval_metrics_reference(scores, prob, distances, valid)
    fast = _student_retrieval_metrics(scores, prob, distances, valid)
    assert set(fast) == set(reference)
    for key in reference:
        assert float(fast[key]) == 0.0, key


@pytest.mark.parametrize('seed', range(12))
def test_randomized_sweep_matches_reference(seed):
    generator = torch.Generator().manual_seed(100 + seed)
    bsz = int(torch.randint(1, 10, (1,), generator=generator))
    num_cand = int(torch.randint(3, 300, (1,), generator=generator))
    keep = float(torch.rand(1, generator=generator)) * 0.9 + 0.05
    valid = torch.rand(bsz, num_cand, generator=generator) < keep
    tie = seed % 3 == 0
    scores, prob, distances = _inputs(bsz, num_cand, valid, seed=200 + seed, tie=tie)
    if not bool(valid.any()):
        pytest.skip('degenerate draw with no valid candidate at all')
    _compare(scores, prob, distances, valid)


# --------------------------------------------------------------------------
# _ranking_source_topk_metrics
# --------------------------------------------------------------------------

from models.RelationStage1 import (  # noqa: E402
    _ranking_source_topk_metrics,
    _ranking_source_topk_metrics_reference,
)


def _ranking_inputs(bsz, num_cand, seed, tie=False):
    generator = torch.Generator().manual_seed(seed)
    if tie:
        make = lambda: torch.randint(0, 3, (bsz, num_cand), generator=generator).float()
    else:
        make = lambda: torch.randn(bsz, num_cand, generator=generator)
    return make(), make(), make().abs(), make()


def _compare_ranking(student, teacher, mse, cos, valid):
    reference = _ranking_source_topk_metrics_reference(student, teacher, mse, cos, valid)
    fast = _ranking_source_topk_metrics(student, teacher, mse, cos, valid)
    assert set(fast) == set(reference)
    mismatched = {
        key: (float(reference[key]), float(fast[key]))
        for key in reference
        if not torch.allclose(fast[key].float(), reference[key].float(), **TOL)
    }
    assert not mismatched, f'ranking-source metrics drifted: {mismatched}'


def test_ranking_sources_dense_batch():
    valid = torch.ones(8, 150, dtype=torch.bool)
    _compare_ranking(*_ranking_inputs(8, 150, seed=11), valid)


def test_ranking_sources_with_mask_and_short_rows():
    valid = torch.zeros(6, 90, dtype=torch.bool)
    for row, count in enumerate([1, 2, 4, 8, 30, 90]):
        valid[row, :count] = True
    _compare_ranking(*_ranking_inputs(6, 90, seed=12), valid)


def test_ranking_sources_drops_empty_queries():
    valid = torch.ones(6, 70, dtype=torch.bool)
    valid[2] = False
    _compare_ranking(*_ranking_inputs(6, 70, seed=13), valid)


def test_ranking_sources_heavy_ties():
    generator = torch.Generator().manual_seed(14)
    valid = torch.rand(8, 100, generator=generator) > 0.4
    valid[:, :12] = True
    _compare_ranking(*_ranking_inputs(8, 100, seed=14, tie=True), valid)


def test_ranking_sources_all_invalid_returns_zeros():
    valid = torch.zeros(4, 40, dtype=torch.bool)
    student, teacher, mse, cos = _ranking_inputs(4, 40, seed=15)
    fast = _ranking_source_topk_metrics(student, teacher, mse, cos, valid)
    reference = _ranking_source_topk_metrics_reference(student, teacher, mse, cos, valid)
    assert set(fast) == set(reference)
    assert all(float(v) == 0.0 for v in fast.values())


@pytest.mark.parametrize('seed', range(8))
def test_ranking_sources_randomized_sweep(seed):
    generator = torch.Generator().manual_seed(300 + seed)
    bsz = int(torch.randint(1, 8, (1,), generator=generator))
    num_cand = int(torch.randint(3, 220, (1,), generator=generator))
    keep = float(torch.rand(1, generator=generator)) * 0.9 + 0.05
    valid = torch.rand(bsz, num_cand, generator=generator) < keep
    if not bool(valid.any()):
        pytest.skip('degenerate draw with no valid candidate at all')
    _compare_ranking(*_ranking_inputs(bsz, num_cand, seed=400 + seed, tie=seed % 2 == 0), valid)


# --------------------------------------------------------------------------
# _teacher_student_distribution_metrics
# --------------------------------------------------------------------------

from models.RelationStage1 import (  # noqa: E402
    _teacher_student_distribution_metrics,
    _teacher_student_distribution_metrics_reference,
)


def _dist_inputs(bsz, num_cand, valid, seed, tie=False):
    generator = torch.Generator().manual_seed(seed)
    if tie:
        raw_t = torch.randint(0, 3, (bsz, num_cand), generator=generator).float()
        raw_s = torch.randint(0, 3, (bsz, num_cand), generator=generator).float()
    else:
        raw_t = torch.randn(bsz, num_cand, generator=generator)
        raw_s = torch.randn(bsz, num_cand, generator=generator)
    masked_t = raw_t.masked_fill(~valid, float('-inf'))
    masked_s = raw_s.masked_fill(~valid, float('-inf'))
    teacher = torch.nan_to_num(torch.softmax(masked_t, dim=-1), nan=0.0)
    student = torch.nan_to_num(torch.softmax(masked_s, dim=-1), nan=0.0)
    return teacher, student


def _compare_dist(teacher, student, valid):
    reference = _teacher_student_distribution_metrics_reference(teacher, student, valid)
    fast = _teacher_student_distribution_metrics(teacher, student, valid)
    assert set(fast) == set(reference)
    mismatched = {
        key: (float(reference[key]), float(fast[key]))
        for key in reference
        if not torch.allclose(fast[key].float(), reference[key].float(), **TOL)
    }
    assert not mismatched, f'distribution metrics drifted: {mismatched}'


def test_distribution_dense_batch():
    valid = torch.ones(8, 150, dtype=torch.bool)
    _compare_dist(*_dist_inputs(8, 150, valid, seed=21), valid)


def test_distribution_with_mask_and_short_rows():
    valid = torch.zeros(6, 90, dtype=torch.bool)
    for row, count in enumerate([1, 2, 4, 8, 30, 90]):
        valid[row, :count] = True
    _compare_dist(*_dist_inputs(6, 90, valid, seed=22), valid)


def test_distribution_drops_empty_queries():
    valid = torch.ones(6, 70, dtype=torch.bool)
    valid[1] = False
    valid[4] = False
    _compare_dist(*_dist_inputs(6, 70, valid, seed=23), valid)


def test_distribution_heavy_ties():
    generator = torch.Generator().manual_seed(24)
    valid = torch.rand(8, 100, generator=generator) > 0.4
    valid[:, :12] = True
    _compare_dist(*_dist_inputs(8, 100, valid, seed=24, tie=True), valid)


def test_distribution_all_invalid_returns_zeros():
    valid = torch.zeros(4, 40, dtype=torch.bool)
    teacher, student = _dist_inputs(4, 40, valid, seed=25)
    fast = _teacher_student_distribution_metrics(teacher, student, valid)
    reference = _teacher_student_distribution_metrics_reference(teacher, student, valid)
    assert set(fast) == set(reference)
    assert all(float(v) == 0.0 for v in fast.values())


@pytest.mark.parametrize('seed', range(8))
def test_distribution_randomized_sweep(seed):
    generator = torch.Generator().manual_seed(500 + seed)
    bsz = int(torch.randint(1, 8, (1,), generator=generator))
    num_cand = int(torch.randint(3, 220, (1,), generator=generator))
    keep = float(torch.rand(1, generator=generator)) * 0.9 + 0.05
    valid = torch.rand(bsz, num_cand, generator=generator) < keep
    if not bool(valid.any()):
        pytest.skip('degenerate draw with no valid candidate at all')
    _compare_dist(*_dist_inputs(bsz, num_cand, valid, seed=600 + seed, tie=seed % 2 == 0), valid)
