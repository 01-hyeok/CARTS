"""Boundary hard-pair ranking loss and the diagnostics that read it.

The loss supervises pairs the retriever currently orders wrongly at the Top-K
edge. Two properties decide whether it can work at all: the hinge must fire on
exactly the wrong orderings, and its gradient must reach the encoder through
both the query and the candidate side -- a detached candidate score would train
only half the model and nothing would raise an error.
"""

import pytest
import torch

from models.RelationStage1 import (
    boundary_hard_rank_loss, ranking_diagnostics, score_geometry,
)


def _case():
    """Two candidates: index 0 scores higher, index 1 has the better future."""
    scores = torch.tensor([[0.9, 0.3]], requires_grad=True)
    future = torch.tensor([[1.0, 0.2]])          # d_0 = 1.0, d_1 = 0.2
    mask = torch.ones(1, 2, dtype=torch.bool)
    return scores, future, mask


def test_a_wrong_ordering_costs_something():
    scores, future, mask = _case()
    loss, m = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01, pairs_per_query=8)
    assert float(loss) > 0
    assert float(m['num_valid_rank_pairs']) == 1
    assert float(m['rank_loss_active_fraction']) == 1.0


def test_a_correct_ordering_beyond_the_margin_is_free():
    scores = torch.tensor([[0.3, 0.9]])          # better future already scores higher
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, _ = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01, pairs_per_query=8)
    assert float(loss) == 0.0


def test_near_ties_are_excluded_by_the_gap_threshold():
    scores = torch.tensor([[0.9, 0.3]])
    future = torch.tensor([[0.51, 0.50]])        # a 0.01 future gap
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, m = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01, gap_threshold=0.1)
    assert float(loss) == 0.0
    assert float(m['num_valid_rank_pairs']) == 0


def test_gradient_reaches_both_sides_of_the_pair():
    """Detaching the candidate score would leave this half zero and stay silent."""
    scores, future, mask = _case()
    loss, _ = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01)
    loss.backward()
    grad = scores.grad[0]
    assert grad[0] != 0, 'the selected candidate got no gradient'
    assert grad[1] != 0, 'the mined candidate got no gradient (detached score?)'
    # Push the wrongly-preferred one down and the better one up.
    assert grad[0] > 0 and grad[1] < 0


def test_gap_weighting_prefers_the_costlier_inversion():
    # The pool scores must differ, or every pair carries the same hinge value
    # and weighting cannot change the mean whatever the gaps are.
    scores = torch.tensor([[0.9, 0.5, 0.3, 0.45]])
    future = torch.tensor([[1.0, 0.51, 0.3, 0.50]])   # pair gaps 0.7 down to 0.01
    mask = torch.ones(1, 4, dtype=torch.bool)
    kw = dict(top_k=2, pool_end=4, margin=0.01, pairs_per_query=4)
    weighted, _ = boundary_hard_rank_loss(scores, future, mask, gap_weighted=True, **kw)
    plain, _ = boundary_hard_rank_loss(scores, future, mask, gap_weighted=False, **kw)
    assert not torch.isclose(weighted, plain)


def test_missed_better_counts_what_the_topk_left_behind():
    scores = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    future = torch.tensor([[1.0, 2.0, 0.5, 3.0]])     # index 2 beats both selected
    mask = torch.ones(1, 4, dtype=torch.bool)
    d = ranking_diagnostics(scores, future, mask, top_k=2, pool_end=4)
    assert float(d['missed_better_100_mean']) == 1.0
    assert float(d['oracle_in_model_top10_frac']) == 0.5


def test_pair_order_accuracy_is_not_the_degenerate_inversion_rate():
    """A rate over pools split by score is identically 1; this must not be."""
    torch.manual_seed(0)
    scores = torch.randn(2, 40)
    future = torch.rand(2, 40)
    mask = torch.ones(2, 40, dtype=torch.bool)
    acc = float(ranking_diagnostics(scores, future, mask, top_k=10,
                                    pool_end=40)['pair_order_accuracy_top100'])
    assert 0.0 < acc < 1.0
    perfect = ranking_diagnostics(-future, future, mask, top_k=10, pool_end=40)
    assert float(perfect['pair_order_accuracy_top100']) == pytest.approx(1.0)


def test_score_geometry_reports_a_positive_boundary_gap():
    torch.manual_seed(0)
    scores = torch.randn(4, 200)
    mask = torch.ones(4, 200, dtype=torch.bool)
    g = score_geometry(scores, mask, top_k=10, pool_end=100)
    assert float(g['score_rank1_mean']) > float(g['score_rank10_mean'])
    assert float(g['rank10_rank11_score_gap_mean']) > 0
    assert float(g['hard_pair_score_gap_p25']) <= float(g['hard_pair_score_gap_p75'])


def test_reports_how_often_the_two_objectives_disagree():
    """A mined candidate outside the Oracle Top-K is one the cross-entropy is
    pushing down while this loss pushes it up."""
    scores = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    future = torch.tensor([[1.0, 2.0, 0.5, 0.4]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    # Oracle Top-2 by future MSE is {3 (0.4), 2 (0.5)}; both mined j are in it.
    _, m = boundary_hard_rank_loss(scores, future, mask, top_k=2, pool_end=4,
                                   margin=0.01, pairs_per_query=8)
    assert float(m['rank_positive_outside_oracle_frac']) == 0.0
    # Now the better-than-selected candidates sit outside the Oracle Top-1.
    future2 = torch.tensor([[1.0, 2.0, 0.5, 0.4]])
    _, m2 = boundary_hard_rank_loss(scores, future2, mask, top_k=1, pool_end=4,
                                    margin=0.01, pairs_per_query=8)
    assert 0.0 < float(m2['rank_positive_outside_oracle_frac']) <= 1.0


def test_invalid_candidates_are_never_mined():
    scores = torch.tensor([[0.9, 0.3]])
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.tensor([[True, False]])          # the better one is masked out
    loss, m = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01)
    assert float(loss) == 0.0
    assert float(m['num_valid_rank_pairs']) == 0
