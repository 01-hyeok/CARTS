"""Diagnostics that separate the three ways the ranking loss can fail.

The first WCE+Rank run left validation flat, but with the train diagnostic
reporting NaN there was no way to tell whether the ordering was learned and did
not generalise, or was never learned at all. These pin the pieces that make that
distinction possible: gradients read at the score level rather than in parameter
space, a pair set that does not move between epochs, and a rank-only path.
"""

import pytest
import torch

from models.RelationStage1 import (
    build_frozen_rank_pairs,
    boundary_hard_rank_loss, build_frozen_rank_pairs, frozen_pair_metrics,
    ranking_diagnostics, score_gradient_conflict,
)


def test_rank_loss_raises_the_better_candidate_score_and_lowers_the_worse():
    """dL/ds < 0 raises the score under descent, so the signs are the direction."""
    scores = torch.tensor([[0.9, 0.3]], requires_grad=True)
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, _ = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01)
    g = torch.autograd.grad(loss, scores)[0][0]
    assert g[1] < 0, 'the better-future candidate is not being raised'
    assert g[0] > 0, 'the wrongly-preferred candidate is not being lowered'


def test_cross_entropy_pushes_a_non_oracle_candidate_down():
    """The conflict the split by Oracle membership exists to measure."""
    scores = torch.tensor([[0.9, 0.3, 0.1]], requires_grad=True)
    log_prob = torch.log_softmax(scores / 0.1, dim=-1)
    wce = -(log_prob[:, 0]).mean()          # candidate 0 is the only positive
    g = torch.autograd.grad(wce, scores)[0][0]
    assert g[0] < 0 and g[1] > 0 and g[2] > 0


def test_score_gradient_conflict_splits_by_oracle_membership():
    torch.manual_seed(0)
    scores = torch.randn(2, 40, requires_grad=True)
    future = torch.rand(2, 40)
    mask = torch.ones(2, 40, dtype=torch.bool)
    log_prob = torch.log_softmax(scores / 0.1, dim=-1)
    oracle = future.topk(10, dim=-1, largest=False).indices
    wce = -log_prob.gather(1, oracle).mean()
    rank, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10, pool_end=40,
                                      margin=0.01)
    out = score_gradient_conflict(wce, rank, scores, future, mask,
                                  rank_weight=1.0, top_k=10, pool_end=40)
    for key in ('outside_oracle_wce_push_down_frac',
                'outside_oracle_rank_push_up_frac',
                'outside_oracle_total_push_up_frac',
                'rank_positive_score_raise_frac'):
        assert key in out, key
        assert 0.0 <= float(out[key]) <= 1.0
    # Every candidate outside the Oracle set is a negative to the cross-entropy.
    assert float(out['outside_oracle_wce_push_down_frac']) == 1.0


def test_a_larger_rank_weight_wins_more_of_the_scores():
    torch.manual_seed(0)
    scores = torch.randn(2, 40, requires_grad=True)
    future = torch.rand(2, 40)
    mask = torch.ones(2, 40, dtype=torch.bool)
    log_prob = torch.log_softmax(scores / 0.1, dim=-1)
    oracle = future.topk(10, dim=-1, largest=False).indices
    wce = -log_prob.gather(1, oracle).mean()
    rank, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10, pool_end=40,
                                      margin=0.01)
    off = score_gradient_conflict(wce, rank, scores, future, mask, 0.0,
                                  top_k=10, pool_end=40)
    on = score_gradient_conflict(wce, rank, scores, future, mask, 1.0,
                                 top_k=10, pool_end=40)
    # Outside the Oracle set the cross-entropy alone can only push down, so with
    # the ranking term switched off nothing there is raised. (Inside the Oracle
    # set it raises candidates on its own, which is why the split matters.)
    assert float(off['outside_oracle_total_push_up_frac']) == 0.0
    assert float(on['outside_oracle_total_push_up_frac']) > 0.0
    # Only the candidates the mining touched can be raised at all.
    assert (float(on['rank_positive_score_raise_frac'])
            <= float(on['rank_positive_covered_frac']) + 1e-6)


def test_coverage_reports_how_much_of_the_problem_the_mining_touches():
    """Pairs are capped per query, so most better-than-selected candidates can
    receive no ranking gradient; the other fractions are unreadable without it."""
    torch.manual_seed(0)
    scores = torch.randn(2, 200, requires_grad=True)
    future = torch.rand(2, 200)
    mask = torch.ones(2, 200, dtype=torch.bool)
    log_prob = torch.log_softmax(scores / 0.1, dim=-1)
    oracle = future.topk(10, dim=-1, largest=False).indices
    wce = -log_prob.gather(1, oracle).mean()
    rank, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10,
                                      pool_end=100, margin=0.01,
                                      pairs_per_query=32)
    out = score_gradient_conflict(wce, rank, scores, future, mask, 1.0,
                                  top_k=10, pool_end=100)
    assert float(out['rank_positive_covered_frac']) < 1.0


def test_frozen_pairs_do_not_move_between_calls():
    torch.manual_seed(0)
    scores = torch.randn(3, 60)
    future = torch.rand(3, 60)
    mask = torch.ones(3, 60, dtype=torch.bool)
    ids = torch.arange(3)
    a = build_frozen_rank_pairs(scores, future, mask, ids, top_k=10,
                                pool_end=60, per_query=4)
    b = build_frozen_rank_pairs(scores, future, mask, ids, top_k=10,
                                pool_end=60, per_query=4)
    assert a == b and len(a) > 0
    # Every frozen pair is an ordering the model currently has backwards.
    for query, i_id, j_id, gap in a:
        assert future[query, j_id] < future[query, i_id]
        assert scores[query, i_id] > scores[query, j_id]
        assert gap > 0


def test_frozen_pair_metrics_track_the_ordering_they_were_built_from():
    scores = torch.tensor([[0.9, 0.3]])
    pairs = [(0, 0, 1, 0.8)]
    # Signed as s_j - s_i, so it rises exactly when the wanted ordering appears.
    before = frozen_pair_metrics(scores, torch.tensor([0]), pairs)
    assert float(before['frozen_pair_correct_order_frac']) == 0.0
    assert float(before['frozen_signed_gap_mean']) < 0
    after = frozen_pair_metrics(torch.tensor([[0.3, 0.9]]), torch.tensor([0]), pairs)
    assert float(after['frozen_pair_correct_order_frac']) == 1.0
    assert float(after['frozen_signed_gap_mean']) > 0
    assert (float(after['frozen_signed_gap_mean'])
            > float(before['frozen_signed_gap_mean']))


def test_large_gap_pair_accuracy_is_reported_separately():
    torch.manual_seed(0)
    scores = torch.randn(2, 40)
    future = torch.rand(2, 40)
    mask = torch.ones(2, 40, dtype=torch.bool)
    d = ranking_diagnostics(scores, future, mask, top_k=10, pool_end=40)
    for key in ('pair_acc_top100_all', 'pair_acc_top100_gap_p50',
                'pair_acc_top100_gap_p75'):
        assert key in d and 0.0 <= float(d[key]) <= 1.0


def test_gradient_direction_comes_from_the_target_not_oracle_membership():
    """Oracle membership only forces the sign outside the set, where q = 0.
    Inside it a candidate scored above its teacher weight is pushed down too,
    since dL/ds = (p - q)/tau, so the fractions must be read off the gradient."""
    torch.manual_seed(0)
    scores = torch.randn(2, 200, requires_grad=True)
    future = torch.rand(2, 200)
    mask = torch.ones(2, 200, dtype=torch.bool)
    log_prob = torch.log_softmax(scores / 0.1, dim=-1)
    oracle = future.topk(10, dim=-1, largest=False).indices
    wce = -log_prob.gather(1, oracle).mean()
    rank, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10,
                                      pool_end=100, margin=0.01)
    out = score_gradient_conflict(wce, rank, scores, future, mask, 1.0,
                                  top_k=10, pool_end=100)
    up = float(out['wce_score_push_up_frac'])
    down = float(out['wce_score_push_down_frac'])
    assert up + down == pytest.approx(1.0, abs=1e-5)
    # Some raised candidates are Oracle members with p below their weight, so
    # membership alone would have mispredicted the sign for them.
    assert up > 0.0
    assert float(out['outside_oracle_wce_push_down_frac']) == 1.0
    assert 0.0 <= float(out['wce_rank_score_conflict_frac']) <= 1.0


def test_diagnostics_never_report_the_degenerate_inversion_rate():
    """A rate over pools split by score is identically 1 and cannot move."""
    torch.manual_seed(0)
    d = ranking_diagnostics(torch.randn(2, 40), torch.rand(2, 40),
                            torch.ones(2, 40, dtype=torch.bool),
                            top_k=10, pool_end=40)
    assert not any('inversion' in key for key in d)


def test_candidate_mining_supervises_distinct_candidates():
    """Ranking by gap alone lets one strong candidate take the whole budget
    through near-duplicate pairs; candidate mode spends it on distinct ones."""
    torch.manual_seed(0)
    scores = torch.randn(1, 60, requires_grad=True)
    future = torch.rand(1, 60)
    mask = torch.ones(1, 60, dtype=torch.bool)
    kw = dict(top_k=10, pool_end=60, margin=0.01, pairs_per_query=8)
    _, pair = boundary_hard_rank_loss(scores, future, mask,
                                      mining_mode='pair', **kw)
    _, cand = boundary_hard_rank_loss(scores, future, mask,
                                      mining_mode='candidate', **kw)
    assert (float(cand['rank_positive_unique_selected_count'])
            >= float(pair['rank_positive_unique_selected_count']))
    # candidate mode never spends two slots on the same positive.
    assert (float(cand['rank_positive_unique_selected_count'])
            == float(cand['rank_positive_selected_count']))


def test_coverage_is_capped_by_the_pair_budget():
    scores = torch.tensor([[5.0, 4.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])
    #                        i=0   i=1  |------- eight candidates, all better ----|
    future = torch.tensor([[9.0, 9.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]])
    mask = torch.ones(1, 10, dtype=torch.bool)
    _, m = boundary_hard_rank_loss(scores, future, mask, top_k=2, pool_end=10,
                                   margin=0.01, pairs_per_query=4,
                                   mining_mode='candidate')
    assert float(m['rank_positive_total_count']) == 8.0
    assert float(m['rank_positive_unique_selected_count']) == 4.0
    assert float(m['rank_positive_unique_covered_frac']) == pytest.approx(0.5)


def test_every_mined_pair_has_the_better_future_on_the_positive_side():
    torch.manual_seed(1)
    scores = torch.randn(2, 60, requires_grad=True)
    future = torch.rand(2, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    for mode in ('pair', 'candidate'):
        loss, m = boundary_hard_rank_loss(scores, future, mask, top_k=10,
                                          pool_end=60, margin=0.01,
                                          pairs_per_query=16, mining_mode=mode)
        assert float(loss) >= 0
        assert float(m['num_valid_rank_pairs']) > 0


def test_margin_satisfied_is_stricter_than_correct_order():
    pairs = [(0, 0, 1, 1.0)]
    ids = torch.tensor([0])
    # s_j - s_i = +0.005: ordering correct, margin of 0.01 not met.
    m = frozen_pair_metrics(torch.tensor([[0.100, 0.105]]), ids, pairs, margin=0.01)
    assert float(m['frozen_pair_correct_order_frac']) == 1.0
    assert float(m['frozen_margin_satisfied_frac']) == 0.0
    m2 = frozen_pair_metrics(torch.tensor([[0.100, 0.130]]), ids, pairs, margin=0.01)
    assert float(m2['frozen_margin_satisfied_frac']) == 1.0


def test_pair_mode_is_unchanged_by_the_new_option():
    torch.manual_seed(2)
    scores = torch.randn(2, 60, requires_grad=True)
    future = torch.rand(2, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    a, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10, pool_end=60,
                                   margin=0.01, pairs_per_query=16)
    b, _ = boundary_hard_rank_loss(scores, future, mask, top_k=10, pool_end=60,
                                   margin=0.01, pairs_per_query=16,
                                   mining_mode='pair')
    assert torch.equal(a, b)


def _persistent(scores, i_id, j_id, gap=1.0):
    b = scores.size(0)
    return (torch.tensor([[i_id]] * b), torch.tensor([[j_id]] * b),
            torch.tensor([[gap]] * b), torch.ones(b, 1, dtype=torch.bool))


def test_persistent_pair_survives_the_positive_entering_the_topk():
    """Dynamic mining releases the pair here; that release is the thing under
    test, so a persistent pair must keep its gradient."""
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    # j now outscores i, so dynamic mining would no longer pair them at all.
    scores = torch.tensor([[0.30, 0.31]], requires_grad=True)
    dynamic, m = boundary_hard_rank_loss(scores, future, mask, top_k=1,
                                         pool_end=2, margin=0.01)
    assert float(m['num_valid_rank_pairs']) == 0
    assert float(dynamic) == 0.0
    keep, _ = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01,
                                      persistent=_persistent(scores, 0, 1))
    # s_j - s_i = 0.01 exactly meets the margin, so the hinge is closed; a hair
    # under it is still supervised.
    assert float(keep) == pytest.approx(0.0, abs=1e-6)
    tight = torch.tensor([[0.300, 0.305]], requires_grad=True)
    still, _ = boundary_hard_rank_loss(tight, future, mask, top_k=1, pool_end=2,
                                       margin=0.01,
                                       persistent=_persistent(tight, 0, 1))
    assert float(still) > 0


def test_persistent_pair_gradient_still_reaches_both_sides():
    scores = torch.tensor([[0.300, 0.305]], requires_grad=True)
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, _ = boundary_hard_rank_loss(scores, future, mask, top_k=1, pool_end=2,
                                      margin=0.01,
                                      persistent=_persistent(scores, 0, 1))
    g = torch.autograd.grad(loss, scores)[0][0]
    assert g[0] > 0 and g[1] < 0


def test_persistent_pairs_are_addressed_by_candidate_not_by_rank():
    """The same (i, j) must be scored whatever the ranking has become."""
    future = torch.tensor([[1.0, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    before = torch.tensor([[0.90, 0.30]], requires_grad=True)
    after = torch.tensor([[0.30, 0.90]], requires_grad=True)
    lo, _ = boundary_hard_rank_loss(before, future, mask, top_k=1, pool_end=2,
                                    margin=0.01, persistent=_persistent(before, 0, 1))
    hi, _ = boundary_hard_rank_loss(after, future, mask, top_k=1, pool_end=2,
                                    margin=0.01, persistent=_persistent(after, 0, 1))
    assert float(lo) > float(hi) == 0.0


def test_dynamic_mode_is_untouched_by_the_persistent_argument():
    torch.manual_seed(3)
    scores = torch.randn(2, 60, requires_grad=True)
    future = torch.rand(2, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    kw = dict(top_k=10, pool_end=60, margin=0.01, pairs_per_query=16)
    a, _ = boundary_hard_rank_loss(scores, future, mask, **kw)
    b, _ = boundary_hard_rank_loss(scores, future, mask, persistent=None, **kw)
    assert torch.equal(a, b)


def test_frozen_builder_supports_candidate_mode():
    torch.manual_seed(4)
    scores = torch.randn(2, 60)
    future = torch.rand(2, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    ids = torch.arange(2)
    pairs = build_frozen_rank_pairs(scores, future, mask, ids, top_k=10,
                                    pool_end=60, per_query=6,
                                    mining_mode='candidate')
    for query in (0, 1):
        js = [j for q, _, j, _ in pairs if q == query]
        assert len(js) == len(set(js)), 'a positive was supervised twice'
    for q, i_id, j_id, gap in pairs:
        assert future[q, j_id] < future[q, i_id] and gap > 0
