"""Oracle selection over a fixed candidate pool: individual, diverse, and set.

The rank scorer pulled candidates with lower individual error into the Top-10
and made the aggregate worse, because the spread that had been cancelling
errors went with them. These check the selectors that separate the two effects,
on cases where the right answer is known by construction.
"""

import pytest
import torch

from models.RelationStage1 import (
    select_good_diverse, select_greedy_set, select_individual_oracle,
    set_utility_metrics,
)


def test_the_decomposition_holds_on_random_sets():
    torch.manual_seed(0)
    y = torch.randn(4, 10, 32)
    q = torch.randn(4, 32)
    i, a, v, residual = set_utility_metrics(y, q)
    assert torch.allclose(i, a + v, atol=1e-5)
    assert float(residual.abs().max()) < 1e-5


def test_individual_oracle_takes_the_lowest_errors():
    d = torch.tensor([[0.5, 0.1, 0.9, 0.3]])
    assert set(select_individual_oracle(d, 2)[0].tolist()) == {1, 3}


def test_a_complementary_candidate_beats_a_second_individually_close_one():
    """Both selections carry the same individual error, and only one of them
    averages near the target: candidates 8.5 and 12 straddle it, 8.5 and 8 sit
    on the same side."""
    q = torch.tensor([[10.0]])
    #                    A=8    B=8.5   C=12       errors 4.0, 2.25, 4.0
    pool = torch.tensor([[[8.0], [8.5], [12.0]]])
    d = ((pool - q.unsqueeze(1)) ** 2).mean(-1)
    ind = select_individual_oracle(d, 2)
    st = select_greedy_set(pool, q, 2)
    assert set(ind[0].tolist()) == {0, 1}
    assert set(st[0].tolist()) == {1, 2}

    gather = lambda idx: torch.gather(
        pool, 1, idx.unsqueeze(-1).expand(-1, -1, pool.size(-1)))
    i_ind, a_ind, _, _ = set_utility_metrics(gather(ind), q)
    i_set, a_set, _, _ = set_utility_metrics(gather(st), q)
    # The joint condition the diagnostic tests for: individual quality no better,
    # aggregate error far lower.
    assert float(i_set) >= float(i_ind)
    assert float(a_set) < float(a_ind)
    assert float(a_ind) == pytest.approx(3.0625, abs=1e-4)
    assert float(a_set) == pytest.approx(0.0625, abs=1e-4)


def test_greedy_can_miss_the_optimum_because_its_first_pick_is_fixed():
    """Starting from the individually best candidate is the specification, and
    it can rule out the optimal pair: 8 and 12 average exactly to 10, but 8 is
    never the first pick. The arm is a lower bound on set-level headroom."""
    q = torch.tensor([[10.0]])
    pool = torch.tensor([[[8.0], [8.5], [12.0]]])
    st = select_greedy_set(pool, q, 2)
    gather = lambda idx: torch.gather(
        pool, 1, idx.unsqueeze(-1).expand(-1, -1, pool.size(-1)))
    _, a_greedy, _, _ = set_utility_metrics(gather(st), q)
    optimal = torch.tensor([[0, 2]])
    _, a_opt, _, _ = set_utility_metrics(gather(optimal), q)
    assert float(a_opt) == pytest.approx(0.0, abs=1e-6)
    assert float(a_greedy) > float(a_opt)


def test_greedy_set_starts_from_the_individually_best_candidate():
    torch.manual_seed(0)
    pool = torch.randn(3, 20, 8)
    q = torch.randn(3, 8)
    d = ((pool - q.unsqueeze(1)) ** 2).mean(-1)
    first = select_greedy_set(pool, q, 5)[:, 0]
    assert torch.equal(first, d.argmin(dim=-1))


def test_greedy_set_never_beats_itself_by_going_backwards():
    """Each step is chosen to minimise the aggregate error, so the running
    aggregate cannot get worse than starting over with fewer members."""
    torch.manual_seed(1)
    pool = torch.randn(2, 40, 16)
    q = torch.randn(2, 16)
    gather = lambda idx: torch.gather(
        pool, 1, idx.unsqueeze(-1).expand(-1, -1, pool.size(-1)))
    errs = []
    for k in (1, 2, 4, 8):
        _, a, _, _ = set_utility_metrics(gather(select_greedy_set(pool, q, k)), q)
        errs.append(float(a.mean()))
    assert errs[0] >= errs[-1]


def test_good_diverse_stays_inside_the_quality_shortlist():
    torch.manual_seed(0)
    pool = torch.randn(2, 50, 8)
    q = torch.randn(2, 8)
    d = ((pool - q.unsqueeze(1)) ** 2).mean(-1)
    good = set(d.topk(10, dim=-1, largest=False).indices[0].tolist())
    picked = select_good_diverse(pool, d, k=5, good_n=10)[0].tolist()
    assert set(picked) <= good
    assert len(set(picked)) == 5, 'a candidate was selected twice'


def test_good_diverse_prefers_spread_over_the_next_best_individual():
    """With quality fixed by the shortlist, it must pick the distant one."""
    q = torch.tensor([[0.0]])
    #                   near-duplicates            far
    pool = torch.tensor([[[1.0], [1.01], [-1.02]]])
    d = ((pool - q.unsqueeze(1)) ** 2).mean(-1)
    picked = select_good_diverse(pool, d, k=2, good_n=3)[0].tolist()
    assert picked[0] == 0            # individually best starts the set
    assert picked[1] == 2            # then the one furthest from it


def test_selectors_return_distinct_indices():
    torch.manual_seed(2)
    pool = torch.randn(3, 60, 12)
    q = torch.randn(3, 12)
    d = ((pool - q.unsqueeze(1)) ** 2).mean(-1)
    for idx in (select_individual_oracle(d, 10),
                select_greedy_set(pool, q, 10),
                select_good_diverse(pool, d, 10, good_n=30)):
        for row in idx:
            assert len(set(row.tolist())) == 10


def test_imitation_loss_falls_when_the_target_is_ranked_highest():
    from models.RelationStage1 import oracle_imitation_loss
    pool = torch.tensor([[0, 1, 2, 3]])
    target = torch.tensor([[0, 1]])
    wrong = torch.tensor([[0.1, 0.1, 5.0, 5.0]])   # mass on non-targets
    right = torch.tensor([[5.0, 5.0, 0.1, 0.1]])
    lo, m = oracle_imitation_loss(right, pool, target, tau=1.0)
    hi, _ = oracle_imitation_loss(wrong, pool, target, tau=1.0)
    assert float(lo) < float(hi)
    assert float(m['teacher_set_recall_at_k']) == 1.0


def test_imitation_only_ranks_inside_the_shared_pool():
    """A candidate outside the pool must not affect the loss, or the two arms
    would differ by their pools rather than by their targets."""
    from models.RelationStage1 import oracle_imitation_loss
    pool = torch.tensor([[0, 1]])
    target = torch.tensor([[0]])
    a = torch.tensor([[2.0, 0.0, 9.0]])
    b = torch.tensor([[2.0, 0.0, -9.0]])          # differs only outside the pool
    la, _ = oracle_imitation_loss(a, pool, target, tau=1.0)
    lb, _ = oracle_imitation_loss(b, pool, target, tau=1.0)
    assert float(la) == pytest.approx(float(lb), abs=1e-7)


def test_stability_probe_separates_a_unique_set_from_many_equivalent_ones():
    from models.RelationStage1 import greedy_set_stability
    # One clearly best pair: restarting cannot find an equally good alternative.
    q = torch.tensor([[0.0]])
    sharp = torch.tensor([[[0.01], [-0.01], [5.0], [-5.0], [9.0]]])
    s = greedy_set_stability(sharp, q, k=2, restarts=3)
    # Many interchangeable candidates: restarts land on different, equally good sets.
    flat = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0], [1.0]]])
    f = greedy_set_stability(flat, q, k=2, restarts=3)
    assert float(f['greedy_restart_rel_gap_mean']) <= float(s['greedy_restart_rel_gap_mean'])
    assert float(f['greedy_restart_overlap_mean']) < 1.0
