"""Selection-rule intervention: does only the selection change, and is it the one asked for?

The experiment is causal only if the arms differ in exactly one thing. Two
failure modes would silently destroy that and produce a publishable-looking
number anyway: a selector that returns something other than what it claims, and
a selection that never reaches the forward pass -- the second is precisely the
wiring bug that already invalidated one Stage-2 sweep, so it is pinned here
against the real retrieval call rather than a mock.

The cases with a known answer are built so that the individually-best ten and
the best ten *together* are different sets; on a pool where they coincide the
whole diagnostic would pass while measuring nothing.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.oracle_intervention import (  # noqa: E402
    ALL_ARMS, DEFAULT_ARMS, build_common_support, overlap,
    relation_equals_target, select_greedy_weighted_set, select_within_support,
    to_global, uniform_metrics, weighted_metrics,
)
from utils.retrieval_ops import retrieve_relation_future  # noqa: E402


def _pool(bsz=2, pool=12, dim=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(bsz, pool, dim)


# --------------------------------------------------------------------------
# 1. each arm returns the indices it claims to
# --------------------------------------------------------------------------

def test_individual_arm_returns_the_individually_closest():
    # Candidate j sits at distance j from the query, so the answer is 0,1,2.
    q = torch.zeros(1, 3)
    futures = torch.stack([torch.full((3,), float(j)) for j in range(5)]).unsqueeze(0)
    valid = torch.ones(1, 5, dtype=torch.bool)
    pool_idx = torch.arange(5).unsqueeze(0)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    got = select_within_support('R1', pool_idx, valid, torch.zeros(1, 5),
                                futures, rel, q, q_rel, k=3)
    assert got[0].tolist() == [0, 1, 2]


def test_set_arm_trades_individual_quality_for_a_better_aggregate():
    """The individually best pair averages to 0.5; a mixed pair averages nearer 0.

    An individual selector takes both 0.5s -- each beats 1.0 on its own -- and
    the mean stays at 0.5. Greedy opens on the same individually-best candidate
    and then has to accept a *worse* one, -1.0, because that is what cancels the
    error already banked. The set it lands on is therefore neither the
    individually best pair nor the exact cancelling pair, which is precisely the
    trade this arm exists to expose.
    """
    q = torch.zeros(1, 2)
    futures = torch.tensor([[[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [0.5, 0.5]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    pool_idx = torch.arange(4).unsqueeze(0)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)

    ind = select_within_support('R1', pool_idx, valid, torch.zeros(1, 4),
                                futures, rel, q, q_rel, k=2)
    st = select_within_support('R2-target', pool_idx, valid, torch.zeros(1, 4),
                               futures, rel, q, q_rel, k=2)
    gather = lambda idx: torch.gather(
        futures, 1, idx.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
    _, a_ind, _, _ = uniform_metrics(gather(ind), q)
    i_set, a_set, _, _ = uniform_metrics(gather(st), q)
    i_ind, _, _, _ = uniform_metrics(gather(ind), q)

    assert set(st[0].tolist()) != set(ind[0].tolist())
    assert 1 in st[0].tolist(), 'greedy never took the cancelling candidate'
    assert float(a_set) < float(a_ind)      # better together
    assert float(i_set) > float(i_ind)      # worse individually


def test_r0_arm_follows_the_learned_score_not_the_future():
    q = torch.zeros(1, 2)
    futures = torch.tensor([[[9.0, 9.0], [0.0, 0.0], [8.0, 8.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    pool_idx = torch.arange(3).unsqueeze(0)
    learned = torch.tensor([[5.0, -5.0, 4.0]])     # ranks the far candidates first
    rel = torch.cat([futures, futures], dim=-1)
    got = select_within_support('R0', pool_idx, valid, learned, futures, rel,
                                q, torch.cat([q, q], dim=-1), k=2)
    assert got[0].tolist() == [0, 2]


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match='unknown arm'):
        select_within_support('R9', torch.zeros(1, 2, dtype=torch.long),
                              torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2),
                              _pool(1, 2), _pool(1, 2, 8), torch.zeros(1, 4),
                              torch.zeros(1, 8), k=1)


# --------------------------------------------------------------------------
# 2. individual and set really do differ
# --------------------------------------------------------------------------

def test_individual_and_set_select_different_candidates():
    q = torch.zeros(2, 6)
    futures = _pool(2, 40, 6, seed=3)
    valid = torch.ones(2, 40, dtype=torch.bool)
    pool_idx = torch.arange(40).expand(2, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    ind = select_within_support('R1', pool_idx, valid, torch.zeros(2, 40),
                                futures, rel, q, q_rel, k=10)
    st = select_within_support('R2-target', pool_idx, valid, torch.zeros(2, 40),
                               futures, rel, q, q_rel, k=10)
    assert float(overlap(ind, st).mean()) < 1.0, 'arms are indistinguishable'


def test_k1_set_equals_individual():
    """With one member the aggregate is that member, so the arms must agree."""
    q = torch.zeros(2, 5)
    futures = _pool(2, 20, 5, seed=7)
    valid = torch.ones(2, 20, dtype=torch.bool)
    pool_idx = torch.arange(20).expand(2, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    ind = select_within_support('R1', pool_idx, valid, torch.zeros(2, 20),
                                futures, rel, q, q_rel, k=1)
    st = select_within_support('R2-target', pool_idx, valid, torch.zeros(2, 20),
                               futures, rel, q, q_rel, k=1)
    assert ind.tolist() == st.tolist()


# --------------------------------------------------------------------------
# 3. the selection actually reaches retrieval  (wiring regression)
# --------------------------------------------------------------------------

def test_forced_indices_are_the_ones_retrieval_uses():
    """The whole point of the intervention: forced_idx must win over Top-K.

    A previous wiring bug let a configured scorer be trained and then quietly
    ignored at selection time. This asserts against the real retrieval call:
    the returned indices, and the values gathered from memory, must both follow
    the forced selection rather than the scores.
    """
    torch.manual_seed(0)
    z_q, z_mem = torch.randn(3, 8), torch.randn(20, 8)
    values = torch.randn(20, 5)
    valid = torch.ones(3, 20, dtype=torch.bool)
    forced = torch.tensor([[7, 2, 11], [0, 19, 5], [3, 3, 4]])

    _, _, top_idx, _, dbg = retrieve_relation_future(
        z_q, z_mem, values, valid, top_k=3, tau_topk=0.1, forced_idx=forced)

    assert top_idx.tolist() == forced.tolist()
    assert torch.allclose(dbg['v_top'], values[forced])


def test_forced_selection_changes_the_retrieved_value():
    torch.manual_seed(1)
    z_q, z_mem = torch.randn(2, 8), torch.randn(30, 8)
    values = torch.randn(30, 6)
    valid = torch.ones(2, 30, dtype=torch.bool)

    natural, _, nat_idx, _, _ = retrieve_relation_future(
        z_q, z_mem, values, valid, top_k=5, tau_topk=0.1)
    # Anything the model would not have chosen.
    alt = torch.stack([
        torch.tensor([i for i in range(30) if i not in set(nat_idx[b].tolist())][:5])
        for b in range(2)])
    forced, _, forced_idx, _, _ = retrieve_relation_future(
        z_q, z_mem, values, valid, top_k=5, tau_topk=0.1, forced_idx=alt)

    assert forced_idx.tolist() == alt.tolist()
    assert not torch.allclose(natural, forced)


def test_forced_selection_leaves_scores_and_weighting_rule_untouched():
    """Only selection may change: the score matrix must be identical."""
    torch.manual_seed(2)
    z_q, z_mem = torch.randn(2, 8), torch.randn(15, 8)
    values = torch.randn(15, 4)
    valid = torch.ones(2, 15, dtype=torch.bool)
    forced = torch.tensor([[1, 2], [3, 4]])

    _, _, _, _, a = retrieve_relation_future(z_q, z_mem, values, valid, 2, 0.1)
    _, alpha, idx, top_scores, b = retrieve_relation_future(
        z_q, z_mem, values, valid, 2, 0.1, forced_idx=forced)

    assert torch.allclose(a['scores'], b['scores'])
    # Weights are still softmax(score/tau) over whatever was selected.
    assert torch.allclose(top_scores, a['scores'].gather(-1, forced))
    expected = torch.softmax(top_scores / 0.1, dim=-1)
    assert torch.allclose(alpha, expected, atol=1e-6)


def test_forced_idx_batch_mismatch_is_rejected():
    z_q, z_mem = torch.randn(2, 4), torch.randn(10, 4)
    with pytest.raises(ValueError, match='forced_idx batch'):
        retrieve_relation_future(
            z_q, z_mem, torch.randn(10, 3), torch.ones(2, 10, dtype=torch.bool),
            top_k=2, tau_topk=0.1, forced_idx=torch.tensor([[1, 2]]))


# --------------------------------------------------------------------------
# 4. the model does not move between arms
# --------------------------------------------------------------------------

def test_set_forced_selection_holds_no_parameters_and_clears():
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(3))
        set_forced_selection = None

    from models.RelationStage2 import Model
    model = Tiny()
    model.set_forced_selection = Model.set_forced_selection.__get__(model)
    model._forced_selection_for = Model._forced_selection_for.__get__(model)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.set_forced_selection({(0, 0): torch.tensor([[1, 2]])})
    assert model._forced_selection_for(0, 0).tolist() == [[1, 2]]
    assert model._forced_selection_for(1, 0) is None
    model.set_forced_selection(None)
    assert model._forced_selection_for(0, 0) is None
    for key, val in model.state_dict().items():
        assert torch.equal(val, before[key]), 'forcing a selection changed a parameter'


# --------------------------------------------------------------------------
# 5. the decomposition identity
# --------------------------------------------------------------------------

def test_uniform_decomposition_residual_is_negligible():
    futures, q = _pool(4, 10, 16, seed=11), torch.randn(4, 16)
    _, _, _, res = uniform_metrics(futures, q)
    assert float(res.abs().max()) < 1e-4


def test_weighted_decomposition_residual_is_negligible():
    """I_w = A_w + V_w must hold for the weights Stage-2 actually applies."""
    torch.manual_seed(5)
    futures, q = _pool(4, 10, 16, seed=13), torch.randn(4, 16)
    alpha = torch.softmax(torch.randn(4, 10) / 0.1, dim=-1)
    _, _, _, res = weighted_metrics(futures, q, alpha)
    assert float(res.abs().max()) < 1e-4


def test_weighted_and_uniform_agree_when_weights_are_uniform():
    futures, q = _pool(3, 8, 12, seed=17), torch.randn(3, 12)
    alpha = torch.full((3, 8), 1.0 / 8.0)
    i_u, a_u, v_u, _ = uniform_metrics(futures, q)
    i_w, a_w, v_w, _ = weighted_metrics(futures, q, alpha)
    for lhs, rhs in ((i_u, i_w), (a_u, a_w), (v_u, v_w)):
        assert torch.allclose(lhs, rhs, atol=1e-5)


# --------------------------------------------------------------------------
# 6. every arm selects inside one shared support
# --------------------------------------------------------------------------

def test_support_is_identical_across_arms_and_respects_the_mask():
    torch.manual_seed(4)
    cosine = torch.randn(3, 50)
    valid = torch.ones(3, 50, dtype=torch.bool)
    valid[:, 40:] = False
    pool_idx, pool_valid = build_common_support(cosine, valid, pool_m=20, k=10)

    assert pool_idx.shape == (3, 20)
    assert bool(pool_valid.all()), 'masked candidates entered the support'
    assert (pool_idx < 40).all()

    futures = torch.randn(3, 20, 6)
    q = torch.randn(3, 6)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    for arm in ALL_ARMS:
        local = select_within_support(arm, pool_idx, pool_valid, torch.randn(3, 20),
                                      futures, rel, q, q_rel, k=10)
        glob = to_global(pool_idx, local)
        assert local.max() < 20
        # Every arm's picks are drawn from the one support.
        for b in range(3):
            assert set(glob[b].tolist()) <= set(pool_idx[b].tolist())


def test_invalid_support_slots_are_never_selected():
    """Greedy has no -inf, so invalid rows must be neutralised in value space."""
    q = torch.zeros(1, 3)
    futures = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.1, 0.1, 0.1]]])
    pool_valid = torch.tensor([[True, False, True]])
    pool_idx = torch.arange(3).unsqueeze(0)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    for arm in ('R1', 'R2-target', 'R2-relation', 'R3'):
        got = select_within_support(arm, pool_idx, pool_valid, torch.zeros(1, 3),
                                    futures, rel, q, q_rel, k=2)
        assert 1 not in got[0].tolist(), f'{arm} selected an invalid candidate'


def test_build_common_support_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='must agree'):
        build_common_support(torch.randn(2, 10), torch.ones(3, 10, dtype=torch.bool),
                             pool_m=5, k=2)


# --------------------------------------------------------------------------
# 7. the relation-space degeneracy is detected, not assumed away
# --------------------------------------------------------------------------

def test_self_only_graph_is_reported_as_relation_equals_target():
    assert relation_equals_target([[0], [1], [2]]) is True
    assert relation_equals_target([[0, 2], [1], [2]]) is False


def test_r2_relation_matches_r2_target_when_source_is_self():
    """Under a self-only graph the two set arms are one experiment, not two."""
    q = torch.randn(2, 5)
    futures = _pool(2, 25, 5, seed=21)
    valid = torch.ones(2, 25, dtype=torch.bool)
    pool_idx = torch.arange(25).expand(2, -1)
    rel = torch.cat([futures, futures], dim=-1)      # source == target
    q_rel = torch.cat([q, q], dim=-1)
    a = select_within_support('R2-target', pool_idx, valid, torch.zeros(2, 25),
                              futures, rel, q, q_rel, k=8)
    b = select_within_support('R2-relation', pool_idx, valid, torch.zeros(2, 25),
                              futures, rel, q, q_rel, k=8)
    assert a.tolist() == b.tolist()


def test_r2_relation_differs_from_r2_target_with_a_real_source():
    torch.manual_seed(23)
    q_t, q_s = torch.randn(2, 5), torch.randn(2, 5)
    tgt, src = _pool(2, 30, 5, seed=29), _pool(2, 30, 5, seed=31)
    valid = torch.ones(2, 30, dtype=torch.bool)
    pool_idx = torch.arange(30).expand(2, -1)
    rel = torch.cat([tgt, src], dim=-1)
    q_rel = torch.cat([q_t, q_s], dim=-1)
    a = select_within_support('R2-target', pool_idx, valid, torch.zeros(2, 30),
                              tgt, rel, q_t, q_rel, k=10)
    b = select_within_support('R2-relation', pool_idx, valid, torch.zeros(2, 30),
                              tgt, rel, q_t, q_rel, k=10)
    assert a.tolist() != b.tolist()


# --------------------------------------------------------------------------
# 8. overlap reporting
# --------------------------------------------------------------------------

def test_overlap_is_one_for_identical_and_zero_for_disjoint():
    a = torch.tensor([[1, 2, 3]])
    assert float(overlap(a, a).mean()) == pytest.approx(1.0)
    assert float(overlap(a, torch.tensor([[4, 5, 6]])).mean()) == pytest.approx(0.0)
    assert float(overlap(a, torch.tensor([[1, 5, 6]])).mean()) == pytest.approx(1 / 3)


def test_overlap_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='equal shapes'):
        overlap(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))


# --------------------------------------------------------------------------
# 9. weighted set oracle -- it must optimise the aggregate Stage-2 forms
# --------------------------------------------------------------------------

def test_weighted_greedy_renormalises_over_the_whole_selected_set():
    """Adding a candidate must dilute the weights already assigned.

    This is the property that separates R2-W from "uniform greedy with weights
    bolted on". If the softmax were computed once and new terms appended, the
    weight of the first pick would not move when a second is added -- so the
    check is that it does move, and that what is applied is a softmax over the
    selected pair rather than over the pool.
    """
    scores = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    futures = torch.tensor([[[1.0], [0.0], [-1.0], [3.0]]])
    q = torch.zeros(1, 1)
    tau = 0.5

    picked = select_greedy_weighted_set(futures, q, scores, valid, k=2, tau=tau)
    sel = picked[0].tolist()

    pair = torch.softmax(scores[0, sel] / tau, dim=-1)
    single = torch.softmax(scores[0, sel[:1]] / tau, dim=-1)
    assert float(single[0]) == pytest.approx(1.0)          # alone it carries all mass
    assert float(pair[0]) < 1.0                            # ... and is diluted by the second
    # The reported aggregate must be the one the pair-softmax produces.
    agg = (pair.unsqueeze(-1) * futures[0, sel]).sum(0)
    _, a_w, _, _ = weighted_metrics(futures[:, sel], q, pair.unsqueeze(0))
    assert float(a_w) == pytest.approx(float(agg.pow(2).mean()), abs=1e-6)


def test_weighted_set_beats_uniform_set_on_the_weighted_objective():
    """Each arm should win on the objective it optimises."""
    torch.manual_seed(41)
    pool, k, tau = 24, 6, 0.1
    futures = _pool(4, pool, 8, seed=41)
    q = torch.randn(4, 8)
    scores = torch.randn(4, pool)
    valid = torch.ones(4, pool, dtype=torch.bool)
    pool_idx = torch.arange(pool).expand(4, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)

    def weighted_a(local):
        chosen = torch.gather(
            futures, 1, local.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
        alpha = torch.softmax(scores.gather(1, local) / tau, dim=-1)
        return weighted_metrics(chosen, q, alpha)[1].mean()

    u = select_within_support('R2-U', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=tau)
    w = select_within_support('R2-W', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=tau)
    assert float(weighted_a(w)) < float(weighted_a(u))


def test_uniform_set_beats_weighted_set_on_the_uniform_objective():
    torch.manual_seed(43)
    pool, k, tau = 24, 6, 0.1
    futures = _pool(4, pool, 8, seed=43)
    q = torch.randn(4, 8)
    scores = torch.randn(4, pool)
    valid = torch.ones(4, pool, dtype=torch.bool)
    pool_idx = torch.arange(pool).expand(4, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)

    def uniform_a(local):
        chosen = torch.gather(
            futures, 1, local.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
        return uniform_metrics(chosen, q)[1].mean()

    u = select_within_support('R2-U', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=tau)
    w = select_within_support('R2-W', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=tau)
    assert float(uniform_a(u)) < float(uniform_a(w))


def test_weighted_set_matches_uniform_set_when_all_scores_are_equal():
    """Equal scores make the softmax uniform, so the two objectives coincide."""
    torch.manual_seed(47)
    pool, k = 20, 5
    futures = _pool(3, pool, 6, seed=47)
    q = torch.randn(3, 6)
    scores = torch.zeros(3, pool)
    valid = torch.ones(3, pool, dtype=torch.bool)
    pool_idx = torch.arange(pool).expand(3, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    u = select_within_support('R2-U', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=0.1)
    w = select_within_support('R2-W', pool_idx, valid, scores, futures, rel,
                              q, q_rel, k=k, tau=0.1)
    assert u.tolist() == w.tolist()


def test_weighted_set_never_selects_an_invalid_candidate():
    torch.manual_seed(53)
    futures = torch.tensor([[[0.0], [50.0], [0.2], [0.3]]])
    q = torch.zeros(1, 1)
    scores = torch.tensor([[0.0, 9.0, 0.1, 0.2]])       # invalid one scores highest
    valid = torch.tensor([[True, False, True, True]])
    got = select_greedy_weighted_set(futures, q, scores, valid, k=2, tau=0.1)
    assert 1 not in got[0].tolist()


def test_default_arms_exclude_the_degenerate_relation_arm():
    assert 'R2-relation' not in DEFAULT_ARMS
    assert set(DEFAULT_ARMS) == {'R0', 'R1', 'R2-U', 'R2-W', 'R3'}
    assert all(a in ALL_ARMS for a in DEFAULT_ARMS)


def test_r2_u_is_an_alias_of_r2_target():
    torch.manual_seed(59)
    futures = _pool(2, 20, 5, seed=59)
    q = torch.randn(2, 5)
    valid = torch.ones(2, 20, dtype=torch.bool)
    pool_idx = torch.arange(20).expand(2, -1)
    rel = torch.cat([futures, futures], dim=-1)
    q_rel = torch.cat([q, q], dim=-1)
    a = select_within_support('R2-U', pool_idx, valid, torch.zeros(2, 20),
                              futures, rel, q, q_rel, k=7)
    b = select_within_support('R2-target', pool_idx, valid, torch.zeros(2, 20),
                              futures, rel, q, q_rel, k=7)
    assert a.tolist() == b.tolist()


def test_weighted_greedy_closed_form_matches_the_explicit_softmax():
    """The O(P*D) recurrence must agree with recomputing the softmax outright.

    The closed form is what lets this arm run over a full memory bank, so it is
    checked against a literal per-candidate softmax on a small pool where the
    naive version is affordable.
    """
    torch.manual_seed(101)
    bsz, pool, dim, k, tau = 3, 14, 5, 4, 0.1
    y = torch.randn(bsz, pool, dim)
    q = torch.randn(bsz, dim)
    scores = torch.randn(bsz, pool)
    valid = torch.ones(bsz, pool, dtype=torch.bool)

    def naive():
        taken = [set() for _ in range(bsz)]
        picks = [[] for _ in range(bsz)]
        for _ in range(k):
            for b in range(bsz):
                best, best_err = None, float('inf')
                for c in range(pool):
                    if c in taken[b]:
                        continue
                    sel = sorted(taken[b] | {c})
                    w = torch.softmax(scores[b, sel] / tau, dim=-1)
                    agg = (w.unsqueeze(-1) * y[b, sel]).sum(0)
                    err = float((agg - q[b]).pow(2).mean())
                    if err < best_err - 1e-12:
                        best, best_err = c, err
                taken[b].add(best)
                picks[b].append(best)
        return picks

    fast = select_greedy_weighted_set(y, q, scores, valid, k, tau).tolist()
    assert fast == naive()


def test_weighted_greedy_scales_to_a_large_pool():
    """Full-memory arms need this to run at all; O(P^2) would not fit."""
    torch.manual_seed(103)
    y = torch.randn(2, 8000, 16)
    q = torch.randn(2, 16)
    scores = torch.randn(2, 8000)
    valid = torch.ones(2, 8000, dtype=torch.bool)
    picked = select_greedy_weighted_set(y, q, scores, valid, k=10, tau=0.1)
    assert picked.shape == (2, 10)
    assert all(len(set(row)) == 10 for row in picked.tolist())


def test_weighted_greedy_is_stable_at_a_small_temperature():
    """exp(s/tau) must not overflow when tau is small and scores are spread."""
    y = torch.randn(2, 40, 8)
    q = torch.randn(2, 8)
    scores = torch.randn(2, 40) * 5.0
    valid = torch.ones(2, 40, dtype=torch.bool)
    picked = select_greedy_weighted_set(y, q, scores, valid, k=5, tau=0.01)
    assert torch.isfinite(picked.float()).all()
    assert all(len(set(row)) == 5 for row in picked.tolist())


def test_uniform_metrics_matches_the_stage1_helper():
    """The double-precision copy must agree with the shared float32 helper."""
    from models.RelationStage1 import set_utility_metrics
    torch.manual_seed(211)
    y, q = torch.randn(4, 10, 12), torch.randn(4, 12)
    a = uniform_metrics(y, q)
    b = set_utility_metrics(y, q)
    for mine, theirs in zip(a[:3], b[:3]):
        assert torch.allclose(mine.float(), theirs, atol=1e-5)


def test_decomposition_residual_survives_a_large_scale_selection():
    """A full-memory arm can select candidates whose error dwarfs the aggregate.

    Weather full memory tripped the correctness gate at 1.8e-4 purely on float32
    rounding, so the identity is checked here at a magnitude where that happens.
    """
    torch.manual_seed(213)
    y = torch.randn(4, 10, 720) * 30.0
    q = torch.randn(4, 720)
    alpha = torch.softmax(torch.randn(4, 10) / 0.1, dim=-1)
    assert float(uniform_metrics(y, q)[3].abs().max()) < 1e-6
    assert float(weighted_metrics(y, q, alpha)[3].abs().max()) < 1e-6
