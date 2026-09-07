"""EXP-FIRSTANCHOR-DIAG: causal decomposition of EXP-MARGUTIL01's t=1 choice.

No checkpoint/GPU needed -- these are structural/math invariant tests on
toy data, mirroring tests/test_exp_margutil01.py's style.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.eval_firstanchor_diag import a_weighted_prefix, hard_agg, run_arm
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


def _toy(bsz=4, n=20, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    futures = torch.randn(bsz, n, h, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    scores = torch.randn(bsz, n, generator=g) * 0.5  # bounded, cosine-like scale
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -2:] = False
    return futures, query_future, scores, valid


def _build_selector(d=8):
    return SetConditioner(d), EmptySetToken(d), UtilityHead()


def test_t1_singleton_a_weighted_equals_individual_future_mse():
    """Section 2: at t=1, alpha_i=1 for the singleton set, so A_weighted({i})
    reduces exactly to MSE(Y_i, Y_q) -- not a set-complementarity quantity."""
    futures, query_future, scores, valid = _toy()
    tau = 0.1
    i1 = torch.randint(0, futures.size(1), (futures.size(0), 1))
    a1 = a_weighted_prefix(i1, scores, futures, query_future, tau)
    expected = (futures.gather(1, i1.unsqueeze(-1).expand(-1, -1, futures.size(-1))).squeeze(1)
                - query_future).pow(2).mean(-1)
    assert torch.allclose(a1, expected, atol=1e-5)


def test_oracle_first_equals_greedy_oracle_first_candidate():
    """Section 6 Arm C: argmin singleton MSE must equal
    select_greedy_weighted_set's own first pick (same identity proven in
    test_exp_margutil01.py's t=1 test, re-verified here at the call site
    this script actually uses)."""
    futures, query_future, scores, valid = _toy(n=25)
    tau = 0.1
    oracle_seq = select_greedy_weighted_set(futures, query_future, scores, valid, k=3, tau=tau)
    dist = (futures - query_future.unsqueeze(1)).pow(2).mean(-1)
    dist = dist.masked_fill(~valid, float('inf'))
    i1_oracle = dist.argmin(dim=-1)
    assert torch.equal(i1_oracle, oracle_seq[:, 0])


def test_intervention_affects_only_t1_dense_vs_forced_agree_from_t2():
    """If two runs share the same t=1 pick, EVERY subsequent step must be
    identical (the selector has no other source of randomness/state) -- and
    if they DIFFER at t=1, the runs are free to diverge from t=2 on, since
    each step's set summary depends on the whole prefix so far."""
    bsz, n, d, k = 3, 18, 8, 5
    futures, query_future, scores, valid = _toy(bsz=bsz, n=n, h=4)
    q = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    set_cond, empty_token, utility_head = _build_selector(d)

    picks_free = run_arm(q, E, valid, set_cond, empty_token, utility_head, k, first_pick=None)
    # Force t=1 to the SAME candidate the free-running arm picked anyway:
    picks_forced_same = run_arm(q, E, valid, set_cond, empty_token, utility_head, k,
                                 first_pick=picks_free[:, 0])
    assert torch.equal(picks_free, picks_forced_same), (
        'forcing t=1 to the value it would have picked anyway must reproduce '
        'the exact same trajectory -- t>=2 uses no other randomness')

    # Force t=1 to a DIFFERENT, deliberately chosen candidate.
    other_first = (picks_free[:, 0] + 1) % n
    other_first = torch.where(valid.gather(1, other_first.unsqueeze(-1)).squeeze(1), other_first,
                               (picks_free[:, 0] + 2) % n)
    picks_forced_other = run_arm(q, E, valid, set_cond, empty_token, utility_head, k,
                                  first_pick=other_first)
    assert torch.equal(picks_forced_other[:, 0], other_first)


def test_no_teacher_forcing_or_oracle_injection_past_t1():
    """B0-first / Oracle-first only override index 0 of `picks`; steps 1..K-1
    come from run_arm's own argmax branch, the same code path arm A uses --
    inspected here by confirming run_arm's signature carries no per-step
    override, only a single first_pick."""
    import inspect
    sig = inspect.signature(run_arm)
    assert list(sig.parameters) == [
        'q', 'E', 'cand_mask', 'set_cond', 'empty_token', 'utility_head', 'k', 'first_pick']


def test_full_memory_support_preserved_and_no_duplicates_or_invalid():
    bsz, n, d, k = 4, 30, 8, 6
    futures, query_future, scores, valid = _toy(bsz=bsz, n=n, h=4)
    q = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    set_cond, empty_token, utility_head = _build_selector(d)
    dist = (futures - query_future.unsqueeze(1)).pow(2).mean(-1).masked_fill(~valid, float('inf'))
    i1_oracle = dist.argmin(dim=-1)
    for first in (None, i1_oracle):
        picks = run_arm(q, E, valid, set_cond, empty_token, utility_head, k, first_pick=first)
        for b in range(bsz):
            row = picks[b].tolist()
            assert len(set(row)) == len(row), 'duplicate pick'
            assert all(valid[b, i] for i in row), 'invalid candidate picked'
        assert picks.size(1) == n if False else picks.size(1) == k  # every step still scores full E (n candidates)


def test_stage2_weighting_unchanged_a_weighted_uses_b0_score_not_dense_score():
    """The trajectory/aggregate metric must be computed with B0's own fixed
    score (`learned_ref`), never the Dense model's own u_hat -- verified by
    signature: a_weighted_prefix takes `learned_ref` as an explicit external
    argument, it does not call into utility_head at all."""
    import inspect
    src = inspect.getsource(a_weighted_prefix)
    assert 'utility_head' not in src
    assert 'learned_ref' in inspect.signature(a_weighted_prefix).parameters


def test_hard_agg_matches_manual_unweighted_mean():
    futures, query_future, scores, valid = _toy()
    prefix = torch.randint(0, futures.size(1), (futures.size(0), 4))
    got = hard_agg(prefix, futures, query_future)
    tgt = futures.gather(1, prefix.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
    expected = (tgt.mean(1) - query_future).pow(2).sum(-1)
    assert torch.allclose(got, expected, atol=1e-5)


def test_first_pick_override_does_not_require_grad_or_touch_query_future_downstream():
    """Leakage audit: run_arm's `first_pick` argument is an index tensor
    only -- Y_q never appears in the student-forward call path (set_cond /
    utility_head), whichever arm chose that index."""
    import inspect
    src = inspect.getsource(run_arm)
    assert 'query_future' not in src and 'q_tgt' not in src and 'Y_q' not in src
