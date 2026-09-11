"""Mandatory sanity checks for TRACK-A-FACTORIAL-E2E01 (spec section 26).

Every check runs on CPU with tiny synthetic tensors -- none of them needs a
checkpoint or a GPU, so the whole file is a pre-flight gate for the 32-run
campaign.
"""
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts import train_factorial_e2e01 as T
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_ops import retrieve_relation_future

D, N, B, K, H = 8, 24, 3, 4, 5
TAU = 0.1


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    z_q = torch.randn(B, D, generator=g)
    E = torch.randn(N, D, generator=g)
    futures = torch.randn(B, N, H, generator=g)
    q_future = torch.randn(B, H, generator=g)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, -3:] = False                       # some invalid candidates
    # Host scores are COSINE similarities in production (the Stage-2 host uses
    # output='cosine'), so they live in [-1, 1]. Using that range here matters:
    # `candidate_weights` row-max-shifts and exponentiates at tau=0.1, so the
    # smallest possible weight is exp(-2/0.1) = 2.06e-9, comfortably above
    # `dense_utility`'s 1e-12 denominator clamp. Unbounded scores would
    # underflow and make the Oracle target meaningless -- see
    # `test_host_weights_never_underflow_dense_utility_clamp`.
    host_scores = torch.rand(B, N, generator=g) * 2.0 - 1.0
    host_scores = host_scores.masked_fill(~cand_mask, torch.finfo(host_scores.dtype).min / 4)
    w_host = candidate_weights(host_scores, cand_mask, TAU)
    sc = SetConditioner(D)
    return z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc


# ---------------------------------------------------------------- Oracle ---
def test_individual_oracle_is_argmin_of_individual_future_mse():
    _, _, futures, q_future, _, _, _, _ = _fixture()
    u = T.individual_utility(futures, q_future)
    mse = ((futures - q_future.unsqueeze(1)) ** 2).mean(-1)
    assert torch.allclose(u, -mse)
    assert torch.equal(u.argmax(dim=-1), mse.argmin(dim=-1))


def test_individual_oracle_is_prefix_invariant():
    """Spec S6: the Individual Oracle ignores the already-selected set."""
    _, _, futures, q_future, _, _, _, _ = _fixture()
    a = T.individual_utility(futures, q_future)
    b = T.individual_utility(futures, q_future)
    assert torch.equal(a, b)
    assert len(inspect.signature(T.individual_utility).parameters) == 2


def test_greedy_set_oracle_matches_aggregate_definition():
    """u_i = -MSE(Aggregate(S u {i}), y_q) with the host weights."""
    _, _, futures, q_future, cand_mask, _, w_host, _ = _fixture()
    prefix = torch.tensor([[0, 1], [2, 3], [4, 5]])
    u = T.greedy_set_utility(prefix, w_host, futures, q_future, chunk_size=None)
    for b in range(B):
        for i in (6, 7, 8):
            idx = prefix[b].tolist() + [i]
            w = w_host[b, idx]
            agg = (w.unsqueeze(-1) * futures[b, idx]).sum(0) / w.sum()
            expect = -((agg - q_future[b]) ** 2).mean()
            assert torch.allclose(u[b, i], expect, atol=1e-5)


def test_greedy_set_oracle_empty_prefix_equals_individual_oracle():
    """With S = {} the weighted aggregate of {i} is y_i, so the two Oracles
    must coincide at t=1 -- a structural cross-check of both definitions."""
    _, _, futures, q_future, cand_mask, _, w_host, _ = _fixture()
    empty = torch.zeros(B, 0, dtype=torch.long)
    u_set = T.greedy_set_utility(empty, w_host, futures, q_future, chunk_size=None)
    u_ind = T.individual_utility(futures, q_future)
    assert torch.allclose(u_set[cand_mask], u_ind[cand_mask], atol=1e-5)
    # At INVALID positions w_i = 0, so `dense_utility`'s clamped denominator
    # makes the aggregate 0 and the two Oracles disagree there. That is
    # harmless only because every consumer masks with `valid_now` before the
    # argmax and before the CE -- asserted by the next test.
    assert not torch.allclose(u_set[~cand_mask], u_ind[~cand_mask], atol=1e-5)


def test_invalid_candidates_can_never_be_chosen_or_enter_the_loss():
    """The Oracle target is garbage at invalid positions (previous test), so
    the masking is what keeps both Oracles well-defined."""
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    for target in ('individual', 'greedy_set'):
        for policy in ('tf', 'onpolicy'):
            _, _, picks, steps = T.run_sequence(
                z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                target, policy, TAU, K, None)
            for st in steps:
                assert bool(cand_mask.gather(1, st['oracle_idx'].unsqueeze(-1)).all())
                assert bool(cand_mask.gather(1, st['model_idx'].unsqueeze(-1)).all())
                assert not bool((st['valid_now'] & ~cand_mask).any())
            for b in range(B):
                assert bool(cand_mask[b, picks[b]].all())


def test_host_weights_never_underflow_dense_utility_clamp():
    """`dense_utility` clamps the aggregate denominator at 1e-12. With cosine
    host scores (range [-1, 1]) and tau=0.1 the smallest row-max-shifted
    weight is exp(-2/0.1) = 2.06e-9, so no valid candidate can be silently
    mapped to a zero-weight aggregate. This bound is what makes the Greedy
    Set Oracle well-defined at t=1."""
    scores = torch.tensor([[1.0, -1.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    w = candidate_weights(scores, mask, TAU)
    assert float(w[w > 0].min()) > 1e-11
    assert float(w.max()) == pytest.approx(1.0, abs=1e-6)


def test_oracle_and_aggregation_weights_are_host_derived_not_arm_derived():
    """The Greedy Set Oracle must take w from the caller (host), never
    recompute it from the arm's own score."""
    src = inspect.getsource(T.greedy_set_utility)
    body = src.split('"""')[-1]
    assert 'candidate_weights' not in body
    assert 'arm_score' not in body
    assert 'w_host' in inspect.signature(T.greedy_set_utility).parameters


# ---------------------------------------------------------------- prefix ---
def _const_scorer(preference):
    """A fake scorer whose ranking is fixed, so the prefix policy is the only
    thing that can move the trajectory."""
    class M(torch.nn.Module):
        def score(self, z_q, z_k):
            return preference.expand(z_q.size(0), -1).clone()
        def train(self, mode=True):
            return self
    return M()


def test_teacher_forcing_follows_oracle_and_onpolicy_follows_model():
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    pref = torch.zeros(1, N)
    pref[0, 0] = 10.0      # the model always wants candidate 0 first
    metric = _const_scorer(pref)
    u_ind = T.individual_utility(futures, q_future)
    oracle_first = u_ind.masked_fill(~cand_mask, torch.finfo(u_ind.dtype).min / 4).argmax(-1)

    _, _, picks_tf, _ = T.run_sequence(
        z_q, E, cand_mask, sc, metric, w_host, futures, q_future,
        'individual', 'tf', TAU, K, None)
    _, _, picks_op, _ = T.run_sequence(
        z_q, E, cand_mask, sc, metric, w_host, futures, q_future,
        'individual', 'onpolicy', TAU, K, None)

    assert torch.equal(picks_tf[:, 0], oracle_first)
    assert torch.equal(picks_op[:, 0], torch.zeros(B, dtype=torch.long))


def test_t1_state_and_logits_identical_for_tf_and_onpolicy():
    """Spec S18: at t=1 the prefix is empty for both regimes, so the first
    step's logits and Oracle target must be bit-identical."""
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    for target in ('individual', 'greedy_set'):
        _, _, _, s_tf = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                       futures, q_future, target, 'tf', TAU, K, None)
        _, _, _, s_op = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                       futures, q_future, target, 'onpolicy', TAU, K, None)
        assert torch.equal(s_tf[0]['u_hat'], s_op[0]['u_hat'])
        assert torch.equal(s_tf[0]['u_target'], s_op[0]['u_target'])
        assert torch.equal(s_tf[0]['valid_now'], s_op[0]['valid_now'])


def test_onpolicy_target_is_the_oracle_not_the_model_choice():
    """Spec S9: the model's own pick must never be used as the CE label."""
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    pref = torch.zeros(1, N)
    pref[0, 0] = 10.0
    metric = _const_scorer(pref)
    _, _, picks, steps = T.run_sequence(
        z_q, E, cand_mask, sc, metric, w_host, futures, q_future,
        'individual', 'onpolicy', TAU, K, None)
    for st in steps:
        # the recorded label is the oracle's argmax, computed from u_target
        recomputed = st['u_target'].masked_fill(
            ~st['valid_now'], torch.finfo(st['u_target'].dtype).min / 4).argmax(-1)
        assert torch.equal(st['oracle_idx'], recomputed)
    assert not torch.equal(steps[0]['oracle_idx'], steps[0]['model_idx'])


# ------------------------------------------------------------ free-running -
def test_free_running_never_touches_the_future():
    """Poison the future tensors with NaN: free-running picks must be
    unchanged, proving no future information reaches selection (spec S15)."""
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    _, _, clean, _ = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                    futures, q_future, 'greedy_set', 'tf',
                                    TAU, K, None, free_running=True)
    _, _, poisoned, _ = T.run_sequence(
        z_q, E, cand_mask, sc, None, w_host,
        torch.full_like(futures, float('nan')), torch.full_like(q_future, float('nan')),
        'greedy_set', 'tf', TAU, K, None, free_running=True)
    assert torch.equal(clean, poisoned)


def test_free_running_is_identical_across_all_three_axes_given_same_weights():
    """Inference must not branch on target/prefix_policy (spec S15)."""
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    ref = None
    for target in ('individual', 'greedy_set'):
        for policy in ('tf', 'onpolicy'):
            _, _, p, _ = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                        futures, q_future, target, policy,
                                        TAU, K, None, free_running=True)
            if ref is None:
                ref = p
            else:
                assert torch.equal(ref, p)


def test_no_duplicate_and_no_invalid_picks():
    z_q, E, futures, q_future, cand_mask, _, w_host, sc = _fixture()
    for target in ('individual', 'greedy_set'):
        for policy in ('tf', 'onpolicy'):
            for fr in (False, True):
                _, _, picks, _ = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                                futures, q_future, target, policy,
                                                TAU, K, None, free_running=fr)
                for b in range(B):
                    row = picks[b].tolist()
                    assert len(set(row)) == len(row), (target, policy, fr, row)
                    assert bool(cand_mask[b, picks[b]].all())


# ------------------------------------------------------------- aggregation -
def test_primary_metric_matches_stage2_aggregation_exactly():
    """`free_running_aggregate_future_mse` must equal what Stage-2 computes
    for the same forced indices (spec S16)."""
    _, _, futures, q_future, cand_mask, host_scores, _, _ = _fixture()
    picks = torch.stack([torch.arange(K) for _ in range(B)])
    z_dummy = torch.zeros(B, 2)
    retrieved, alpha, top_idx, _, _ = retrieve_relation_future(
        z_q=z_dummy, z_mem=torch.zeros(N, 2), memory_value_c=futures[0],
        valid_mask=cand_mask, top_k=K, tau_topk=TAU,
        similarity='cosine', soft_all=False,
        score_fn=lambda a, b: host_scores, forced_idx=picks)
    mine_alpha = torch.softmax(host_scores.gather(1, picks) / TAU, dim=-1)
    assert torch.allclose(alpha, mine_alpha, atol=1e-6)
    ours = T.free_running_aggregate_future_mse(picks, host_scores, futures, q_future, TAU)
    tgt = futures.gather(1, picks.unsqueeze(-1).expand(-1, -1, H))
    expect = ((mine_alpha.unsqueeze(-1) * tgt).sum(1) - q_future).pow(2).mean(-1)
    assert torch.allclose(ours, expect, atol=1e-6)


# ------------------------------------------------------------------ scorer -
def test_asymmetric_scorer_is_identity_at_init():
    metric = RetrievalMetric(kind='asymmetric', dim=D, output='cosine', layer_norm=False)
    dev = float(cosine_init_deviation(metric))
    assert dev < 1e-5, dev
    z_q = torch.randn(B, D)
    E = torch.randn(N, D)
    a = T.arm_score(z_q, E, metric)
    b = T.arm_score(z_q, E, None)
    assert torch.allclose(a, b, atol=1e-5)


def test_cosine_arm_uses_normalised_dot_product():
    z_q = torch.randn(B, D)
    E = torch.randn(N, D)
    s = T.arm_score(z_q, E, None)
    expect = torch.matmul(F.normalize(z_q, dim=-1), F.normalize(E, dim=-1).T)
    assert torch.allclose(s, expect)
    assert s.max() <= 1.0 + 1e-5 and s.min() >= -1.0 - 1e-5


# -------------------------------------------------------------------- loss -
def test_all_arms_use_the_same_shared_choice_ce():
    src = inspect.getsource(T)
    assert 'from scripts.train_oracle_choice01 import oracle_choice_step_loss' in src
    body = src.split('def run_sequence(')[1].split('\ndef ')[0]
    for forbidden in ('smooth_l1', 'SmoothL1', 'margin_ranking', 'kl_div', 'pairwise'):
        assert forbidden not in body, forbidden
    assert body.count('oracle_choice_step_loss(') == 1


# ------------------------------------------------------------- gradients ---
def test_encoder_receives_gradient_from_choice_ce():
    """Spec S3: the encoder is trainable and CE gradient must reach it."""
    torch.manual_seed(0)
    enc = torch.nn.Linear(6, D)
    sc = SetConditioner(D)
    raw_q = torch.randn(B, 6)
    raw_k = torch.randn(N, 6)
    z_q, E = enc(raw_q), enc(raw_k)
    _, _, futures, q_future, cand_mask, _, w_host, _ = _fixture()
    losses, _, _, _ = T.run_sequence(z_q, E, cand_mask, sc, None, w_host,
                                     futures, q_future, 'individual', 'tf', TAU, K, None)
    sum(losses).backward()
    assert enc.weight.grad is not None
    assert float(enc.weight.grad.abs().sum()) > 0


def test_argmax_selection_is_detached():
    src = inspect.getsource(T.run_sequence)
    body = src.split('"""')[2]            # executable code only, skip the docstring
    for line in body.splitlines():
        if '.argmax(' in line and 'oracle_next' not in line:
            assert '.detach()' in line, line


# ------------------------------------------------------- representation ----
def test_representation_diagnostics_detect_collapse():
    healthy = torch.randn(64, D)
    # Real collapse (as observed in EXP-ENCODER-UNFREEZE01) is the embeddings
    # occupying ONE direction, not being literally identical -- the project's
    # own `embedding_geometry` also centers before taking the spectrum.
    direction = F.normalize(torch.randn(1, D), dim=-1)
    collapsed = torch.randn(64, 1) * direction + 1e-6 * torch.randn(64, D)
    h = T.representation_diagnostics(healthy)
    c = T.representation_diagnostics(collapsed)
    assert h['effective_rank'] > c['effective_rank']
    assert c['sv1_fraction'] > 0.99
    assert not h['has_nan'] and not c['has_nan']
    nan = T.representation_diagnostics(torch.full((16, D), float('nan')))
    assert nan['has_nan']


def test_state_sha_is_order_independent_and_value_sensitive():
    a = {'x': torch.ones(3), 'y': torch.zeros(2)}
    b = {'y': torch.zeros(2), 'x': torch.ones(3)}
    assert T.state_sha(a) == T.state_sha(b)
    c = {'x': torch.ones(3) * 1.0001, 'y': torch.zeros(2)}
    assert T.state_sha(a) != T.state_sha(c)


# ------------------------------------------------------------ stage-2 path -
def test_stage2_eval_reads_base_before_fusion_and_guards_relation_top_n():
    src = Path('scripts/eval_factorial_e2e01_stage2.py').read_text()
    assert 'y_final, y_base, y_ret, beta, lam, debug' in src
    assert "relation_top_n" in src and "Refusing to evaluate" in src
    assert 'set_forced_selection(forced)' in src
    assert 'set_forced_selection(None)' in src
    assert 'free_running=True' in src
    assert 'torch.equal(ref, base_all)' in src


def test_stage2_eval_never_passes_host_weights_into_selection():
    """Selection must not see the host weighting; only the metric does."""
    src = Path('scripts/eval_factorial_e2e01_stage2.py').read_text()
    call = src.split('run_sequence(')[1].split(')')[0]
    assert 'None,' in call          # w_host argument is None on the selection call


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
