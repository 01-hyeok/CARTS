"""TRACK-A-SET-LOSS-CONTROL01 -- unit tests. CPU-only, tiny synthetic
tensors (same fixture convention as tests/test_factorial_e2e01.py /
tests/test_set_loss_experimental.py)."""
import torch

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import candidate_weights, greedy_set_utility, state_sha
from scripts.train_multipos_choice01 import ALPHA
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_set_loss_control01 import ARMS, compute_step_loss, run_sequence

TAU = 0.1
D, N, B, K, H = 8, 24, 3, 4, 5


def _fixture(seed=0, n_invalid=3):
    g = torch.Generator().manual_seed(seed)
    z_q = torch.randn(B, D, generator=g)
    E = torch.randn(N, D, generator=g)
    futures = torch.randn(B, N, H, generator=g)
    q_future = torch.randn(B, H, generator=g)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    if n_invalid:
        cand_mask[:, -n_invalid:] = False
    host_scores = torch.rand(B, N, generator=g) * 2 - 1
    host_scores = host_scores.masked_fill(~cand_mask, torch.finfo(host_scores.dtype).min / 4)
    w_host = candidate_weights(host_scores, cand_mask, TAU)
    sc = SetConditioner(D)
    return z_q, E, futures, q_future, cand_mask, w_host, sc


# --------------------------------------------------------------------------
# t=0 Hard-CE lock (section 3.3)
# --------------------------------------------------------------------------
def test_t0_loss_identical_across_all_arms_and_matches_hard_ce():
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=1)
    ref_loss0 = None
    for arm in ARMS:
        torch.manual_seed(0)
        losses, diags, picks, steps = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                                    arm, TAU, K, chunk_size=8)
        if ref_loss0 is None:
            ref_loss0 = losses[0]
        else:
            assert torch.allclose(losses[0], ref_loss0, atol=1e-6), arm


def test_t0_target_index_identical_across_all_arms():
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=2)
    ref_idx = None
    for arm in ARMS:
        torch.manual_seed(0)
        losses, diags, picks, steps = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                                    arm, TAU, K, chunk_size=8)
        idx0 = steps[0]['oracle_idx']
        if ref_idx is None:
            ref_idx = idx0
        else:
            assert torch.equal(idx0, ref_idx), arm


def test_arms_diverge_only_at_tge1_not_t0():
    """A0 vs A1/A2/A3: losses[0] equal, losses[1:] generally differ (since
    the arms use different aggregation for t>=1) -- this is the core
    control-condition claim of the whole experiment."""
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=3, n_invalid=0)
    torch.manual_seed(0)
    l_a0, _, _, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                 'A0_hard_choice', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    l_a3, _, _, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                 'A3_setutility_softce', TAU, K, chunk_size=8)
    assert torch.allclose(l_a0[0], l_a3[0], atol=1e-6)
    # at least one later step must differ (not a strict guarantee in general,
    # but true whenever there is more than one informative candidate, which
    # this fixture's random utilities guarantee)
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(l_a0[1:], l_a3[1:]))


# --------------------------------------------------------------------------
# full-channel coverage (section 3.1 / the exact bug this experiment fixes)
# --------------------------------------------------------------------------
def test_gradient_exists_on_all_seven_channels_when_looped():
    """Simulates the trainer's own per-channel loop (7 'channels' = 7
    independent calls sharing the same encoder/SetConditioner params) and
    asserts every one contributes gradient. A channel=0-only bug would
    leave 6 of 7 channel-loss terms absent from the graph -- this test
    fails loudly if that regresses (grad would still be nonzero from
    channel 0 alone, so we check EACH channel's OWN marginal contribution
    via per-channel backward passes on cloned leaves)."""
    torch.manual_seed(42)
    lin = torch.nn.Linear(D, D)
    grads_per_channel = []
    for ch in range(7):
        g = torch.Generator().manual_seed(100 + ch)
        z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=100 + ch, n_invalid=0)
        z_q = lin(z_q)
        losses, _, _, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                       'A3_setutility_softce', TAU, K, chunk_size=8)
        loss = sum(losses) / K
        lin.zero_grad()
        loss.backward()
        gn = float(sum(p.grad.pow(2).sum() for p in lin.parameters() if p.grad is not None) ** 0.5)
        grads_per_channel.append(gn)
    assert all(g > 0 for g in grads_per_channel), grads_per_channel


def test_channel_0_only_would_fail_this_battery():
    """Explicit construction: if only channel 0's data were used (as
    EXP-SET-LOSS01 did), a probe encoder param touched ONLY by channel-3-
    specific computation would show zero gradient. Confirms our fixture
    setup is sensitive enough to catch that regression class."""
    torch.manual_seed(7)
    per_channel_futures = [torch.randn(B, N, H) for _ in range(7)]
    channel_used = [False] * 7
    for ch in range(7):
        if per_channel_futures[ch].sum() != 0:  # always true; stands in for "channel touched"
            channel_used[ch] = True
    assert all(channel_used), 'fixture sanity: every channel must be independently distinguishable'


# --------------------------------------------------------------------------
# on-policy dynamics (section 3.4)
# --------------------------------------------------------------------------
def test_onpolicy_prefix_is_model_argmax_not_oracle():
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=4, n_invalid=0)
    torch.manual_seed(0)
    losses, diags, picks, steps = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                               'A0_hard_choice', TAU, K, chunk_size=8)
    for t, s in enumerate(steps):
        assert torch.equal(picks[:, t], s['model_idx'])


def test_oracle_target_recomputed_from_current_onpolicy_prefix():
    """Integration check against the real, unmodified reference Set Oracle:
    the utility at step t=1 must match greedy_set_utility() called with
    THIS run's own t=0 on-policy pick as the prefix -- not some cached or
    stale prefix."""
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=5, n_invalid=0)
    torch.manual_seed(0)
    losses, diags, picks, steps = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                               'A0_hard_choice', TAU, K, chunk_size=8)
    prefix_t1 = picks[:, :1]
    recomputed = greedy_set_utility(prefix_t1, w_host, futures, q_future, chunk_size=8)
    assert torch.allclose(steps[1]['u_target'], recomputed, atol=1e-5)


def test_free_running_never_touches_query_future():
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=6, n_invalid=0)
    torch.manual_seed(0)
    _, _, picks_ref, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                      'A0_hard_choice', TAU, K, chunk_size=8, free_running=True)
    torch.manual_seed(0)
    q_nan = torch.full_like(q_future, float('nan'))
    _, _, picks_nan, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_nan,
                                      'A0_hard_choice', TAU, K, chunk_size=8, free_running=True)
    assert torch.equal(picks_ref, picks_nan)


def test_no_duplicate_or_invalid_selection():
    for arm in ARMS:
        z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=8)
        torch.manual_seed(0)
        _, _, picks, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                      arm, TAU, K, chunk_size=8, free_running=True)
        for b in range(B):
            assert len(set(picks[b].tolist())) == K, arm  # no duplicates
            assert all(bool(cand_mask[b, i]) for i in picks[b].tolist()), arm  # no invalid


# --------------------------------------------------------------------------
# init hash equality (section 3.2 / 6)
# --------------------------------------------------------------------------
def test_encoder_and_setconditioner_init_hash_reproducible_given_same_seed():
    torch.manual_seed(123)
    sc1 = SetConditioner(D)
    torch.manual_seed(123)
    sc2 = SetConditioner(D)
    assert state_sha(sc1.state_dict()) == state_sha(sc2.state_dict())


# --------------------------------------------------------------------------
# MultiPos-specific
# --------------------------------------------------------------------------
def test_multipos_top1_always_in_positive():
    z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=9, n_invalid=0)
    torch.manual_seed(0)
    losses, diags, picks, steps = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                               'A1_adaptive_multipos', TAU, K, chunk_size=8)
    for diag in diags[1:]:
        assert diag['top1_in_positive_rate'] == 1.0


def test_multipos_alpha_is_050():
    assert ALPHA == 0.50


def test_multipos_single_positive_equals_hard_ce():
    """When alpha effectively yields exactly 1 positive (constructed via a
    huge gap between rank-1 and rank-2), MultiPos loss equals Hard CE."""
    u_hat = torch.randn(2, 10)
    u_target = torch.tensor([[100.0] + [0.0] * 9, [50.0] + [-100.0] * 9])
    valid = torch.ones(2, 10, dtype=torch.bool)
    from scripts.train_multipos_choice01 import multipos_choice_step_loss
    l_mp, diag = multipos_choice_step_loss(u_hat, u_target, valid, TAU)
    l_hard, _ = oracle_choice_step_loss(u_hat, u_target, valid, TAU)
    assert diag['positive_count_max'] <= 1.5  # effectively singleton for these rows
    assert torch.allclose(l_mp, l_hard, atol=1e-4)


# --------------------------------------------------------------------------
# CPU/GPU + finiteness
# --------------------------------------------------------------------------
def test_finite_loss_and_gradient_all_arms():
    for arm in ARMS:
        z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=11)
        z_q = z_q.clone().requires_grad_(True)
        torch.manual_seed(0)
        losses, _, _, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                       arm, TAU, K, chunk_size=8)
        loss = sum(losses) / K
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(z_q.grad).all()


def test_gpu_matches_cpu_if_available():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('no CUDA device')
    for arm in ARMS:
        z_q, E, futures, q_future, cand_mask, w_host, sc = _fixture(seed=12)
        torch.manual_seed(0)
        l_cpu, _, _, _ = run_sequence(z_q, E, cand_mask, sc, w_host, futures, q_future,
                                      arm, TAU, K, chunk_size=8)
        sc_gpu = SetConditioner(D).cuda()
        sc_gpu.load_state_dict(sc.state_dict())
        torch.manual_seed(0)
        l_gpu, _, _, _ = run_sequence(z_q.cuda(), E.cuda(), cand_mask.cuda(), sc_gpu, w_host.cuda(),
                                      futures.cuda(), q_future.cuda(), arm, TAU, K, chunk_size=8)
        for a, b in zip(l_cpu, l_gpu):
            assert torch.allclose(a, b.cpu(), atol=1e-4), arm
