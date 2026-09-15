"""TRACK-A-WEATHER-OPT04 -- production dispatcher wiring tests.

CPU-only, tiny synthetic tensors (same fixture convention as
`tests/test_factorial_e2e01.py`) -- pre-flight gate for the
`--oracle_compute_impl` flag added to `scripts/train_factorial_e2e01.py`.
Verifies: (1) the resolver's per-path logic including the chunk<4096
fallback, (2) `run_sequence` with every dispatcher combination is either
bit-exact (C, A) or agreement-verified (E, OPT02 algebra) against the
reference, (3) the DEFAULT (no flag, or `--oracle_compute_impl reference`)
path is byte-for-byte what `run_sequence` computed before this round --
the explicit "optimized flag가 없을 때 기존 동작과 동일" regression test.
"""
import torch

from scripts import train_factorial_e2e01 as T
from utils.dense_utility import candidate_weights
from models.SequentialSetRetriever import SetConditioner

D, N, B, K, H = 8, 24, 3, 4, 5
TAU = 0.1


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    z_q = torch.randn(B, D, generator=g)
    E = torch.randn(N, D, generator=g)
    futures = torch.randn(B, N, H, generator=g)
    q_future = torch.randn(B, H, generator=g)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, -3:] = False
    host_scores = torch.rand(B, N, generator=g) * 2.0 - 1.0
    host_scores = host_scores.masked_fill(~cand_mask, torch.finfo(host_scores.dtype).min / 4)
    w_host = candidate_weights(host_scores, cand_mask, TAU)
    sc = SetConditioner(D)
    return z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc


# --------------------------------------------------------------------------
# resolve_oracle_compute_impl
# --------------------------------------------------------------------------
def test_resolve_reference_is_reference_everywhere_with_reasons():
    r = T.resolve_oracle_compute_impl('reference', 4096)
    assert r['choice_ce_impl'] == 'reference'
    assert r['individual_impl'] == 'reference'
    assert r['greedy_set_impl'] == 'reference'
    assert all(r['fallback_reason'].values())  # every path explains itself


def test_resolve_optimized_at_min_chunk_size_enables_all_three():
    r = T.resolve_oracle_compute_impl('optimized', T.GREEDY_SET_OPTIMIZED_MIN_CHUNK_SIZE)
    assert r == {
        'choice_ce_impl': 'optimized', 'individual_impl': 'optimized_cache',
        'greedy_set_impl': 'optimized',
        'fallback_reason': {'choice_ce': '', 'individual': '', 'greedy_set': ''},
    }


def test_resolve_optimized_below_min_chunk_size_falls_back_greedy_set_only():
    r = T.resolve_oracle_compute_impl('optimized', T.GREEDY_SET_OPTIMIZED_MIN_CHUNK_SIZE - 1)
    assert r['choice_ce_impl'] == 'optimized'          # C: unconditional
    assert r['individual_impl'] == 'optimized_cache'   # A: unconditional
    assert r['greedy_set_impl'] == 'reference'          # E: falls back
    assert r['fallback_reason']['greedy_set'] != ''
    assert str(T.GREEDY_SET_OPTIMIZED_MIN_CHUNK_SIZE) in r['fallback_reason']['greedy_set']
    assert r['fallback_reason']['choice_ce'] == ''
    assert r['fallback_reason']['individual'] == ''


# --------------------------------------------------------------------------
# TRACK-A-WEATHER-OPT05 -- 'safe' mode (per-Oracle-type separation)
# --------------------------------------------------------------------------
def test_resolve_safe_enables_c_and_a_but_never_e_at_large_chunk():
    r = T.resolve_oracle_compute_impl('safe', 4096)
    assert r['choice_ce_impl'] == 'optimized'
    assert r['individual_impl'] == 'optimized_cache'
    assert r['greedy_set_impl'] == 'reference'
    assert r['fallback_reason']['greedy_set'] != ''
    assert r['fallback_reason']['choice_ce'] == ''
    assert r['fallback_reason']['individual'] == ''


def test_resolve_safe_never_enables_e_at_small_chunk_either():
    r = T.resolve_oracle_compute_impl('safe', 128)
    assert r['greedy_set_impl'] == 'reference'


def test_resolve_safe_greedy_set_never_optimized_across_a_chunk_size_sweep():
    """The exact regression 'safe' exists to prevent: unlike 'optimized',
    'safe' must NEVER flip greedy_set_impl to 'optimized' no matter how
    large chunk_size is."""
    for cs in (1, 128, 2048, 4096, 8192, 65536):
        assert T.resolve_oracle_compute_impl('safe', cs)['greedy_set_impl'] == 'reference'


def test_resolve_safe_vs_optimized_agree_on_c_and_a_disagree_on_e_at_large_chunk():
    safe = T.resolve_oracle_compute_impl('safe', 4096)
    opt = T.resolve_oracle_compute_impl('optimized', 4096)
    assert safe['choice_ce_impl'] == opt['choice_ce_impl'] == 'optimized'
    assert safe['individual_impl'] == opt['individual_impl'] == 'optimized_cache'
    assert safe['greedy_set_impl'] == 'reference'
    assert opt['greedy_set_impl'] == 'optimized'  # the risky bundle 'safe' avoids


def test_run_sequence_safe_greedy_set_bit_exact_vs_reference_at_large_chunk():
    """End-to-end proof at the run_sequence level: under 'safe' resolution,
    a greedy_set arm's picks/losses are BIT-EXACT vs full reference (E is
    never touched), even at chunk_size=4096 where 'optimized' would diverge."""
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=10)
    resolved = T.resolve_oracle_compute_impl('safe', 4096)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'greedy_set', 'onpolicy', TAU, K, chunk_size=4096)
    torch.manual_seed(0)
    safe = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                          'greedy_set', 'onpolicy', TAU, K, chunk_size=4096,
                          choice_ce_impl=resolved['choice_ce_impl'],
                          individual_impl=resolved['individual_impl'],
                          greedy_set_impl=resolved['greedy_set_impl'])
    assert torch.equal(ref[2], safe[2])  # picks bit-exact (E untouched)
    # choice_ce IS optimized under safe, so losses match the dead-code-elimination
    # guarantee (bit-exact), not merely close
    for lr, ls in zip(ref[0], safe[0]):
        assert torch.equal(lr, ls)


def test_resolve_invalid_impl_raises():
    import pytest
    with pytest.raises(ValueError):
        T.resolve_oracle_compute_impl('bogus', 4096)


def test_log_oracle_compute_resolution_prints_fallback_reason(capsys):
    r = T.resolve_oracle_compute_impl('optimized', 128)
    T.log_oracle_compute_resolution('optimized', r, 128)
    out = capsys.readouterr().out
    assert '[oracle-compute]' in out
    assert 'requested_impl=optimized' in out
    assert 'greedy_set_impl=reference' in out
    assert 'fallback_reason[greedy_set]=' in out
    assert 'fallback_reason[choice_ce]=' not in out  # no reason printed when none


# --------------------------------------------------------------------------
# run_sequence -- DEFAULT (no dispatcher args) is unchanged from pre-OPT04
# --------------------------------------------------------------------------
def test_run_sequence_default_args_equal_explicit_reference_args():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=7)
    torch.manual_seed(0)
    default = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                             'individual', 'tf', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    explicit = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                              'individual', 'tf', TAU, K, chunk_size=8,
                              choice_ce_impl='reference', individual_impl='reference',
                              greedy_set_impl='reference')
    for a, b in zip(default[0], explicit[0]):
        assert torch.equal(a, b)
    assert torch.equal(default[2], explicit[2])  # picks


# --------------------------------------------------------------------------
# C -- Oracle-Choice CE optimized dispatch: bit-exact vs reference
# --------------------------------------------------------------------------
def test_choice_ce_optimized_dispatch_bit_exact_individual_target():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=1)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'tf', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    opt = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'tf', TAU, K, chunk_size=8, choice_ce_impl='optimized')
    for lr, lo in zip(ref[0], opt[0]):
        assert torch.equal(lr, lo)
    assert torch.equal(ref[2], opt[2])


def test_choice_ce_optimized_dispatch_bit_exact_greedy_set_target():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=2)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'greedy_set', 'tf', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    opt = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'greedy_set', 'tf', TAU, K, chunk_size=8, choice_ce_impl='optimized')
    for lr, lo in zip(ref[0], opt[0]):
        assert torch.equal(lr, lo)
    assert torch.equal(ref[2], opt[2])


# --------------------------------------------------------------------------
# A -- Individual Oracle caching dispatch: bit-exact vs reference
# --------------------------------------------------------------------------
def test_individual_caching_dispatch_bit_exact_tf():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=3)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'tf', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    opt = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'tf', TAU, K, chunk_size=8, individual_impl='optimized_cache')
    for lr, lo in zip(ref[0], opt[0]):
        assert torch.equal(lr, lo)
    assert torch.equal(ref[2], opt[2])
    for sr, so in zip(ref[3], opt[3]):
        assert torch.equal(sr['u_target'], so['u_target'])
        assert torch.equal(sr['oracle_idx'], so['oracle_idx'])


def test_individual_caching_dispatch_bit_exact_onpolicy():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=4)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'onpolicy', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    opt = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'individual', 'onpolicy', TAU, K, chunk_size=8, individual_impl='optimized_cache')
    assert torch.equal(ref[2], opt[2])
    for lr, lo in zip(ref[0], opt[0]):
        assert torch.equal(lr, lo)


def test_individual_caching_only_calls_individual_utility_once():
    """Direct proof of the caching claim: monkeypatch `individual_utility` to
    count calls inside one `run_sequence` (K=4 steps)."""
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=5)
    calls = {'n': 0}
    real = T.individual_utility

    def counting(*a, **kw):
        calls['n'] += 1
        return real(*a, **kw)

    T.individual_utility = counting
    try:
        T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                       'individual', 'tf', TAU, K, chunk_size=8, individual_impl='optimized_cache')
    finally:
        T.individual_utility = real
    assert calls['n'] == 1  # not K=4


def test_individual_reference_impl_still_calls_every_step():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=6)
    calls = {'n': 0}
    real = T.individual_utility

    def counting(*a, **kw):
        calls['n'] += 1
        return real(*a, **kw)

    T.individual_utility = counting
    try:
        T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                       'individual', 'tf', TAU, K, chunk_size=8, individual_impl='reference')
    finally:
        T.individual_utility = real
    assert calls['n'] == K  # unchanged from pre-OPT04: once per step


# --------------------------------------------------------------------------
# E -- Greedy Set Oracle optimized dispatch: OPT02 algebra (agreement, not
# bit-exact -- see research/TRACK-A-WEATHER-OPT02.md's own tie discussion)
# --------------------------------------------------------------------------
def test_greedy_set_optimized_dispatch_selection_agrees_with_reference():
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=8)
    torch.manual_seed(0)
    ref = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'greedy_set', 'tf', TAU, K, chunk_size=8)
    torch.manual_seed(0)
    opt = T.run_sequence(z_q, E, cand_mask, sc, None, w_host, futures, q_future,
                         'greedy_set', 'tf', TAU, K, chunk_size=8, greedy_set_impl='optimized')
    assert torch.equal(ref[2], opt[2])  # tiny synthetic fixture: no ties expected
    for lr, lo in zip(ref[0], opt[0]):
        assert torch.allclose(lr, lo, atol=1e-4)


def test_greedy_set_utility_optimized_matches_greedy_set_utility():
    """Per the project's established contract (OPT01/OPT02): only VALID
    candidate positions are guaranteed to agree -- invalid (masked, w=0)
    positions are never read by any caller (masked to -inf before argmax),
    so the reference and optimized formulas are free to diverge there."""
    from utils.oracle_compute_optimized import prepare_query_static_chunked
    z_q, E, futures, q_future, cand_mask, host_scores, w_host, sc = _fixture(seed=9)
    prefix = torch.zeros(B, 0, dtype=torch.long)
    ref = T.greedy_set_utility(prefix, w_host, futures, q_future, chunk_size=8)
    d_sq = prepare_query_static_chunked(futures, q_future, candidate_chunk_size=8)
    opt = T.greedy_set_utility_optimized(prefix, w_host, futures, q_future, d_sq, chunk_size=8)
    assert torch.allclose(ref[cand_mask], opt[cand_mask], atol=1e-4)
