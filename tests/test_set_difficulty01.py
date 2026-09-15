"""Mandatory sanity checks for TRACK-A-SET-DIFFICULTY01 (spec §17).

CPU-only where possible. Diagnostic-only script -- no training path to test,
so these focus on: correctness of the new math (ambiguity gaps, near-tie
counting, P1 perturbation, overlap/spearman/kendall), the Individual-Oracle
prefix-invariance control, checkpoint-resolution discipline (no guessing, no
silent fallback), and leak-free free-running selection.
"""
import json

import pytest
import torch

from scripts import diag_set_difficulty01 as D
from scripts.train_factorial_e2e01 import individual_utility

B, N, H = 4, 20, 5


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    futures = torch.randn(B, N, H, generator=g)
    q_future = torch.randn(B, H, generator=g)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, -4:] = False
    return futures, q_future, cand_mask


# --------------------------------------------------------------- ambiguity -
def test_ambiguity_gaps_are_nonneg_and_ordered():
    futures, q_future, cand_mask = _fixture()
    u_target = individual_utility(futures, q_future)
    amb, gap12 = D.ambiguity_diagnostics(u_target, cand_mask)
    assert amb['gap_1_2_mean'] >= 0
    assert amb['gap_1_5_mean'] >= amb['gap_1_2_mean'] - 1e-6
    assert amb['gap_1_10_mean'] >= amb['gap_1_5_mean'] - 1e-6
    assert gap12.shape == (B,)
    assert bool((gap12 >= -1e-6).all())


def test_near_tie_count_monotonic_in_frac():
    """A larger `frac` widens the near-tie band, so the count must not
    decrease."""
    futures, q_future, cand_mask = _fixture()
    u_target = individual_utility(futures, q_future)
    amb, _ = D.ambiguity_diagnostics(u_target, cand_mask)
    counts = [amb[f'near_tie_count_frac{f}'] for f in D.NEAR_TIE_FRACS]
    assert counts == sorted(counts)
    # at frac=0 (approximated by the smallest registered frac) every count
    # must be >= 1 (the best candidate itself is always within its own band)
    assert counts[0] >= 1.0 - 1e-6


def test_near_tie_count_exact_on_synthetic_utility():
    """Exact check: utilities 0, -1, -2, ..., -9 (range=9). At frac=0.05,
    eps=0.45, so candidates with u >= -0.45 qualify -> only rank-1 itself."""
    u_target = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]])
    valid = torch.ones(1, 10, dtype=torch.bool)
    amb, _ = D.ambiguity_diagnostics(u_target, valid)
    assert amb['near_tie_count_frac0.05'] == pytest.approx(1.0, abs=1e-6)
    assert amb['gap_1_10_mean'] == pytest.approx(9.0, abs=1e-6)


def test_near_tie_count_with_true_ties():
    """Two candidates share the top utility -> near_tie_count at the
    smallest frac must be >= 2."""
    u_target = torch.tensor([[0.0, 0.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]])
    valid = torch.ones(1, 10, dtype=torch.bool)
    amb, _ = D.ambiguity_diagnostics(u_target, valid)
    assert amb['near_tie_count_frac0.001'] >= 2.0 - 1e-6


# --------------------------------------------------------- perturbation P1 -
def test_p1_perturbation_replaces_only_the_last_element():
    prefix_before = torch.tensor([[3, 7]])
    u_hat = torch.zeros(1, N)
    u_hat[0, 2] = 5.0   # best
    u_hat[0, 9] = 4.0   # second-best
    valid_now = torch.ones(1, N, dtype=torch.bool)
    model_idx = torch.tensor([2])
    out = D.perturb_prefix_p1(prefix_before, u_hat, valid_now, model_idx)
    assert out.shape == (1, 3)
    assert int(out[0, 0]) == 3 and int(out[0, 1]) == 7  # untouched
    assert int(out[0, 2]) == 9  # replaced by second-best


def test_p1_perturbation_at_t2_empty_prefix():
    prefix_before = torch.zeros(1, 0, dtype=torch.long)
    u_hat = torch.zeros(1, N)
    u_hat[0, 0] = 5.0
    u_hat[0, 1] = 4.0
    valid_now = torch.ones(1, N, dtype=torch.bool)
    model_idx = torch.tensor([0])
    out = D.perturb_prefix_p1(prefix_before, u_hat, valid_now, model_idx)
    assert out.shape == (1, 1)
    assert int(out[0, 0]) == 1


def test_p1_never_picks_an_invalid_candidate():
    prefix_before = torch.zeros(1, 0, dtype=torch.long)
    u_hat = torch.arange(N).float().unsqueeze(0)  # candidate N-1 is best
    valid_now = torch.ones(1, N, dtype=torch.bool)
    valid_now[0, -1] = False  # invalidate the true best
    valid_now[0, -2] = False  # invalidate the true second-best
    model_idx = torch.tensor([N - 3])  # actual pick among the still-valid ones
    out = D.perturb_prefix_p1(prefix_before, u_hat, valid_now, model_idx)
    assert bool(valid_now[0, out[0, 0]])


# -------------------------------------------------- overlap/spearman/kendall
def test_overlap_at_k_identical_orders():
    a = torch.tensor([3, 1, 4, 0, 2])
    assert D.overlap_at_k(a, a, 5) == pytest.approx(1.0)


def test_overlap_at_k_disjoint():
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    assert D.overlap_at_k(a, b, 3) == pytest.approx(0.0)


def test_spearman_perfect_positive_and_negative():
    a = torch.arange(10).float()
    valid = torch.ones(10, dtype=torch.bool)
    sp, kt = D.spearman_kendall(a, a, valid)
    assert sp == pytest.approx(1.0, abs=1e-4)
    assert kt == pytest.approx(1.0, abs=1e-4)
    sp2, kt2 = D.spearman_kendall(a, -a, valid)
    assert sp2 == pytest.approx(-1.0, abs=1e-4)
    assert kt2 == pytest.approx(-1.0, abs=1e-4)


def test_spearman_kendall_masks_invalid():
    a = torch.tensor([0.0, 1.0, 2.0, 100.0])
    b = torch.tensor([0.0, 1.0, 2.0, -100.0])
    valid = torch.tensor([True, True, True, False])
    sp, kt = D.spearman_kendall(a, b, valid)
    assert sp == pytest.approx(1.0, abs=1e-4)  # the outlier at index 3 excluded


# ----------------------------------------------------- Individual invariance
def test_individual_oracle_is_provably_prefix_invariant():
    """The control described in spec S10: `individual_utility` must not even
    accept a prefix argument, so invariance holds by construction, not by
    accident."""
    import inspect
    sig = inspect.signature(individual_utility)
    assert list(sig.parameters) == ['futures', 'query_future']
    futures, q_future, cand_mask = _fixture()
    u_a = individual_utility(futures, q_future)
    u_b = individual_utility(futures, q_future)
    assert torch.equal(u_a, u_b)


# --------------------------------------------------- checkpoint resolution -
def test_resolve_checkpoint_raises_on_missing_json(tmp_path):
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(str(tmp_path), 'ETTh1_96', 'individual_onpolicy_cosine')


def test_resolve_checkpoint_raises_on_missing_ckpt_file(tmp_path):
    cell_dir = tmp_path / 'ETTh1_96'
    cell_dir.mkdir()
    (cell_dir / 'retrieval_metrics_individual_onpolicy_cosine.json').write_text(
        json.dumps({'checkpoint': str(tmp_path / 'nonexistent.pth'), 'best_epoch': 1}))
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(str(tmp_path), 'ETTh1_96', 'individual_onpolicy_cosine')


def test_resolve_checkpoint_reads_the_real_factorial_result_if_present():
    """If the actual campaign artifact exists, resolution must read it
    verbatim (no path guessing)."""
    import os
    root = 'results/track_a_factorial_e2e'
    if not os.path.exists(f'{root}/ETTh1_96/retrieval_metrics_individual_onpolicy_cosine.json'):
        pytest.skip('factorial artifact not present in this environment')
    ckpt_path, d = D.resolve_checkpoint(root, 'ETTh1_96', 'individual_onpolicy_cosine')
    assert ckpt_path == d['checkpoint']
    assert os.path.exists(ckpt_path)


# ------------------------------------------------------------- free-running
def test_free_running_trajectory_never_touches_future_for_selection():
    """Poison futures/query_future with NaN: the model's OWN picks
    (`model_idx`, hence the actual free-running trajectory) must be
    unaffected, because selection reads only `u_hat` (model score), never
    `u_target`."""
    torch.manual_seed(0)
    D_dim = 6
    enc_state = lambda: None
    from models.SequentialSetRetriever import SetConditioner
    sc = SetConditioner(D_dim)

    class FakeModel:
        pass

    z_q = torch.randn(B, D_dim)
    E = torch.randn(N, D_dim)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    futures, q_future, _ = _fixture()

    class FakeModelWrapper:
        pass

    steps_clean, picks_clean = D.free_running_trajectory(
        FakeModelWrapper(), sc, z_q, E, cand_mask, 'individual', None, futures, q_future, k=4)
    steps_poison, picks_poison = D.free_running_trajectory(
        FakeModelWrapper(), sc, z_q, E, cand_mask, 'individual', None,
        torch.full_like(futures, float('nan')), torch.full_like(q_future, float('nan')), k=4)
    assert torch.equal(picks_clean, picks_poison)


def test_free_running_trajectory_no_duplicate_no_invalid():
    torch.manual_seed(1)
    from models.SequentialSetRetriever import SetConditioner
    D_dim = 6
    sc = SetConditioner(D_dim)
    z_q = torch.randn(B, D_dim)
    E = torch.randn(N, D_dim)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, -3:] = False
    futures, q_future, _ = _fixture()

    class FakeModelWrapper:
        pass

    for kind in ('individual', 'set'):
        w_host = torch.rand(B, N) if kind == 'set' else None
        if kind == 'set':
            cm = (torch.rand(B, N) * 2 - 1).masked_fill(~cand_mask, torch.finfo(torch.float32).min / 4)
            from utils.dense_utility import candidate_weights
            w_host = candidate_weights(cm, cand_mask, 0.1)
        steps, picks = D.free_running_trajectory(
            FakeModelWrapper(), sc, z_q, E, cand_mask, kind, w_host, futures, q_future, k=5)
        for b in range(B):
            row = picks[b].tolist()
            assert len(set(row)) == len(row)
            assert bool(cand_mask[b, picks[b]].all())


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
