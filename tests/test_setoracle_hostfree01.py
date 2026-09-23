"""TRACK-A-SETORACLE-HOSTFREE01 sanity tests -- UniformHost must make the
EXISTING weighted-aggregate math (`utils/dense_utility.py`) reduce to a
literal, unweighted arithmetic mean, with no new aggregation code path."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setoracle_hostfree01 import UniformHost
from utils.dense_utility import candidate_weights


def test_uniform_host_scores_are_equal_and_masked_correctly():
    host = UniformHost(top_k=10, tau_topk=1.0)
    cand_mask = torch.tensor([[True, True, False, True]])
    batch_x = torch.randn(1, 8, 3)
    s = host.scores(batch_x, c=0, cand_mask=cand_mask)
    assert s.shape == cand_mask.shape
    # all entries equal (0.0) regardless of validity -- masking happens
    # downstream in candidate_weights, not inside scores() itself
    assert torch.equal(s, torch.zeros_like(s))


@pytest.mark.parametrize('tau', [0.05, 1.0, 10.0])
def test_candidate_weights_from_uniform_host_are_literally_uniform(tau):
    """Regardless of tau, equal scores -> equal (all-1) weights at valid
    positions, 0 at invalid -- this is candidate_weights' own documented
    behavior (exp((s-max)/tau)), not anything new written for this test."""
    host = UniformHost(top_k=10, tau_topk=tau)
    cand_mask = torch.tensor([[True, True, False, True, True]])
    s = host.scores(torch.randn(1, 8, 3), c=0, cand_mask=cand_mask)
    w = candidate_weights(s, cand_mask, tau)
    valid_w = w[cand_mask]
    assert torch.allclose(valid_w, torch.ones_like(valid_w))
    assert torch.equal(w[~cand_mask], torch.zeros_like(w[~cand_mask]))


def test_uniform_weighted_aggregate_equals_literal_mean():
    """End-to-end: candidate_weights(UniformHost.scores(...)) fed into the
    SAME dense_utility-style weighted aggregate used by greedy_set_utility
    must equal a plain torch.mean over the selected set -- verified by
    direct re-derivation, not by trusting the formula's own comment."""
    torch.manual_seed(0)
    bsz, n, h = 2, 6, 4
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)

    host = UniformHost(top_k=3, tau_topk=1.0)
    s = host.scores(torch.randn(bsz, 8, 3), c=0, cand_mask=cand_mask)
    w = candidate_weights(s, cand_mask, host.tau_topk)

    from utils.dense_utility import dense_utility, prefix_weighted_sums
    prefix_idx = torch.tensor([[0, 1], [2, 3]])  # 2 candidates already picked
    z_s, m_s = prefix_weighted_sums(prefix_idx, w, futures)
    # simulate adding candidate index 4 for every row (matches dense_utility's
    # own per-candidate trial construction for one specific candidate)
    trial_idx = 4
    w_c = w[:, trial_idx:trial_idx + 1]
    y_c = futures[:, trial_idx:trial_idx + 1, :]
    trial_num = m_s.unsqueeze(1) + w_c.unsqueeze(-1) * y_c
    trial_den = (z_s + w_c).unsqueeze(-1).clamp_min(1e-12)
    y_ret = (trial_num / trial_den).squeeze(1)

    manual_mean = torch.stack([
        futures[b, prefix_idx[b].tolist() + [trial_idx], :].mean(dim=0)
        for b in range(bsz)
    ], dim=0)
    assert torch.allclose(y_ret, manual_mean, atol=1e-5), \
        f'uniform-weighted aggregate {y_ret} != literal mean {manual_mean}'


def test_arm_mapping_matches_target_axis():
    from scripts.train_setoracle_hostfree01 import ARMS
    assert ARMS['individual_tf_hostfree_cosine'] == 'individual'
    assert ARMS['set_tf_hostfree_cosine'] == 'greedy_set'
