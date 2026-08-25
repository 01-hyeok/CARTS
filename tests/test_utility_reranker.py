"""Guards for the shortlist-reranking experiment.

The whole design rests on one claim: restricting the candidate mask to a chosen
set and running the real `forward` reproduces production aggregation on exactly
that set. If that is not true, every number the study produces is about a
pipeline nobody ships. These tests pin it down on tensors, plus the leakage and
shape rules the spec requires.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_oracle_rerank_headroom import (
    mse_mae_per_channel, restricted_mask,
)

B, N, C, K = 4, 20, 3, 5


def test_restricted_mask_admits_only_the_chosen_ids():
    """(1)(2) the shortlist is exactly what was asked for, never more."""
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    ids = torch.stack([torch.randperm(N)[:K] for _ in range(B)])
    mask = restricted_mask(cand_mask, ids)
    assert mask.sum(-1).tolist() == [K] * B
    for row in range(B):
        assert set(mask[row].nonzero().flatten().tolist()) == set(ids[row].tolist())


def test_restricted_mask_never_revives_an_invalid_candidate():
    """(7) reranking cannot smuggle a temporally invalid window back in."""
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, :K] = False                    # the chosen ids are all invalid
    ids = torch.arange(K).unsqueeze(0).expand(B, K)
    mask = restricted_mask(cand_mask, ids)
    assert mask.sum() == 0


def test_restricted_mask_is_a_subset_of_the_production_mask():
    torch.manual_seed(0)
    cand_mask = torch.rand(B, N) > 0.3
    ids = torch.stack([torch.randperm(N)[:K] for _ in range(B)])
    mask = restricted_mask(cand_mask, ids)
    assert bool((mask & ~cand_mask).sum() == 0)
    assert bool((mask <= cand_mask).all())


def test_full_shortlist_is_a_no_op_for_selection():
    """(2) 'reranker off' must leave the model with its original choice set."""
    cand_mask = torch.rand(B, N) > 0.3
    everything = torch.arange(N).unsqueeze(0).expand(B, N)
    assert torch.equal(restricted_mask(cand_mask, everything), cand_mask)


def test_shape_mismatch_fails_immediately():
    """(8)(9) no implicit broadcasting between forecast and target."""
    prediction = torch.randn(B, 8, C)
    with pytest.raises(ValueError):
        mse_mae_per_channel(prediction, torch.randn(B, 8, C + 1))
    with pytest.raises(ValueError):
        mse_mae_per_channel(prediction, torch.randn(B, 1, C))
    mse, mae = mse_mae_per_channel(prediction, torch.randn(B, 8, C))
    assert mse.shape == (B, C) and mae.shape == (B, C)


def test_metrics_match_a_plain_reference():
    prediction = torch.randn(B, 8, C)
    target = torch.randn(B, 8, C)
    mse, mae = mse_mae_per_channel(prediction, target)
    for row in range(B):
        for channel in range(C):
            expected = (prediction[row, :, channel] - target[row, :, channel])
            assert torch.allclose(mse[row, channel], expected.square().mean(), atol=1e-6)
            assert torch.allclose(mae[row, channel], expected.abs().mean(), atol=1e-6)


def test_no_query_future_in_the_selection_signature():
    """(5) the selection helpers take pasts and ids; labels stay outside."""
    import inspect

    from scripts.analyze_oracle_rerank_headroom import (
        channelwise_selection_forecast, forecast_with_mask,
    )
    for function in (forecast_with_mask, channelwise_selection_forecast):
        names = set(inspect.signature(function).parameters)
        assert not names & {'batch_y', 'query_y', 'utility', 'query_residual'}, (
            f'{function.__name__} can see a label'
        )


def test_headroom_columns_cover_the_required_table():
    from scripts.analyze_oracle_rerank_headroom import COLUMNS

    for required in ('original_topk_mse', 'oracle_individual_mse',
                     'oracle_best_single_mse', 'greedy_set_oracle_mse',
                     'original_set_utility', 'oracle_set_utility',
                     'oracle_rerank_gain'):
        assert required in COLUMNS


# --- Phase 2 reranker guards -------------------------------------------------

from models.utility_reranker import build_reranker  # noqa: E402
from scripts.train_utility_reranker import NEG_INF, choose, ranking_metrics  # noqa: E402

D, M = 8, 12


def test_reranker_forward_takes_no_label():
    """(5) query future, query residual and utility cannot reach the score."""
    import inspect

    reranker = build_reranker('residual_aware', D, horizon=4)
    names = set(inspect.signature(reranker.forward).parameters)
    assert not names & {'query_y', 'batch_y', 'utility', 'query_residual', 'future'}
    assert names == {'z_q', 'z_k', 'retriever_score', 'candidate_residual'}


def test_residual_arm_requires_its_inputs_and_checks_shapes():
    """(8)(9) no silent broadcasting between candidates and residuals."""
    reranker = build_reranker('residual_aware', D, horizon=4)
    z_q, z_k = torch.randn(3, 1, D), torch.randn(3, M, D)
    score = torch.randn(3, M)
    assert reranker(z_q, z_k, score, torch.randn(3, M, 4)).shape == (3, M)
    with pytest.raises(ValueError):
        reranker(z_q, z_k, score)
    with pytest.raises(ValueError):
        reranker(z_q, z_k, score, torch.randn(3, M + 1, 4))
    with pytest.raises(ValueError):
        reranker(torch.randn(3, M + 1, D), z_k, score, torch.randn(3, M, 4))


def test_past_pair_arm_has_no_residual_branch():
    past_pair = build_reranker('past_pair', D, horizon=4)
    assert past_pair.residual_proj is None
    assert past_pair.arm == 'past_pair'
    assert build_reranker('residual_aware', D, horizon=4).arm == 'residual_aware'


def test_choose_reproduces_the_retriever_order_when_score_is_the_retriever_score():
    """(2) reranking with the original score is a no-op on the selection."""
    torch.manual_seed(0)
    queries, channels, top_k = 5, 3, 4
    ids = torch.stack([torch.stack([torch.randperm(50)[:M] for _ in range(channels)])
                       for _ in range(queries)])
    valid = torch.ones(queries, channels, M, dtype=torch.bool)
    score = torch.randn(queries * channels, M)
    picked = choose(score, valid, ids, top_k)
    expected = ids.gather(
        2, score.view(queries, channels, M).topk(top_k, dim=-1).indices)
    assert torch.equal(picked, expected)
    # A shortlist already sorted by score keeps its own head.
    sorted_score = torch.sort(score, dim=-1, descending=True).values
    assert torch.equal(
        choose(sorted_score, valid, ids, top_k), ids[:, :, :top_k])


def test_choose_never_returns_an_invalid_candidate():
    torch.manual_seed(0)
    queries, channels, top_k = 4, 2, 3
    ids = torch.arange(queries * channels * M).view(queries, channels, M)
    valid = torch.ones(queries, channels, M, dtype=torch.bool)
    valid[..., :top_k] = False
    score = torch.zeros(queries * channels, M)
    score[:, :top_k] = 100.0                     # invalid slots score highest
    picked = choose(score, valid, ids, top_k)
    assert not bool((picked < ids[..., top_k:top_k + 1]).any())


def test_oracle_score_is_the_metric_ceiling():
    """Scoring by utility itself must give perfect alignment and recovery 1."""
    torch.manual_seed(0)
    utility = torch.randn(6, M)
    valid = torch.ones(6, M, dtype=torch.bool)
    metrics = ranking_metrics(utility.clone(), utility, valid, top_k=5)
    assert metrics['spearman'] == pytest.approx(1.0, abs=1e-6)
    assert metrics['ndcg_at_10'] == pytest.approx(1.0, abs=1e-6)
    assert metrics['gap_recovery_at_10'] == pytest.approx(1.0, abs=1e-4)


def test_ranking_metrics_ignore_invalid_candidates():
    torch.manual_seed(0)
    utility = torch.randn(4, M)
    valid = torch.ones(4, M, dtype=torch.bool)
    valid[:, -3:] = False
    utility[:, -3:] = 1e3                        # poisoned but masked out
    metrics = ranking_metrics(torch.randn(4, M), utility, valid, top_k=3)
    assert metrics['utility_at_10'] < 10.0
    assert metrics['oracle_utility_at_10'] < 10.0
