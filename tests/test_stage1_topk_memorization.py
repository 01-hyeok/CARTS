"""Stage-1 Oracle Top-K memorization sanity check.

These cover the two pieces the diagnostic adds on top of the existing
tiny-overfit mode: fully differentiable candidate encoding, and the
`student_`-prefixed retrieval metric names the teacher-free objectives
otherwise never publish.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import Model, student_retrieval_metric_aliases
from tests.test_stage1_new_losses import _model_batch, _model_config


def _coverage_model(relation_input_space='absolute'):
    config = _model_config(loss_mode='topk_coverage')
    config.relation_input_space = relation_input_space
    config.stage1_coverage_top_k = 2
    return Model(config)


def _candidate_x(num_candidates=3):
    torch.manual_seed(11)
    return torch.randn(num_candidates, 4, 1)


def test_differentiable_keys_match_a_fresh_key_bank():
    """The in-graph key path must score exactly like a bank rebuilt right now."""
    model = _coverage_model().eval()
    query_x, query_y, cand_mask, memory_y, _ = _model_batch()
    candidate_x = _candidate_x()

    fresh_bank = model.build_embedding_bank(candidate_x.numpy(), torch.device('cpu'))
    with torch.no_grad():
        bank_loss, bank_metrics = model(
            query_x, query_y, cand_mask, memory_y, fresh_bank,
            compute_detailed_metrics=True,
        )
        diff_loss, diff_metrics = model(
            query_x, query_y, cand_mask, memory_y, key_bank=None,
            compute_detailed_metrics=True,
            candidate_x=candidate_x,
            differentiable_keys=True,
        )

    assert torch.allclose(bank_loss, diff_loss, atol=1e-6)
    assert torch.allclose(
        bank_metrics['student_oracle_recall_at_10'],
        diff_metrics['student_oracle_recall_at_10'],
    )


def test_differentiable_keys_send_gradient_through_the_candidate_side():
    """A stale bank is a constant; the diagnostic path must not be."""
    model = _coverage_model()
    query_x, query_y, cand_mask, memory_y, _ = _model_batch()
    candidate_x = _candidate_x().requires_grad_(True)

    loss, _ = model(
        query_x, query_y, cand_mask, memory_y, key_bank=None,
        candidate_x=candidate_x,
        differentiable_keys=True,
    )
    loss.backward()

    assert candidate_x.grad is not None
    assert candidate_x.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.encoder.parameters()
    )


def test_differentiable_keys_reject_missing_or_misshaped_candidates():
    model = _coverage_model()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    for kwargs in (
        {'candidate_x': None},
        {'candidate_x': _candidate_x(num_candidates=5)},
        {'candidate_x': _candidate_x(), 'direct_retrieval': True},
    ):
        try:
            model(
                query_x, query_y, cand_mask, memory_y, key_bank,
                differentiable_keys=True,
                **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for {sorted(kwargs)}')


def test_key_bank_path_is_unchanged_when_the_flag_is_off():
    model = _coverage_model().eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    with torch.no_grad():
        loss, metrics = model(query_x, query_y, cand_mask, memory_y, key_bank)
    assert torch.isfinite(loss)
    assert metrics['topk_coverage_loss'] > 0


def test_student_aliases_restate_the_retrieval_metrics_without_recomputing():
    source = {
        'oracle_recall_at_10': torch.tensor(0.75),
        'retrieval_regret_at_10': torch.tensor(0.5),
        'retrieved_future_mse_at_10': torch.tensor(1.25),
        'oracle_topk_probability_mass_at_10': torch.tensor(0.4),
        'ndcg_at_5': torch.tensor(0.9),
        'spearman_score_vs_negative_mse': torch.tensor(-0.2),
    }
    aliases = student_retrieval_metric_aliases(source)

    assert aliases['student_oracle_recall_at_10'] is source['oracle_recall_at_10']
    assert aliases['student_retrieval_regret_at_10'] is source['retrieval_regret_at_10']
    assert (
        aliases['student_oracle_topk_probability_mass_at_10']
        is source['oracle_topk_probability_mass_at_10']
    )
    assert aliases['student_ndcg_at_5'] is source['ndcg_at_5']
    # Keys the caller never produced must not be invented.
    assert 'student_oracle_recall_at_1' not in aliases
    assert 'student_ndcg_at_10' not in aliases


def test_coverage_forward_publishes_every_required_diagnostic_metric():
    model = _coverage_model().eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    with torch.no_grad():
        _, metrics = model(
            query_x, query_y, cand_mask, memory_y, key_bank,
            compute_detailed_metrics=True,
        )

    for key in (
        'topk_coverage_loss',
        'student_oracle_recall_at_1',
        'student_oracle_recall_at_5',
        'student_oracle_recall_at_10',
        'coverage_oracle_student_overlap',
        'student_retrieval_regret_at_10',
        'student_retrieved_future_mse_at_10',
        'oracle_future_mse_at_10',
        'student_oracle_topk_probability_mass_at_10',
    ):
        assert key in metrics, key
        assert torch.isfinite(torch.as_tensor(metrics[key])), key
