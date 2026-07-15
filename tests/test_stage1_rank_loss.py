import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import future_aware_topk_ranking_loss


def test_future_aware_topk_ranking_loss_synthetic_case():
    # Candidates: A, B, C, D, E.
    future_mse = torch.tensor([[0.10, 0.12, 0.15, 0.55, 0.60]], requires_grad=True)
    scores = torch.tensor([[0.90, 0.58, 0.55, 0.85, 0.80]], requires_grad=True)
    valid_mask = torch.ones_like(future_mse, dtype=torch.bool)

    loss, metrics, debug = future_aware_topk_ranking_loss(
        scores,
        future_mse,
        valid_mask,
        top_k=3,
        rank_margin=0.1,
        min_mse_gap=0.0,
        return_debug=True,
    )

    assert debug["teacher_topk_idx"][0, 0].tolist() == [0, 1, 2]
    assert debug["student_topk_idx"][0, 0].tolist() == [0, 3, 4]
    assert debug["missed_positive_mask"][0, 0].tolist() == [False, True, True]
    assert debug["hard_negative_mask"][0, 0].tolist() == [False, True, True]
    assert int(debug["pair_mask"].sum().item()) == 4
    assert metrics["rank_valid_pair_count"].item() == 4

    higher_pos_scores = torch.tensor([[0.90, 0.88, 0.86, 0.85, 0.80]])
    lower_neg_scores = torch.tensor([[0.90, 0.58, 0.55, 0.45, 0.40]])
    higher_pos_loss, _ = future_aware_topk_ranking_loss(
        higher_pos_scores, future_mse.detach(), valid_mask, top_k=3, rank_margin=0.1
    )
    lower_neg_loss, _ = future_aware_topk_ranking_loss(
        lower_neg_scores, future_mse.detach(), valid_mask, top_k=3, rank_margin=0.1
    )
    assert higher_pos_loss.item() < loss.item()
    assert lower_neg_loss.item() < loss.item()

    loss.backward()
    assert scores.grad is not None
    assert future_mse.grad is None


def test_future_aware_topk_ranking_loss_edge_cases():
    future_mse = torch.tensor([[0.10, 0.12, 0.15, 0.55, 0.60]])
    valid_mask = torch.ones_like(future_mse, dtype=torch.bool)

    identical_scores = torch.tensor([[0.90, 0.85, 0.80, 0.58, 0.55]], requires_grad=True)
    identical_loss, identical_metrics = future_aware_topk_ranking_loss(
        identical_scores, future_mse, valid_mask, top_k=3, rank_margin=0.1
    )
    assert identical_loss.item() == 0.0
    assert identical_metrics["rank_valid_pair_count"].item() == 0.0
    identical_loss.backward()
    assert identical_scores.grad is not None

    scores = torch.tensor([[0.90, 0.58, 0.55, 0.85, 0.80]], requires_grad=True)
    invalid_mask = torch.tensor([[True, True, True, False, True]])
    _, _, invalid_debug = future_aware_topk_ranking_loss(
        scores, future_mse, invalid_mask, top_k=3, return_debug=True
    )
    assert 3 not in invalid_debug["teacher_topk_idx"][0, 0].tolist()
    assert 3 not in invalid_debug["student_topk_idx"][0, 0].tolist()

    no_gap_loss, no_gap_metrics = future_aware_topk_ranking_loss(
        scores, future_mse, valid_mask, top_k=3, min_mse_gap=1.0
    )
    assert no_gap_loss.item() == 0.0
    assert no_gap_metrics["rank_valid_pair_count"].item() == 0.0

    short_scores = torch.tensor([[0.5, 0.4]], requires_grad=True)
    short_mse = torch.tensor([[0.1, 0.2]])
    short_mask = torch.ones_like(short_mse, dtype=torch.bool)
    short_loss, short_metrics = future_aware_topk_ranking_loss(
        short_scores, short_mse, short_mask, top_k=10
    )
    assert torch.isfinite(short_loss)
    assert short_metrics["rank_valid_pair_count"].item() == 0.0
