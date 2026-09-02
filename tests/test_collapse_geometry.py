"""Geometry probe: does it read collapse where collapse exists, and only there?

The whole diagnosis turns on effective rank, so the axis it is computed over is
pinned here: taking it over the feature axis instead would report a small number
for a healthy encoder and point the investigation the wrong way.
"""

import pytest
import torch
import torch.nn.functional as F

from models.RelationStage1 import collapse_geometry


def test_full_rank_embeddings_report_a_high_effective_rank():
    torch.manual_seed(0)
    z = torch.randn(512, 32)
    out = collapse_geometry(torch.randn(16, 32), z)
    assert out['effective_rank'] > 20
    assert out['candidate_pairwise_cosine_mean'] < 0.2
    assert out['score_std'] > 0.05


def test_collapsed_embeddings_report_a_rank_near_one():
    """Collapse is the variance concentrating on one direction, not the points
    coinciding: the metric centres first, so a shared offset is removed and only
    the spread is measured."""
    torch.manual_seed(0)
    direction = F.normalize(torch.randn(1, 32), dim=-1)
    scale = torch.randn(512, 1)
    z = 5.0 * direction + scale * direction + 1e-5 * torch.randn(512, 32)
    out = collapse_geometry(5.0 * direction.repeat(16, 1), z)
    assert out['effective_rank'] < 2.0
    assert out['sv1_fraction'] > 0.99
    assert out['candidate_pairwise_cosine_mean'] > 0.9
    assert out['score_std'] < 1e-1


def test_a_shared_offset_alone_is_not_reported_as_collapse():
    """Centring removes the common component, so nearly-identical points with
    full-rank spread around them are not collapse and must not read as such."""
    torch.manual_seed(0)
    z = torch.randn(1, 32).repeat(512, 1) + 1e-4 * torch.randn(512, 32)
    out = collapse_geometry(torch.randn(16, 32), z)
    assert out['effective_rank'] > 20
    # The cosines are still all ~1, which is why the two must be read together.
    assert out['candidate_pairwise_cosine_mean'] > 0.99


def test_effective_rank_is_taken_over_the_candidate_axis():
    """[N, D] with N >> D: the rank can approach D, never N."""
    torch.manual_seed(0)
    out = collapse_geometry(torch.randn(8, 16), torch.randn(2048, 16))
    assert out['effective_rank_input_n'] == 2048
    assert out['effective_rank_input_d'] == 16
    assert out['effective_rank'] <= 16.0 + 1e-6


def test_shape_mistakes_are_rejected_rather_than_silently_measured():
    with pytest.raises(ValueError):
        collapse_geometry(torch.randn(8, 16), torch.randn(4, 8, 16))
    with pytest.raises(ValueError):
        collapse_geometry(torch.randn(8, 32), torch.randn(64, 16))


def test_rank_ordering_of_scores_is_monotone():
    torch.manual_seed(0)
    out = collapse_geometry(torch.randn(16, 32), torch.randn(512, 32))
    assert out['score_rank1_mean'] >= out['score_rank10_mean']
    assert out['score_rank10_mean'] >= out['score_rank11_mean']
    assert out['score_rank11_mean'] >= out['score_rank100_mean']
    assert out['rank10_minus_rank100'] >= 0
