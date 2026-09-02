"""Holding the ranking the frozen encoder already produced.

Training the scorer on a hundred candidates moves the scores of all eight
thousand, and the arm without an anchor improved local ordering while Recall@10
fell by two thirds. These pin the properties that make the anchor readable: it
must be exactly zero where the scorer has not moved, must grow with departure,
and must report retention rather than only a divergence value.
"""

import pytest
import torch

from models.RelationStage1 import global_anchor_kl


def _mask(n, m):
    return torch.ones(n, m, dtype=torch.bool)


def test_identical_scores_cost_nothing():
    """Identity initialisation makes the two score sets equal at step 0."""
    torch.manual_seed(0)
    s = torch.randn(4, 200)
    loss, m = global_anchor_kl(s, s.clone(), _mask(4, 200), tau=0.1)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(m['baseline_top10_retention']) == 1.0
    assert float(m['baseline_top100_retention']) == 1.0
    assert float(m['baseline_vs_new_score_spearman']) == pytest.approx(1.0, abs=1e-5)


def test_cost_grows_as_the_ranking_departs():
    torch.manual_seed(0)
    base = torch.randn(4, 200)
    near = base + 0.01 * torch.randn(4, 200)
    far = torch.randn(4, 200)
    small, _ = global_anchor_kl(base, near, _mask(4, 200), tau=0.1)
    large, m_far = global_anchor_kl(base, far, _mask(4, 200), tau=0.1)
    assert 0.0 < float(small) < float(large)
    assert float(m_far['baseline_top10_retention']) < 1.0


def test_it_charges_for_abandoning_mass_the_baseline_placed():
    """Forward KL: dropping a candidate the baseline ranked top is expensive,
    while the reverse direction would let the new scores concentrate freely."""
    base = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    drop = torch.tensor([[-5.0, 0.0, 0.0, 0.0]])     # abandons the baseline's top
    spread = torch.tensor([[5.0, 4.9, 0.0, 0.0]])    # keeps it, adds another
    dropped, _ = global_anchor_kl(base, drop, _mask(1, 4), tau=1.0)
    kept, _ = global_anchor_kl(base, spread, _mask(1, 4), tau=1.0)
    assert float(dropped) > float(kept)


def test_masked_candidates_take_no_part():
    torch.manual_seed(0)
    base, new = torch.randn(2, 50), torch.randn(2, 50)
    mask = _mask(2, 50)
    mask[:, 25:] = False
    loss, _ = global_anchor_kl(base, new, mask, tau=0.1)
    # Changing only the masked half must not change the divergence.
    new2 = new.clone()
    new2[:, 25:] = torch.randn(2, 25) * 10
    loss2, _ = global_anchor_kl(base, new2, mask, tau=0.1)
    assert float(loss) == pytest.approx(float(loss2), abs=1e-6)


def test_gradient_flows_to_the_new_scores_only():
    torch.manual_seed(0)
    base = torch.randn(2, 100, requires_grad=True)
    new = torch.randn(2, 100, requires_grad=True)
    loss, _ = global_anchor_kl(base, new, _mask(2, 100), tau=0.1)
    g_new, = torch.autograd.grad(loss, new, retain_graph=True)
    assert torch.isfinite(g_new).all() and g_new.abs().sum() > 0
    # The baseline is a fixed target and must not be pulled toward the scorer.
    g_base, = torch.autograd.grad(loss, base, allow_unused=True)
    assert g_base is None or float(g_base.abs().sum()) == 0.0


def test_retention_counts_membership_not_order():
    base = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    reordered = torch.tensor([[2.0, 3.0, 1.0, 0.0]])   # same top-2, swapped
    _, m = global_anchor_kl(base, reordered, _mask(1, 4), tau=1.0, top_k=2, pool_end=4)
    assert float(m['baseline_top10_retention']) == 1.0
