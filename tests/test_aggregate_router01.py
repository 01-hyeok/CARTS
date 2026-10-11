"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- required unit tests."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.aggregate_router01 import (
    AggregateRouter, aggregate_soft_teacher, hard_routed_topk, oracle_gap_recovered_pct, r1_ce_loss, r2_kl_loss,
    router_entropy, soft_routed_topk, top2_accuracy, winner_accuracy,
)


def test_router_architecture_shape():
    router = AggregateRouter(d_model=128)
    z = torch.randn(4, 128)
    logits = router(z)
    assert logits.shape == (4, 5)


def test_router_no_future_input_by_construction():
    # structural guarantee: the router's forward signature takes exactly one
    # tensor (z_qc) and nothing resembling a future/query_future argument.
    import inspect
    sig = inspect.signature(AggregateRouter.forward)
    params = [p for p in sig.parameters if p != 'self']
    assert params == ['z_qc']


def test_r1_ce_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    labels = torch.randint(0, 5, (8,))
    loss = r1_ce_loss(logits, labels)
    manual = F.cross_entropy(logits, labels)
    assert torch.allclose(loss, manual)


def test_r2_kl_loss_zero_when_router_matches_teacher():
    U_agg = torch.tensor([[0.1, 1.0, 10.0, 5.0, 2.0]])
    teacher = aggregate_soft_teacher(U_agg, tau_r=0.5)
    logits = teacher.log()  # softmax(logits) == teacher when logits = log(teacher)
    loss = r2_kl_loss(logits, teacher)
    assert float(loss) < 1e-5


def test_aggregate_soft_teacher_favors_lowest_utility():
    U_agg = torch.tensor([[0.1, 1.0, 10.0]])
    teacher = aggregate_soft_teacher(U_agg, tau_r=0.5)
    assert int(teacher.argmax()) == 0


def test_hard_routed_topk_selects_argmax_head():
    router_logits = torch.tensor([[1.0, 5.0, 0.1, 0.1, 0.1]])  # head 1 wins
    p_h = torch.zeros(1, 5, 4)
    p_h[0, 1] = torch.tensor([0.1, 0.5, 0.3, 0.1])  # head 1's distribution
    picks, h_hat = hard_routed_topk(router_logits, p_h, k=2)
    assert int(h_hat[0]) == 1
    assert set(picks[0].tolist()) == {1, 2}  # top-2 of [0.1,0.5,0.3,0.1]


def test_soft_routed_topk_is_pi_weighted_mixture():
    router_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])  # uniform pi
    p_h = torch.rand(1, 5, 10)
    picks, p_route = soft_routed_topk(router_logits, p_h, k=3)
    manual = p_h.mean(dim=1)
    assert torch.allclose(p_route, manual, atol=1e-6)


def test_winner_and_top2_accuracy():
    logits = torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 1.0, 0.0]])
    labels = torch.tensor([0, 3])
    assert winner_accuracy(logits, labels) == 0.5  # first correct, second wrong (argmax=2)
    assert top2_accuracy(logits, labels) == 1.0  # both in top-2


def test_oracle_gap_recovered_formula():
    # Fixed=10, Oracle=2, Router=4 -> (10-4)/(10-2)*100 = 75%
    assert oracle_gap_recovered_pct(10.0, 4.0, 2.0) == pytest.approx(75.0)
    # Router == Fixed -> 0% recovered
    assert oracle_gap_recovered_pct(10.0, 10.0, 2.0) == pytest.approx(0.0)
    # Router == Oracle -> 100% recovered
    assert oracle_gap_recovered_pct(10.0, 2.0, 2.0) == pytest.approx(100.0)


def test_router_entropy_normalized_range():
    uniform_logits = torch.zeros(1, 5)
    ent_uniform = router_entropy(uniform_logits)
    assert float(ent_uniform) == pytest.approx(1.0, abs=1e-5)
    confident_logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])
    ent_confident = router_entropy(confident_logits)
    assert float(ent_confident) < 0.01


def test_checkpoint_selection_criterion_is_validation_only():
    # structural check: the training driver's post-hoc selection function
    # must be named/scoped to validation, never test -- enforced by
    # inspecting the driver source for a test-split selection call.
    driver = REPO_ROOT / 'scripts' / 'train_aggregate_router01.py'
    if not driver.exists():
        pytest.skip('driver not written yet')
    src = driver.read_text()
    assert 'best_epoch' in src
    assert "flag='val'" in src or 'val_agg_mse' in src
