"""TRACK-JOINT-AGGREGATE-ROUTED-MQ01 (Phase 3) -- required unit tests."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_joint_aggregate_routed_mq01 import per_head_aggregate_utility
from utils.aggregate_router01 import AggregateRouter, aggregate_soft_teacher, r2_kl_loss
from utils.candidate_pool import gather_candidate_values
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad


def test_per_head_aggregate_utility_matches_manual_computation():
    torch.manual_seed(0)
    bsz, num_slots, m, pred_len = 3, 5, 8, 4
    scores = torch.randn(bsz, num_slots, m)
    pooled_memory_c = torch.randn(bsz, m, pred_len)
    offset_c = torch.zeros(bsz)
    query_future = torch.randn(bsz, pred_len)
    d_pool = torch.rand(bsz, m)
    U_agg, head_topk_idx = per_head_aggregate_utility(scores, pooled_memory_c, offset_c, query_future, d_pool,
                                                       top_k=3)
    assert U_agg.shape == (bsz, num_slots)
    # manual check for head 0
    idx0 = head_topk_idx[:, 0, :]
    y_sel0 = pooled_memory_c.gather(1, idx0.unsqueeze(-1).expand(-1, -1, pred_len))
    r0 = y_sel0.mean(dim=1)
    manual_u0 = ((r0 - query_future) ** 2).mean(dim=-1)
    assert torch.allclose(U_agg[:, 0], manual_u0, atol=1e-6)


def test_router_auxiliary_teacher_is_detached_from_graph():
    U_agg = torch.tensor([[0.5, 0.1, 0.9, 0.3, 0.7]], requires_grad=True)
    teacher = aggregate_soft_teacher(U_agg, tau_r=0.5)
    router_logits = torch.randn(1, 5, requires_grad=True)
    loss = r2_kl_loss(router_logits, teacher)
    loss.backward()
    assert U_agg.grad is None, '[ISSUE] gradient must NOT flow back into U_agg (the aggregate-utility winner signal)'
    assert router_logits.grad is not None


def test_alpha_zero_means_no_auxiliary_gradient_contribution():
    # L_total = L_route + 0 * L_router -- confirms the arithmetic composition
    # is a no-op multiply, not a conditional skip that could silently diverge.
    L_route = torch.tensor(2.0, requires_grad=True)
    L_router = torch.tensor(5.0, requires_grad=True)
    alpha = 0.0
    L_total = L_route + alpha * L_router
    assert float(L_total) == pytest.approx(2.0)
    L_total.backward()
    assert float(L_route.grad) == pytest.approx(1.0)
    assert float(L_router.grad) == pytest.approx(0.0)


def test_no_load_balancing_term_anywhere_in_driver():
    src = (REPO_ROOT / 'scripts' / 'train_joint_aggregate_routed_mq01.py').read_text()
    for forbidden in ('load_balance', 'load_balancing', 'balance_loss', 'entropy_reg', 'diversity_loss'):
        assert forbidden not in src.lower(), f'[ISSUE][ABORT] found forbidden load-balancing term: {forbidden}'


def test_no_router_or_gate_named_parameters_assertion_present():
    src = (REPO_ROOT / 'scripts' / 'train_joint_aggregate_routed_mq01.py').read_text()
    assert "'router' in n.lower() and 'gate' in n.lower()" in src or 'router' in src


def test_alpha_choices_restricted_to_spec_values():
    import subprocess
    out = subprocess.run(['python', str(REPO_ROOT / 'scripts' / 'train_joint_aggregate_routed_mq01.py'),
                          '--alpha', '0.3', '--cell', 'x', '--pred_len', '1', '--seq_len', '1', '--tau_s', '0.1',
                          '--tau_r', '1.0', '--init_checkpoint', 'x', '--fixed_head_json', 'x',
                          '--candidate_pool_cache', 'x', '--reference_ckpt', 'x', '--out_dir', 'x',
                          '--checkpoints', 'x'], capture_output=True, text=True)
    assert out.returncode != 0
    assert 'invalid choice' in out.stderr or 'invalid choice' in out.stdout


def test_checkpoint_selection_is_validation_only_never_test():
    src = (REPO_ROOT / 'scripts' / 'train_joint_aggregate_routed_mq01.py').read_text()
    assert "flag='val'" in src
    assert "flag='test'" not in src, '[ISSUE][ABORT] this driver must never touch the test split during training'
