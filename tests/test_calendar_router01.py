"""Unit tests for models/CalendarRouter.py -- spec section 20, Test 1/2."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.CalendarRouter import CalendarRouter, N_CALENDAR_FEATURES


def test_zero_init_gives_uniform_probability():
    router = CalendarRouter(n_channels=3, n_heads=5)
    calendar_feat = torch.randn(4, N_CALENDAR_FEATURES)
    for c in range(3):
        p = router(calendar_feat, c)
        assert p.shape == (4, 5)
        assert torch.allclose(p, torch.full_like(p, 0.2), atol=1e-6)


def test_router_input_is_only_calendar_feature():
    import inspect
    sig = inspect.signature(CalendarRouter.forward)
    names = list(sig.parameters)
    assert names == ['self', 'calendar_feat', 'channel']
    for forbidden in ('batch_x', 'z_q', 'query_embedding', 'series'):
        assert forbidden not in names


def test_param_count_is_tiny():
    router = CalendarRouter(n_channels=7, n_heads=5)
    # 7 channels * (6*5 weight + 5 bias) = 7 * 35 = 245
    assert router.param_count() == 7 * (N_CALENDAR_FEATURES * 5 + 5)


def test_router_gradient_flows_after_backward():
    # p.sum() is a degenerate loss here (softmax output always sums to 1,
    # a constant, so its gradient is trivially zero regardless of init --
    # that would test nothing). Use a non-trivial target instead, matching
    # how this router is actually trained (KL against a non-uniform teacher).
    router = CalendarRouter(n_channels=1, n_heads=5)
    calendar_feat = torch.randn(4, N_CALENDAR_FEATURES)
    p = router(calendar_feat, 0)
    target = torch.tensor([[0.5, 0.2, 0.1, 0.1, 0.1]] * 4)
    loss = F.kl_div(p.clamp_min(1e-8).log(), target, reduction='batchmean')
    loss.backward()
    assert router.linear[0].weight.grad is not None
    assert router.linear[0].weight.grad.abs().sum() > 0
