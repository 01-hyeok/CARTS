import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_c_horizon_clean04_expertwise import BlockAdapter
from scripts.train_factorial_e2e01 import arm_score


def test_block_adapter_zero_init_equals_global_score():
    torch.manual_seed(0)
    adapter = BlockAdapter(16, bottleneck=4)
    h_q = torch.randn(3, 16)
    h_i = torch.randn(5, 16)
    s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
    s_a = arm_score(adapter(h_q), adapter(h_i), None)
    assert torch.allclose(s_a, s_g, atol=1e-6)


def test_block_adapter_output_unit_norm():
    adapter = BlockAdapter(16, bottleneck=4)
    h = torch.randn(5, 16) * 10
    z = adapter(h)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_alpha_residual_endpoints():
    """alpha=0 -> pure global score; alpha=1 -> pure adapter score (spec
    section 15's residual-interpolation identity, checked directly)."""
    torch.manual_seed(1)
    adapter = BlockAdapter(16, bottleneck=4)
    h_q = torch.randn(3, 16)
    h_i = torch.randn(5, 16)
    s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
    s_a = arm_score(adapter(h_q), adapter(h_i), None)
    for alpha, expected in [(0.0, s_g), (1.0, s_a)]:
        s_alpha = (1 - alpha) * s_g + alpha * s_a
        assert torch.allclose(s_alpha, expected, atol=1e-6)
