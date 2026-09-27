import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_c_horizon_clean02 import BLOCKS, HorizonAdapter, block_distance
from scripts.train_factorial_e2e01 import arm_score, individual_utility_memsafe


def test_uniform_aggregate_matches_manual():
    torch.manual_seed(0)
    memory_c = torch.randn(6, 720)
    offset_c = torch.zeros(2)
    query_future = torch.randn(2, 720)
    picks = torch.tensor([[0, 1, 2], [3, 4, 5]])
    y_sel = memory_c[picks] + offset_c.view(-1, 1, 1)
    agg = y_sel.mean(dim=1)
    manual = torch.stack([memory_c[[0, 1, 2]].mean(dim=0), memory_c[[3, 4, 5]].mean(dim=0)])
    assert torch.allclose(agg, manual, atol=1e-6)
    mse = ((agg - query_future) ** 2).mean(dim=-1)
    manual_mse = torch.stack([
        ((manual[0] - query_future[0]) ** 2).mean(),
        ((manual[1] - query_future[1]) ** 2).mean(),
    ])
    assert torch.allclose(mse, manual_mse, atol=1e-6)


def test_blocks_tile_720_exactly():
    covered = torch.zeros(720, dtype=torch.bool)
    for lo, hi in BLOCKS.values():
        assert not covered[lo:hi].any(), 'block overlap detected'
        covered[lo:hi] = True
    assert covered.all(), 'blocks do not fully tile [0:720]'


def test_block_distance_matches_individual_utility_memsafe_on_full_range():
    torch.manual_seed(0)
    memory_c = torch.randn(5, 720)
    offset_c = torch.randn(3)
    query_future = torch.randn(3, 720)
    d_full = block_distance(memory_c, offset_c, query_future, 0, 720, chunk_size=None)
    d_ref = -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=None)
    assert torch.allclose(d_full, d_ref, atol=1e-6)


def test_block_distance_slices_are_disjoint_and_differ_from_global():
    torch.manual_seed(1)
    memory_c = torch.randn(8, 720)
    offset_c = torch.randn(4)
    query_future = torch.randn(4, 720)
    d_g = block_distance(memory_c, offset_c, query_future, 0, 720, chunk_size=None)
    d_b1 = block_distance(memory_c, offset_c, query_future, 0, 96, chunk_size=None)
    d_b3 = block_distance(memory_c, offset_c, query_future, 336, 720, chunk_size=None)
    # generically different (random data) -- confirms slicing actually restricts the horizon
    assert not torch.allclose(d_g, d_b1)
    assert not torch.allclose(d_b1, d_b3)


def test_horizon_adapter_zero_init_equals_global_score():
    torch.manual_seed(0)
    d_model = 16
    adapter = HorizonAdapter(d_model, bottleneck=4)
    h_q = torch.randn(3, d_model)
    h_i = torch.randn(5, d_model)
    s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
    for bidx in range(3):
        z_q_b = adapter(h_q, bidx)
        z_i_b = adapter(h_i, bidx)
        s_b = arm_score(z_q_b, z_i_b, None)
        assert torch.allclose(s_b, s_g, atol=1e-6), f'block {bidx} score != global score at zero-init'


def test_horizon_adapter_applied_identically_to_query_and_candidate():
    """Symmetric-by-construction check: feeding the SAME vector as both
    'query' and 'candidate' through the adapter must give identical output
    (there is only one code path, no separate query-only branch)."""
    torch.manual_seed(2)
    d_model = 16
    adapter = HorizonAdapter(d_model, bottleneck=4)
    h = torch.randn(4, d_model)
    for bidx in range(3):
        out_as_query = adapter(h, bidx)
        out_as_candidate = adapter(h, bidx)
        assert torch.equal(out_as_query, out_as_candidate)


def test_horizon_adapter_output_is_unit_norm():
    torch.manual_seed(3)
    adapter = HorizonAdapter(16, bottleneck=4)
    h = torch.randn(5, 16) * 10  # deliberately not unit norm
    for bidx in range(3):
        z = adapter(h, bidx)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
