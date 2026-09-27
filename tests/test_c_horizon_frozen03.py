import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_c_horizon_frozen03 import candidate_value, cluster_bootstrap_paired, rows_for_batch


def test_candidate_value_is_query_independent_delta():
    """memory_c is a pure bank-level constant: memory_y[...,c] minus the
    bank's own memory_x_last[...,c] -- confirms it never needs batch_x."""
    torch.manual_seed(0)
    n, h, ch = 5, 720, 3
    memory_y = torch.randn(n, h, ch)
    memory_x_last = torch.randn(n, ch)
    out = candidate_value(memory_y, memory_x_last, 1)
    expected = memory_y[:, :, 1] - memory_x_last[:, 1].unsqueeze(-1)
    assert torch.allclose(out, expected)
    assert out.shape == (n, h)


def test_rows_for_batch_lookup():
    cache = {'start_to_row': {10: 0, 20: 1, 30: 2}}
    batch_start_idx = torch.tensor([30, 10, 20])
    rows = rows_for_batch(cache, batch_start_idx)
    assert rows.tolist() == [2, 0, 1]


def test_cluster_bootstrap_paired_zero_when_identical():
    a0 = {0: 1.0, 1: 2.0, 2: 3.0}
    arm = {0: 1.0, 1: 2.0, 2: 3.0}
    res = cluster_bootstrap_paired(a0, arm, n_reps=200, seed=0)
    assert res['mean_delta'] == 0.0
    assert res['ci_low'] == 0.0 and res['ci_high'] == 0.0
    assert res['ci_excludes_zero_positive'] is False


def test_cluster_bootstrap_paired_positive_when_arm_uniformly_better():
    a0 = {i: 2.0 for i in range(50)}
    arm = {i: 1.0 for i in range(50)}  # arm has LOWER mse -> a0-arm = +1 everywhere
    res = cluster_bootstrap_paired(a0, arm, n_reps=500, seed=0)
    assert res['mean_delta'] == 1.0
    assert res['ci_excludes_zero_positive'] is True
    assert res['frac_improved'] == 1.0
