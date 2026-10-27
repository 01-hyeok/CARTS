"""Unit tests for TRACK-P100-RETRIEVAL-UTILIZATION-AUDIT01's pure math
-- the closed-form lambda solver (`diag_p100_oracle_lambda01`). CPU
-only, no GPU/checkpoint/cache dependency."""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_p100_oracle_lambda01 import closed_form_lambda


def test_closed_form_lambda_recovers_known_optimum_1d():
    # Y = B + 0.3*(R-B) exactly, plus independent noise on other rows --
    # the closed-form global optimum over all rows should land near 0.3.
    torch.manual_seed(0)
    n = 2000
    B = torch.randn(n, 5)
    R = B + torch.randn(n, 5)
    true_lambda = 0.3
    Y = B + true_lambda * (R - B)
    d, e = R - B, Y - B
    lam = closed_form_lambda(d, e, dims=(0, 1))
    assert abs(float(lam) - true_lambda) < 1e-5


def test_closed_form_lambda_matches_grid_search():
    torch.manual_seed(1)
    B = torch.randn(50, 7)
    R = torch.randn(50, 7)
    Y = torch.randn(50, 7)
    d, e = R - B, Y - B
    lam = float(closed_form_lambda(d, e, dims=(0, 1)).clamp(0.0, 1.0))

    def mse_at(l):
        fused = B + l * d
        return float(((fused - Y) ** 2).mean())

    best_grid = min((mse_at(l / 1000) for l in range(0, 1001)), default=None)
    assert abs(mse_at(lam) - best_grid) < 1e-4


def test_closed_form_lambda_per_row_reduces_only_given_dims():
    B = torch.randn(4, 3, 2)
    R = torch.randn(4, 3, 2)
    Y = torch.randn(4, 3, 2)
    d, e = R - B, Y - B
    lam_qc = closed_form_lambda(d, e, dims=(1,))  # per (query, channel)
    assert lam_qc.shape == (4, 2)
    lam_q = closed_form_lambda(d, e, dims=(1, 2))  # per query
    assert lam_q.shape == (4,)


def test_closed_form_lambda_zero_when_e_orthogonal_to_d():
    # if e is exactly anti-correlated structure such that sum(e*d)=0,
    # the optimal lambda is 0 regardless of d's own scale.
    d = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    e = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # sum(e*d) = 1-1+1-1 = 0
    lam = closed_form_lambda(d, e, dims=(0, 1))
    assert abs(float(lam)) < 1e-6
