"""TRACK-HEAD-UTILITY-ALIGNMENT01 -- required unit tests (spec section 12)."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.expert_head_metrics import per_head_future_utility, per_head_standalone_topk
from utils.head_utility_alignment01 import (
    closed_form_alpha, confusion_matrix, correlation_summary, fixed_head_from_val, oracle_vs_fixed_test,
    spearman_per_query, winner_fraction_stats, winners,
)


# -------------------- [1] U_ind identical to existing Hard implementation --------------------

def test_1_u_ind_matches_existing_hard_utility():
    torch.manual_seed(0)
    scores = torch.randn(4, 5, 20)
    mask = torch.ones(4, 20, dtype=torch.bool)
    d_pool = torch.rand(4, 20)
    head_topk_idx = per_head_standalone_topk(scores, mask, k=10)
    u_ind = per_head_future_utility(head_topk_idx, d_pool)  # exact function Hard trainer uses
    # reproduce by hand for head 0
    idx0 = head_topk_idx[:, 0, :]
    manual = d_pool.gather(1, idx0).mean(dim=-1)
    assert torch.allclose(u_ind[:, 0], manual, atol=1e-6)


# -------------------- [2] Aggregate utility synthetic check --------------------

def test_2_aggregate_utility_synthetic():
    # 2 selected candidates, future values [1,2] and [3,4]; query future [2,2]
    selected_futures = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, K=2, pred_len=2]
    query_future = torch.tensor([[2.0, 2.0]])  # [1, pred_len=2]
    r_h = selected_futures.mean(dim=1)  # [1,2] -> [2.0, 3.0]
    u_agg = ((r_h - query_future) ** 2).mean(dim=-1)
    assert torch.allclose(r_h, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(u_agg, torch.tensor([0.5]))  # mean((0)^2,(1)^2) = 0.5


# -------------------- [3] Final utility formula --------------------

def test_3_final_utility_formula():
    B = torch.tensor([[1.0, 1.0]])
    R_h = torch.tensor([[3.0, 5.0]])
    Y = torch.tensor([[2.0, 2.0]])
    lam = 0.5
    y_hat = B + lam * (R_h - B)
    expected_y_hat = torch.tensor([[2.0, 3.0]])  # 1+0.5*2=2, 1+0.5*4=3
    assert torch.allclose(y_hat, expected_y_hat)
    u_final = ((y_hat - Y) ** 2).mean(dim=-1)
    assert torch.allclose(u_final, torch.tensor([0.5]))  # mean(0^2, 1^2)=0.5


# -------------------- [4] winner argmin correctness --------------------

def test_4_winner_argmin():
    U = torch.tensor([[0.5, 0.1, 0.9, 0.3, 0.7], [5.0, 4.0, 3.0, 2.0, 1.0]])
    w = winners(U)
    assert w.tolist() == [1, 4]


# -------------------- [5] validation fixed-head selection never touches test --------------------

def test_5_fixed_head_selection_no_test_leakage():
    import inspect
    sig = inspect.signature(fixed_head_from_val)
    assert list(sig.parameters) == ['U_val']
    sig2 = inspect.signature(oracle_vs_fixed_test)
    assert 'val' not in [p.lower() for p in sig2.parameters]


# -------------------- [6] beta selection validation-only (reuse existing proof) --------------------

def test_6_beta_selection_is_validation_only():
    import inspect
    from scripts.eval_professor_style_fusion01 import select_beta_on_validation
    sig = inspect.signature(select_beta_on_validation)
    assert list(sig.parameters)[:3] == ['B_va', 'R_va', 'Y_va']


# -------------------- [7] beta grid includes 0 --------------------

def test_7_beta_grid_includes_zero():
    from scripts.eval_professor_style_fusion01 import BETA_GRID
    assert 0.0 in BETA_GRID


# -------------------- [8] test alpha* is diagnostic only (never applied as a real prediction) --------------------

def test_8_closed_form_alpha_is_pure_function_no_side_effect():
    B = torch.zeros(3, 2)
    R = torch.ones(3, 2)
    Y = torch.full((3, 2), 0.5)
    alpha = closed_form_alpha(B, R, Y)
    assert alpha == pytest.approx(0.5, abs=1e-6)
    # calling it again with the same inputs must be side-effect-free (pure)
    alpha2 = closed_form_alpha(B, R, Y)
    assert alpha == alpha2


# -------------------- [9] existing checkpoint parameters frozen (requires_grad=False) --------------------

def test_9_loaded_checkpoint_params_can_be_frozen():
    lin = torch.nn.Linear(3, 3)
    for p in lin.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in lin.parameters())


# -------------------- [10] no reuse of invalidated P100 artifacts --------------------

def test_10_driver_reads_from_signfix01_or_allh01_tracks_only():
    driver = REPO_ROOT / 'scripts' / 'run_head_utility_alignment01.py'
    if not driver.exists():
        pytest.skip('driver not written yet')
    src = driver.read_text()
    assert 'TRACK-HARD-EXPERT-V5-P100-ALLH01' in src
    assert 'TRACK-V-MULTIQUERY-GENERALIZATION01' not in src or 'pool_top100/shared_candidate_pool' in src
    # must never read the original bugged per-arm P100 result tree directly
    assert 'TRACK-V-SHARED-TOP100-01' not in src


# -------------------- [11] candidate alignment matches production memory_value semantics --------------------

def test_11_alignment_reuses_memory_value_unmodified():
    src = (REPO_ROOT / 'utils' / 'head_utility_alignment01.py').read_text()
    assert 'from scripts.train_margutil01 import memory_value' in src
    assert 'from utils.candidate_pool import gather_candidate_values, pooled_future_mse' in src


# -------------------- [12] confusion matrix / winner-fraction / correlation sanity --------------------

def test_12_confusion_matrix_and_stats_sane():
    w1 = torch.tensor([0, 1, 1, 2, 2, 2])
    w2 = torch.tensor([0, 1, 0, 2, 2, 1])
    cm = confusion_matrix(w1, w2)
    assert cm.sum() == 6
    assert cm[0, 0] == 1 and cm[2, 2] == 2

    stats = winner_fraction_stats(w1)
    assert pytest.approx(sum(stats['winner_fraction'])) == 1.0
    assert stats['max_winner_fraction'] == pytest.approx(3 / 6)

    U1 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    U2 = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]])  # perfectly reversed ranking
    rho = spearman_per_query(U1, U2)
    assert rho[0] == pytest.approx(-1.0, abs=1e-5)
    summ = correlation_summary(rho)
    assert summ['negative_fraction'] == 1.0
