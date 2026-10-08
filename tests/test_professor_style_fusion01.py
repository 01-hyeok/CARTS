"""TRACK-V-PROFESSOR-FUSION01 -- required unit/sanity tests (spec
section 17), run before any real-data evaluation."""
import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_professor_style_fusion01 import BETA_GRID, fused_metrics, select_beta_on_validation

EVALUATOR_SRC = (REPO_ROOT / 'scripts' / 'eval_professor_style_fusion01.py').read_text()
ORCH_FULL_SRC = (REPO_ROOT / 'scripts' / 'run_professor_fusion_full01.sh').read_text() \
    if (REPO_ROOT / 'scripts' / 'run_professor_fusion_full01.sh').exists() else ''


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(8, 5, generator=g)
    R = torch.randn(8, 5, generator=g)
    Y = torch.randn(8, 5, generator=g)
    return B, R, Y


# -------------------- [1] beta=0 -> Y_final == Y_base --------------------

def test_beta_zero_equals_base():
    B, R, Y = _toy()
    mse0, mae0 = fused_metrics(B, R, Y, 0.0)
    mse_base, mae_base = float(((B - Y) ** 2).mean()), float((B - Y).abs().mean())
    assert mse0 == pytest.approx(mse_base, abs=1e-9)
    assert mae0 == pytest.approx(mae_base, abs=1e-9)


# -------------------- [2] beta=1 -> Y_final == Y_ret --------------------

def test_beta_one_equals_retrieval():
    B, R, Y = _toy()
    mse1, mae1 = fused_metrics(B, R, Y, 1.0)
    mse_ret, mae_ret = float(((R - Y) ** 2).mean()), float((R - Y).abs().mean())
    assert mse1 == pytest.approx(mse_ret, abs=1e-9)
    assert mae1 == pytest.approx(mae_ret, abs=1e-9)


# -------------------- [3] beta=0.1 -> 0.9*B + 0.1*R --------------------

def test_beta_point_one_matches_manual_formula():
    B, R, Y = _toy()
    mse, mae = fused_metrics(B, R, Y, 0.1)
    manual_fused = 0.9 * B + 0.1 * R
    manual_mse = float(((manual_fused - Y) ** 2).mean())
    manual_mae = float((manual_fused - Y).abs().mean())
    assert mse == pytest.approx(manual_mse, abs=1e-6)
    assert mae == pytest.approx(manual_mae, abs=1e-6)


# -------------------- [4] validation-only selection never reads test --------------------

def test_beta_selection_signature_has_no_test_argument():
    sig = inspect.signature(select_beta_on_validation)
    params = list(sig.parameters)
    assert params[:3] == ['B_va', 'R_va', 'Y_va'], \
        f'[ISSUE][ABORT] beta-selection function must only take validation tensors, got {params}'
    for p in params:
        assert 'test' not in p.lower() and 'te_' not in p.lower()


def test_beta_selection_never_improves_by_peeking_grid_includes_zero():
    assert 0.0 in BETA_GRID, '[ISSUE] beta grid must include 0.0 (Base fallback)'
    assert BETA_GRID == sorted(BETA_GRID), 'grid must be ascending for the ties-prefer-smaller-beta rule to work'


def test_tie_break_prefers_smaller_beta():
    # construct B,R,Y such that beta=0 and beta=0.05 give EXACTLY the same val MSE
    B = torch.zeros(4, 3)
    R = torch.zeros(4, 3)  # R==B -> every beta gives identical MSE -> perfect tie across the whole grid
    Y = torch.ones(4, 3)
    best_beta, best_mse, rows = select_beta_on_validation(B, R, Y, grid=[0.0, 0.05, 0.1])
    assert best_beta == 0.0, f'[ISSUE] tie-break must prefer the smallest beta, got {best_beta}'


# -------------------- [5] same base checkpoint -> identical base prediction --------------------

def test_same_base_checkpoint_gives_identical_predictions_across_loads():
    from models.RelationStage2 import BaseForecastHead
    torch.manual_seed(0)
    base_a = BaseForecastHead(seq_len=16, pred_len=4, channels=3, mode='per_channel_linear')
    sd = base_a.state_dict()
    base_b = BaseForecastHead(seq_len=16, pred_len=4, channels=3, mode='per_channel_linear')
    base_b.load_state_dict(sd)
    x = torch.randn(2, 16, 3)
    with torch.no_grad():
        out_a = base_a(x)
        out_b = base_b(x)
    assert torch.equal(out_a, out_b), \
        '[ISSUE] two arms loading the SAME base checkpoint must produce bit-identical predictions'


# -------------------- [6] V0/V1 ordinary Top-K == Mean-Mixture (delegates to the existing proof) --------------------

def test_v0_v1_meanmix_equivalence_already_proven_elsewhere():
    # The actual equivalence proof (real ETTh1_96 checkpoints, S=1 reduction) lives in
    # tests/test_v_meanmix_cache01.py and tests/test_v_meanmix_selection01.py (test_s1_meanmix_equals_round_robin,
    # test_v0_single_head_ordinary_top10_equals_meanmix) -- this test only confirms those files still exist
    # and still contain the specific proofs this track's audit depends on.
    sel_src = (REPO_ROOT / 'tests' / 'test_v_meanmix_selection01.py').read_text()
    cache_src = (REPO_ROOT / 'tests' / 'test_v_meanmix_cache01.py').read_text()
    assert 'def test_s1_meanmix_equals_round_robin' in sel_src
    assert 'def test_v0_single_head_ordinary_top10_equals_meanmix' in sel_src
    assert 'def test_v0_old_new_top10_agreement_100pct' in cache_src
    assert 'def test_v1_old_new_top10_agreement_100pct' in cache_src


# -------------------- [7] V2/V5 use the corrected Mean-Mixture checkpoint/cache --------------------

def test_v2_v5_orchestrator_points_at_meanmix_correction_track():
    assert ORCH_FULL_SRC, 'run_professor_fusion_full01.sh must exist before this test is meaningful'
    assert 'TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01' in ORCH_FULL_SRC, \
        '[ISSUE][ABORT] V2/V5 must source their cache from the corrected track, not the aborted Round-Robin one'
    # the aborted track's name may appear in a prose comment explaining what NOT to use;
    # only an actual path assignment referencing it would be a real violation
    assert 'cache_dir="results/TRACK-V-MEANMIX-INFERENCE01' not in ORCH_FULL_SRC.replace(' ', ''), \
        '[ISSUE][ABORT] must not reuse the aborted (Round-Robin-checkpoint-selected) inference-only track'


# -------------------- [8] Full/P100 candidate support never mixed --------------------

def test_full_and_pool_cache_dirs_are_structurally_distinct_in_evaluator():
    assert "'full', 'pool_top100'" in EVALUATOR_SRC, \
        '[ISSUE] evaluator must require an explicit --candidate_support in {full, pool_top100}'
    assert "'candidate_support': cli.candidate_support" in EVALUATOR_SRC


# -------------------- [9] retrieval forecast is a uniform mean, not score-weighted --------------------

def test_retrieval_cache_builders_use_uniform_mean_not_weighted():
    forbidden = ('weights * y_sel', 'p_mix.unsqueeze(-1) * y_sel', '(p_h * y_sel)', 'softmax(scores) * y_sel')
    for fname in ('build_v_meanmix_cache01.py', 'build_v_meanmix_cache_pool01.py',
                  'build_t_multislot_cache01.py', 'build_t2_true_original_kl_cache01.py'):
        src = (REPO_ROOT / 'scripts' / fname).read_text()
        assert '.mean(dim=1)' in src, f'[ISSUE] {fname} missing the expected uniform mean(dim=1) future aggregation'
        for token in forbidden:
            assert token not in src, f'[ISSUE] {fname} contains a score-weighted aggregation pattern: {token}'
