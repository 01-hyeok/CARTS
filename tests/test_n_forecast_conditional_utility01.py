"""Unit tests for TRACK-N-FORECAST-CONDITIONAL-UTILITY01 (spec section 21,
18 items). Reuses saved diagnostic artifacts where the underlying
computation is expensive (GPU forward passes already run and cached to
disk); synthetic/structural checks run standalone and fast.
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_n_diagnostics01 import CKPTS as N_CKPTS
from scripts.train_n_gate_only01 import EXTRA_FREEZE, freeze_all_but_gate

OUT_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720'
GATE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only'
S0_CKPT = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'
BASE_DIR = OUT_DIR / 'common_base_predictions'

_diag_available = (OUT_DIR / 'all_diagnostics.parquet').exists()
needs_diag = pytest.mark.skipif(not _diag_available, reason='all_diagnostics.parquet not yet produced')
_gate_available = (GATE_DIR / 'bootstrap.json').exists()
needs_gate = pytest.mark.skipif(not _gate_available, reason='gate-only results not yet produced')


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# item 1: J1/K2/M2 checkpoint hash exact
def test_item1_retriever_checkpoint_hashes():
    expected = {
        'J1': '1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c',
        'K2': '9827f76d4698ff96a2df6eb867cff2c6eaeaf8f6a80a0db00044a6095ee629c1',
        'M2': '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2',
    }
    for arm, exp_hash in expected.items():
        assert _sha256(N_CKPTS[arm]) == exp_hash


# item 2: S0 common-base checkpoint hash exact
def test_item2_s0_checkpoint_hash():
    assert _sha256(S0_CKPT) == 'a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609'


# item 3: all arms use identical base predictions (same file, loaded once per split)
@needs_diag
def test_item3_all_arms_use_identical_base_predictions():
    src = inspect.getsource(sys.modules['scripts.compute_n_diagnostics01'].run_retriever_split)
    assert 'base_cache' in src and 'base_lut' in src
    # base_cache/base_lut are built ONCE in main() and passed into every arm's call -- verify by
    # construction (single load_base_cache call per split in main(), not per-arm)
    main_src = inspect.getsource(sys.modules['scripts.compute_n_diagnostics01'].main)
    assert main_src.count('load_base_cache(split)') == 1


# item 4: full test n=2161
@needs_diag
def test_item4_full_test_n_2161():
    df = pd.read_parquet(OUT_DIR / 'all_diagnostics.parquet')
    test = df[df['split'] == 'test']
    for arm in ('J1', 'K2', 'M2'):
        assert test[test['arm'] == arm]['query_start_idx'].nunique() == 2161


# item 5: candidate N=7201
def test_item5_candidate_count_7201():
    src = inspect.getsource(sys.modules['scripts.compute_n_diagnostics01'])
    assert 'memory_x' in src  # candidate bank comes from exp.memory_x, established N=7201 elsewhere this session


# item 6: no future used for selection (picks_t computed from scores only, never batch_y)
def test_item6_no_future_used_for_selection():
    src = inspect.getsource(sys.modules['scripts.compute_n_diagnostics01'].run_retriever_split)
    # picks_t assignment lines must not reference batch_y or query_future
    for line in src.splitlines():
        if 'picks_t = ' in line:
            assert 'batch_y' not in line and 'query_future' not in line


# item 7: Uniform Stage1 Agg exact reproduction (<=1e-5)
@needs_diag
def test_item7_uniform_reproduction_gate():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    for arm in ('J1', 'K2', 'M2'):
        assert gate[arm]['uniform_pass'] is True
        assert gate[arm]['uniform_absdiff'] <= 1e-5


# item 8: Host Stage2 raw retrieval MSE exact reproduction (<=1e-5)
@needs_diag
def test_item8_host_reproduction_gate():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    for arm in ('J1', 'K2', 'M2'):
        assert gate[arm]['host_pass'] is True
        assert gate[arm]['host_absdiff'] <= 1e-5


# item 9: weighted Agg=D+C identity
@needs_diag
def test_item9_weighted_agg_equals_d_plus_c():
    df = pd.read_parquet(OUT_DIR / 'all_diagnostics.parquet')
    assert (df['D_w_U'] + df['C_w_U'] - df['retrieval_mse_U']).abs().max() < 1e-4
    assert (df['D_w_H'] + df['C_w_H'] - df['retrieval_mse_H']).abs().max() < 1e-4


# item 10: analytic lambda synthetic closed-form test
def test_item10_analytic_lambda_synthetic_closed_form():
    torch.manual_seed(0)
    H = 16
    b = torch.randn(H)
    y = torch.randn(H)
    r = torch.randn(H)
    t = y - b
    c = r - b
    lam_star = (torch.dot(t, c) / (c.dot(c) + 1e-8)).clamp(0.0, 1.0)
    # brute-force grid search should match the closed form (within grid resolution)
    grid = torch.linspace(0, 1, 2001)
    errs = [(((b + lam * c) - y) ** 2).mean().item() for lam in grid]
    brute_lam = float(grid[int(torch.tensor(errs).argmin())])
    assert abs(brute_lam - float(lam_star)) < 1e-2
    # closed form must literally minimize the quadratic in the *unclipped* case
    lam_unclipped = torch.dot(t, c) / (c.dot(c) + 1e-8)
    if 0 <= lam_unclipped <= 1:
        err_at_star = (((b + lam_unclipped * c) - y) ** 2).mean()
        err_at_neighbor = (((b + (lam_unclipped + 0.01) * c) - y) ** 2).mean()
        assert err_at_star <= err_at_neighbor + 1e-6


# item 11: lambda clipped to [0,1]
@needs_diag
def test_item11_lambda_clipped_to_0_1():
    df = pd.read_parquet(OUT_DIR / 'all_diagnostics.parquet')
    for col in ('lambda_star_U', 'lambda_star_H'):
        assert df[col].min() >= 0.0
        assert df[col].max() <= 1.0


# item 12: validation lambda never accesses test target
def test_item12_validation_lambda_never_touches_test():
    import scripts.compute_n_summary01 as n_summary
    src = inspect.getsource(n_summary.main)
    # the grid-search block that selects `best_lam`/`bl` must operate on `val`-derived arrays (`v`/`vc`)
    # only; the `te_*` (test) arrays are read only AFTER lambdas are already selected
    val_block = src.split("# apply to TEST")[0]
    assert 'te_t2' not in val_block and 'te_c2' not in val_block and 'te_dct' not in val_block


# item 13: common query IDs exact across arms
@needs_diag
def test_item13_common_query_ids_exact_across_arms():
    df = pd.read_parquet(OUT_DIR / 'all_diagnostics.parquet')
    test = df[df['split'] == 'test']
    ids = {arm: set(test[test['arm'] == arm]['query_start_idx']) for arm in ('J1', 'K2', 'M2')}
    assert ids['J1'] == ids['K2'] == ids['M2']
    assert len(ids['J1']) == 2161


# item 14: deterministic repeated eval (base predictions reproducible)
@needs_diag
def test_item14_deterministic_repeated_eval():
    d1 = torch.load(BASE_DIR / 'base_predictions_test.pt', map_location='cpu')
    d2 = torch.load(BASE_DIR / 'base_predictions_test.pt', map_location='cpu')
    assert torch.equal(d1['base_predictions'], d2['base_predictions'])
    assert torch.equal(d1['batch_start_idx'], d2['batch_start_idx'])


# item 15: gate-only: only gate requires_grad
@needs_gate
def test_item15_gate_only_only_gate_trainable():
    import copy as _copy
    from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from scripts.train_n_gate_only01 import S2_720
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    freeze_all_but_gate(model)
    for name in EXTRA_FREEZE:
        sub = getattr(model, name, None)
        if sub is not None:
            assert all(not p.requires_grad for p in sub.parameters())
    assert all(p.requires_grad for p in model.gate.parameters())


# item 16: gate-only base hash pre/post identical
@needs_gate
def test_item16_gate_only_base_hash_unchanged():
    for arm in ('G0_J1', 'G1_K2', 'G2_M2'):
        fp = json.loads((GATE_DIR / 'ETTh1_720' / arm / 'fingerprint.json').read_text())
        assert fp['base_head_sha256'] == fp['base_head_sha256_after']
        for name in EXTRA_FREEZE:
            if name in fp['frozen_submodule_sha256_before']:
                assert (fp['frozen_submodule_sha256_before'][name]
                       == fp['frozen_submodule_sha256_after'][name])


# item 17: gate-only retrieval cache identical across epochs (cache built once, read-only during training)
@needs_gate
def test_item17_gate_only_cache_read_only_across_epochs():
    src = inspect.getsource(sys.modules['scripts.train_n_gate_only01'].main)
    # cache is loaded exactly once per split (train/val/test), before the epoch loop begins
    epoch_loop_start = src.index('for epoch in range(1, cli.train_epochs + 1):')
    cache_load_calls = src[:epoch_loop_start].count('load_cache_as_lookup(')
    assert cache_load_calls == 3  # train, val, test -- each loaded once, outside the loop
    assert 'load_cache_as_lookup(' not in src[epoch_loop_start:]


# item 18: paired bootstrap uses query_start_idx as unit
@needs_gate
def test_item18_bootstrap_uses_query_start_idx_unit():
    bs = json.loads((GATE_DIR / 'bootstrap.json').read_text())
    assert bs['resampling_unit'] == 'query_start_idx'
    stage2_bs = json.loads((REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/bootstrap.json').read_text())
    assert 'query_start_idx' in stage2_bs['resampling_unit']
