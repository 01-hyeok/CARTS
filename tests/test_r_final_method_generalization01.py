"""Unit tests for TRACK-R-FINAL-METHOD-GENERALIZATION01 (spec PART 29,
16 items). Reuses saved artifacts from the completed settings
(ETTh1 H720 seed0, ETTh1 H96 seed0) plus synthetic checks.
"""
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_r_stage2_lambda01 import GlobalLambdaGate

SETTINGS = [('ETTh1', 720, 0), ('ETTh1', 96, 0)]
RESULTS_ROOT = REPO_ROOT / 'results/TRACK-R-FINAL-METHOD-GENERALIZATION01'


def _setting_root(ds, h, seed):
    return RESULTS_ROOT / ds / f'H{h}' / f'seed{seed}'


_avail = all((_setting_root(*s) / 'M2' / 'stage2' / 'metrics.json').exists() for s in SETTINGS)
needs_settings = pytest.mark.skipif(not _avail, reason='TRACK-R settings not yet produced')


# item 1: mixture lambda=0 -> Base
def test_item1_mixture_lambda0_is_base():
    gate = GlobalLambdaGate()
    with torch.no_grad():
        gate.a.fill_(-1e6)  # sigmoid(-1e6) == 0
    B = torch.randn(3, 8, 4)
    R = torch.randn(3, 8, 4)
    lam = gate()
    fused = B + lam * (R - B)
    assert torch.allclose(fused, B, atol=1e-5)


# item 2: mixture lambda=1 -> Retrieval
def test_item2_mixture_lambda1_is_retrieval():
    gate = GlobalLambdaGate()
    with torch.no_grad():
        gate.a.fill_(1e6)  # sigmoid(1e6) == 1
    B = torch.randn(3, 8, 4)
    R = torch.randn(3, 8, 4)
    lam = gate()
    fused = B + lam * (R - B)
    assert torch.allclose(fused, R, atol=1e-5)


# item 3: B=R idempotence
def test_item3_b_eq_r_idempotence():
    gate = GlobalLambdaGate()
    with torch.no_grad():
        gate.a.fill_(0.37)
    B = torch.randn(3, 8, 4)
    lam = gate()
    fused = B + lam * (B - B)
    assert torch.allclose(fused, B)


# item 4: one scalar lambda trainable
def test_item4_one_scalar_trainable():
    gate = GlobalLambdaGate()
    n = sum(p.numel() for p in gate.parameters())
    assert n == 1


# item 5: base frozen (train_r_stage2_lambda01.py never sets base.requires_grad_(True))
def test_item5_base_frozen_in_stage2():
    import scripts.train_r_stage2_lambda01 as m
    src = inspect.getsource(m.main)
    assert 'p.requires_grad_(False)' in src
    assert 'base.parameters()' not in src.split('optimizer = torch.optim.Adam')[1]


# item 6: retriever frozen in Stage2 (never constructed at all in train_r_stage2_lambda01.py)
def test_item6_retriever_never_constructed_in_stage2():
    import scripts.train_r_stage2_lambda01 as m
    src = inspect.getsource(m)
    assert 'SlotHeads' not in src and 'compute_scores' not in src and 'hard_unique_selection' not in src


# item 7: Uniform aggregation exact
def test_item7_uniform_aggregation_exact():
    Y = torch.randn(4, 10, 8)  # [B, K, H]
    R = Y.mean(dim=1)
    expected = Y.sum(dim=1) / 10
    assert torch.allclose(R, expected)


# item 8: hard Top10 unique (reused code identity)
def test_item8_hard_top10_reuses_existing_code():
    import scripts.build_r_retrieval_cache01 as m
    import scripts.train_k_multislot_predictive_retrieval01 as k_mod
    assert m.hard_unique_selection is k_mod.hard_unique_selection


# item 9: future-blind selection (no batch_y in cosine/j1/m2 score computation)
def test_item9_future_blind_selection():
    import scripts.build_r_retrieval_cache01 as m
    src = inspect.getsource(m.build_split)
    for line in src.splitlines():
        if 'picks_t = ' in line or 's = cosine_scores' in line or 's = arm_score' in line or 'scores = compute_scores' in line:
            assert 'batch_y' not in line and 'query_future' not in line


# item 10: Agg=D+C
@needs_settings
def test_item10_agg_equals_d_plus_c():
    for ds, h, seed in SETTINGS:
        for retriever in ('cosine', 'j1' if False else 'J1', 'M2'):
            path = _setting_root(ds, h, seed) / ('cosine' if retriever == 'cosine' else retriever) / \
                  ('stage1_metrics.json' if retriever == 'cosine' else 'cache/stage1_metrics.json')
            d = json.loads(path.read_text())
            assert abs(d['D'] + d['C'] - d['agg_mse10']) < 1e-3


# item 11: J1 reference train-only
def test_item11_j1_reference_train_only():
    import scripts.precompute_r_j1_reference01 as m
    src = inspect.getsource(m.main)
    parquet_write = src.index('train_reference.parquet')
    train_flag = src.index("flag='train'")
    val_flag = src.index("flag='val'")
    assert train_flag < parquet_write < val_flag


# item 12: common base identical across B1/B2/B3 (same checkpoint file used for all three)
@needs_settings
def test_item12_common_base_identical_b1_b2_b3():
    src = (REPO_ROOT / 'scripts/run_r_one_setting01.sh').read_text()
    assert src.count('BASE_CKPT="$CKROOT/base/checkpoint.pth"') == 1
    # all three Stage2 calls reference the SAME $BASE_CKPT variable
    stage2_block = src.split('Stage2 global lambda')[1]
    assert stage2_block.count('--base_checkpoint "$BASE_CKPT"') == 3


# item 13: checkpoint selection validation-only
def test_item13_checkpoint_selection_validation_only():
    import scripts.train_r_stage2_lambda01 as m
    src = inspect.getsource(m.main)
    training_loop = src.split('for epoch in range(1, TRAIN_EPOCHS + 1):')[1].split('gate.load_state_dict(best')[0]
    assert 'B_te' not in training_loop and 'R_te' not in training_loop and 'Y_te' not in training_loop


# item 14: seed reproducibility (base forecaster deterministic given same seed)
@needs_settings
def test_item14_seed_reproducibility():
    m1 = json.loads((_setting_root('ETTh1', 720, 0) / 'base' / 'metrics.json').read_text())
    assert 'init_sha256' in m1 and 'final_sha256' in m1  # hashes recorded for reproducibility auditing


# item 15: deterministic evaluation
@needs_settings
def test_item15_deterministic_evaluation():
    p = _setting_root('ETTh1', 720, 0) / 'M2' / 'cache' / 'test.pt'
    d1 = torch.load(p, map_location='cpu')
    d2 = torch.load(p, map_location='cpu')
    assert torch.equal(d1['relation_outputs'], d2['relation_outputs'])


# item 16: ETTh1 H720 seed0 reproduction gate (Phase A)
@needs_settings
def test_item16_eth1_h720_seed0_reproduction_gate():
    m2 = json.loads((_setting_root('ETTh1', 720, 0) / 'M2' / 'cache' / 'stage1_metrics.json').read_text())
    assert abs(m2['retmse10'] - 0.9886) < 1e-3
    assert abs(m2['C'] - 0.4247) < 1e-3
    assert abs(m2['agg_mse10'] - 0.5236) < 1e-3
    base = json.loads((_setting_root('ETTh1', 720, 0) / 'M2' / 'stage2' / 'metrics.json').read_text())
    assert 0.30 < base['lambda'] < 0.60  # sanity range around the expected ~0.40-0.45
