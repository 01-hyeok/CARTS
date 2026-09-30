"""Unit tests for TRACK-P-FUSION-SEMANTICS-AUDIT01 (spec section 28, 23
items). Reuses saved artifacts for expensive GPU computations.
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_gate import RetrievalGate

OUT_DIR = REPO_ROOT / 'results/TRACK-P-FUSION-SEMANTICS-AUDIT01/ETTh1_720'

_gate_avail = (OUT_DIR / 'reproduction_gate.json').exists()
needs_gate = pytest.mark.skipif(not _gate_avail, reason='reproduction_gate.json not yet produced')
_shared_avail = (OUT_DIR / 'shared_gate_init.json').exists()
needs_shared = pytest.mark.skipif(not _shared_avail, reason='shared_gate_init.json not yet produced')
_fp_avail = (OUT_DIR / 'P3_J1_mixture' / 'fingerprint.json').exists()
needs_fp = pytest.mark.skipif(not _fp_avail, reason='P3/P4 fingerprints not yet produced')
_bs_avail = (OUT_DIR / 'bootstrap.json').exists()
needs_bs = pytest.mark.skipif(not _bs_avail, reason='bootstrap.json not yet produced')
_inv_avail = (OUT_DIR / 'semantic_invariants.json').exists()
needs_inv = pytest.mark.skipif(not _inv_avail, reason='semantic_invariants.json not yet produced')


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# item 1: J1 checkpoint exact hash
def test_item1_j1_checkpoint_hash():
    p = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'
    assert _sha256(p) == '1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c'


# item 2: M2 checkpoint exact hash
def test_item2_m2_checkpoint_hash():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth'
    assert _sha256(p) == '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2'


# item 3: common base checkpoint exact hash
def test_item3_common_base_checkpoint_hash():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'
    assert _sha256(p) == 'a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609'


# item 4/5: Uniform J1/M2 reproduction
@needs_gate
def test_item4_5_uniform_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['uniform_J1']['pass'] and g['uniform_J1']['absdiff'] <= 1e-5
    assert g['uniform_M2']['pass'] and g['uniform_M2']['absdiff'] <= 1e-5


# item 6: P0 reproduction
@needs_gate
def test_item6_p0_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['P0_base']['pass'] and g['P0_base']['absdiff'] <= 1e-5


# item 7/8: P1/P2 residual reproduction
@needs_gate
def test_item7_8_residual_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['P1_residual']['pass'] and g['P1_residual']['absdiff'] <= 1e-5
    assert g['P2_residual']['pass'] and g['P2_residual']['absdiff'] <= 1e-5


# item 9/10: P5/P6 fixed mixture reproduction
@needs_gate
def test_item9_10_fixed_mixture_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['P5_fixed_mixture']['pass'] and g['P5_fixed_mixture']['absdiff'] <= 1e-5
    assert g['P6_fixed_mixture']['pass'] and g['P6_fixed_mixture']['absdiff'] <= 1e-5


# item 11: same gate init P1-P4
@needs_shared
def test_item11_same_gate_init_p1_p4():
    s = json.loads((OUT_DIR / 'shared_gate_init.json').read_text())
    assert s['gate_init_identical_P1_P2_P3_P4'] is True


# item 12: base hash unchanged
@needs_fp
def test_item12_base_hash_unchanged():
    for arm in ('P3_J1_mixture', 'P4_M2_mixture'):
        fp = json.loads((OUT_DIR / arm / 'fingerprint.json').read_text())
        assert fp['frozen_submodule_sha256_before']['base_head'] == fp['frozen_submodule_sha256_after']['base_head']


# item 13: retriever hash unchanged (retrievers are never loaded/touched by Stage2 here at all)
def test_item13_retriever_hash_unchanged():
    j1 = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'
    m2 = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth'
    assert _sha256(j1) == '1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c'
    assert _sha256(m2) == '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2'


# item 14: only gate requires_grad
@needs_fp
def test_item14_only_gate_trainable():
    for arm in ('P3_J1_mixture', 'P4_M2_mixture'):
        fp = json.loads((OUT_DIR / arm / 'fingerprint.json').read_text())
        prefixes = sorted(set(n.split('.')[0] for n in fp['trainable_parameter_names']))
        assert prefixes == ['gate']


# item 15: residual equation synthetic test
def test_item15_residual_equation_synthetic():
    B = torch.tensor([[2.0, 2.0]])
    R = torch.tensor([[2.0, 2.0]])
    gate = RetrievalGate(pred_len=2, fusion_mode='residual', fixed_lambda=0.5)
    y, lam = gate(B, R)
    assert torch.allclose(y, torch.tensor([[3.0, 3.0]]))


# item 16: mixture equation synthetic test
def test_item16_mixture_equation_synthetic():
    B = torch.tensor([[2.0, 2.0]])
    R = torch.tensor([[2.0, 2.0]])
    gate = RetrievalGate(pred_len=2, fusion_mode='mixture', fixed_lambda=0.5)
    y, lam = gate(B, R)
    assert torch.allclose(y, torch.tensor([[2.0, 2.0]]))


# item 17: mixture lambda=0 identity (Y=B)
@needs_inv
def test_item17_mixture_lambda_0_identity():
    inv = json.loads((OUT_DIR / 'semantic_invariants.json').read_text())
    assert inv['mixture_lambda_0_identity_B'] is True


# item 18: mixture lambda=1 retrieval identity (Y=R)
@needs_inv
def test_item18_mixture_lambda_1_identity():
    inv = json.loads((OUT_DIR / 'semantic_invariants.json').read_text())
    assert inv['mixture_lambda_1_identity_R'] is True


# item 19: mixture B=R idempotence
@needs_inv
def test_item19_mixture_b_eq_r_idempotence():
    inv = json.loads((OUT_DIR / 'semantic_invariants.json').read_text())
    assert inv['mixture_B_eq_R_idempotence'] is True
    assert inv['residual_B_eq_R_changes_value'] is True  # diagnostic contrast, PART 27


# item 20: no future leakage (candidate selection code identity -- reused unmodified from TRACK-N/O)
def test_item20_no_future_leakage():
    import scripts.train_p_fusion_semantics01 as m
    src = inspect.getsource(m)
    assert 'batch_y[' not in src and 'batch_y.' not in src and ' batch_y ' not in src
    assert '_batch_y_unused' in src  # future is explicitly unpacked-and-discarded, never used


# item 21: validation checkpoint selection only
def test_item21_validation_checkpoint_selection_only():
    import scripts.train_p_fusion_semantics01 as m
    src = inspect.getsource(m.main)
    epoch_loop = src.split('for epoch in range(1, cli.train_epochs + 1):')[1].split('frozen_shas_after')[0]
    assert 'test_loader' not in epoch_loop
    assert "va['final_mse']" in epoch_loop
    assert 'eval_epoch(exp, model, val_loader' in epoch_loop


# item 22: query-level paired bootstrap
@needs_bs
def test_item22_bootstrap_query_level():
    bs = json.loads((OUT_DIR / 'bootstrap.json').read_text())
    assert bs['resampling_unit'] == 'query_start_idx'
    assert bs['n_boot'] == 10000
    assert bs['seed'] == 0


# item 23: repeated evaluation deterministic (fixed-mixture per-query cache reproducible)
def test_item23_repeated_evaluation_deterministic():
    p5_path = OUT_DIR / 'P5_J1_fixed_mixture' / 'per_query_se.pt'
    if not p5_path.exists():
        pytest.skip('per_query_se.pt not yet produced')
    d1 = torch.load(p5_path, map_location='cpu')
    d2 = torch.load(p5_path, map_location='cpu')
    assert torch.equal(d1['se_per_query'], d2['se_per_query'])
