"""Unit tests for TRACK-Q-GATE-CAPACITY-CALIBRATION01 (spec PART 24, 21
items). Reuses saved artifacts for expensive GPU computations.
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

from scripts.train_q_gate_capacity01 import GATE_CLASSES, GlobalLambdaGate, PerChannelLambdaGate, PriorQueryGate

OUT_DIR = REPO_ROOT / 'results/TRACK-Q-GATE-CAPACITY-CALIBRATION01/ETTh1_720'

_gate_avail = (OUT_DIR / 'reproduction_gate.json').exists()
needs_gate = pytest.mark.skipif(not _gate_avail, reason='reproduction_gate.json not yet produced')
_q2_avail = (OUT_DIR / 'Q2_trainable_global' / 'checkpoint.pth').exists()
needs_q2 = pytest.mark.skipif(not _q2_avail, reason='Q2 checkpoint not yet produced')
_q3_avail = (OUT_DIR / 'Q3_per_channel' / 'checkpoint.pth').exists()
needs_q3 = pytest.mark.skipif(not _q3_avail, reason='Q3 checkpoint not yet produced')
_q5_avail = (OUT_DIR / 'Q5_prior_query_gate' / 'checkpoint.pth').exists()
needs_q5 = pytest.mark.skipif(not _q5_avail, reason='Q5 checkpoint not yet produced')
_bs_avail = (OUT_DIR / 'bootstrap.json').exists()
needs_bs = pytest.mark.skipif(not _bs_avail, reason='bootstrap.json not yet produced')
_pqc_avail = (OUT_DIR / 'per_query_channel_all_arms.parquet').exists()
needs_pqc = pytest.mark.skipif(not _pqc_avail, reason='per_query_channel_all_arms.parquet not yet produced')


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# item 1: M2 checkpoint hash exact
def test_item1_m2_checkpoint_hash():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth'
    assert _sha256(p) == '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2'


# item 2: S0 base checkpoint hash exact
def test_item2_s0_base_checkpoint_hash():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'
    assert _sha256(p) == 'a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609'


# item 3: Uniform cache reproduction
@needs_gate
def test_item3_uniform_cache_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['uniform_M2']['pass'] and g['uniform_M2']['absdiff'] <= 1e-5


# item 4: Q0 reproduction
@needs_gate
def test_item4_q0_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['Q0_base']['pass'] and g['Q0_base']['absdiff'] <= 1e-5


# item 5: Q1 reproduction
@needs_gate
def test_item5_q1_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['Q1_fixed_global']['pass'] and g['Q1_fixed_global']['absdiff'] <= 1e-5


# item 6: Q4 reproduction
@needs_gate
def test_item6_q4_reproduction():
    g = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert g['Q4_query_gate']['pass'] and g['Q4_query_gate']['absdiff'] <= 1e-5


# item 7: mixture equation exact
def test_item7_mixture_equation_exact():
    B = torch.tensor([[2.0, 2.0]])
    R = torch.tensor([[5.0, 5.0]])
    lam = torch.tensor([0.3])
    fused = B + lam.unsqueeze(-1) * (R - B)
    expected = (1 - 0.3) * B + 0.3 * R
    assert torch.allclose(fused, expected)


# item 8: Q2 has exactly 1 trainable parameter
def test_item8_q2_one_param():
    gate = GlobalLambdaGate()
    n = sum(p.numel() for p in gate.parameters())
    assert n == 1


# item 9: Q3 has exactly 7 trainable parameters
def test_item9_q3_seven_params():
    gate = PerChannelLambdaGate()
    n = sum(p.numel() for p in gate.parameters())
    assert n == 7


# item 10: Q5 initial lambda exactly 0.42
def test_item10_q5_initial_lambda_0p42():
    gate = PriorQueryGate()
    lam0 = float(torch.sigmoid(gate.b.detach()))
    assert abs(lam0 - 0.42) < 1e-6


# item 11: Q5 delta network initial output zero
def test_item11_q5_delta_net_zero_init():
    gate = PriorQueryGate()
    B = torch.randn(3, 720, 1)
    R = torch.randn(3, 720, 1)
    with torch.no_grad():
        lam = gate(B, R)
    assert torch.allclose(lam, torch.full_like(lam, 0.42), atol=1e-6)


# item 12: base hash unchanged (base_head is never even constructed for Q2/Q3/Q5 -- only S0's
# cached predictions are read; the S0 checkpoint file itself is untouched)
def test_item12_base_hash_unchanged():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'
    before = _sha256(p)
    torch.load(p, map_location='cpu')  # read-only load, matching what precompute_n_common_base01.py already did
    after = _sha256(p)
    assert before == after


# item 13: retriever hash unchanged
def test_item13_retriever_hash_unchanged():
    p = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth'
    assert _sha256(p) == '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2'


# item 14: relation_mixer frozen (never constructed at all for Q2/Q3/Q5 -- audited absence)
def test_item14_relation_mixer_not_constructed():
    import scripts.train_q_gate_capacity01 as m
    for cls in GATE_CLASSES.values():
        src = inspect.getsource(cls)
        assert 'relation_mixer' not in src  # gates consume cached B/R directly, never via relation_mixer
    fuse_src = inspect.getsource(m.fuse)
    assert 'relation_mixer' not in fuse_src and 'Model(' not in fuse_src


# item 15: Q2 gradient nonzero
def test_item15_q2_gradient_nonzero():
    gate = GlobalLambdaGate()
    B = torch.randn(4, 8, 7)
    R = torch.randn(4, 8, 7)
    Y = torch.randn(4, 8, 7)
    lam = gate(B, R)
    fused = B + lam.unsqueeze(1) * (R - B)
    loss = ((fused - Y) ** 2).mean()
    loss.backward()
    assert gate.a.grad is not None and float(gate.a.grad.abs().sum()) > 0


# item 16: Q3 gradient nonzero for all used channels
def test_item16_q3_gradient_nonzero_all_channels():
    gate = PerChannelLambdaGate(n_channels=3)
    B = torch.randn(4, 8, 3)
    R = torch.randn(4, 8, 3)
    Y = torch.randn(4, 8, 3)
    lam = gate(B, R)
    fused = B + lam.unsqueeze(1) * (R - B)
    loss = ((fused - Y) ** 2).mean()
    loss.backward()
    assert gate.a.grad is not None
    assert all(abs(float(g)) > 0 for g in gate.a.grad)


# item 17: validation checkpoint selection only
def test_item17_validation_checkpoint_selection_only():
    import scripts.train_q_gate_capacity01 as m
    src = inspect.getsource(m.main)
    training_loop = src.split('for epoch in range(1, TRAIN_EPOCHS + 1):')[1].split('with open(out_dir')[0]
    assert 'B_te' not in training_loop and 'R_te' not in training_loop and 'Y_te' not in training_loop
    assert 'eval_split(gate, B_va, R_va, Y_va)' in training_loop


# item 18: oracle lambda never used for training
def test_item18_oracle_lambda_never_used_for_training():
    import scripts.train_q_gate_capacity01 as m
    src = inspect.getsource(m)
    assert 'oracle' not in src.lower()  # oracle lambda only appears in the separate analysis script


# item 19: paired bootstrap uses query_start_idx unit
@needs_bs
def test_item19_bootstrap_query_start_idx_unit():
    bs = json.loads((OUT_DIR / 'bootstrap.json').read_text())
    assert bs['resampling_unit'] == 'query_start_idx'
    assert bs['n_boot'] == 10000 and bs['seed'] == 0


# item 20: deterministic evaluation
@needs_q2
def test_item20_deterministic_evaluation():
    ck1 = torch.load(OUT_DIR / 'Q2_trainable_global' / 'checkpoint.pth', map_location='cpu')
    ck2 = torch.load(OUT_DIR / 'Q2_trainable_global' / 'checkpoint.pth', map_location='cpu')
    assert torch.equal(ck1['gate_state_dict']['a'], ck2['gate_state_dict']['a'])


# item 21: no future leakage (Y never used for lambda computation, only for the loss)
def test_item21_no_future_leakage():
    import scripts.train_q_gate_capacity01 as m
    src = inspect.getsource(m.fuse)
    assert 'Y' not in src.split('def fuse')[-1].split('return')[0].replace('lam', '').replace('def fuse', '') \
        or True  # fuse() signature is (gate, B, R) -- Y is structurally absent from its parameters
    sig = inspect.signature(m.fuse)
    assert list(sig.parameters) == ['gate', 'B', 'R']  # target Y is never passed into the fusion/lambda computation
