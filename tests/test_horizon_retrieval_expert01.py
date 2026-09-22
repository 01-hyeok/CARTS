"""TRACK-A-HORIZON-RETRIEVAL-EXPERT01 sanity tests."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score
from scripts.train_horizon_retrieval_expert01 import (BlockCorrectionHeads, kl_loss,
                                                       normalized_teacher_prob,
                                                       teacher_diagnostics)


# ---- zero-init correction head: s_b == s_G exactly at init ----
def test_zero_init_block_head_matches_global_score():
    torch.manual_seed(0)
    d_model, bsz, n = 16, 4, 30
    heads = BlockCorrectionHeads(d_model)
    z_q = torch.randn(bsz, d_model)
    E = torch.randn(n, d_model)
    s_g = arm_score(z_q, E, None)
    for bidx in range(3):
        z_q_b = heads(z_q, bidx)
        assert torch.allclose(z_q_b, z_q)
        s_b = arm_score(z_q_b, E, None)
        assert torch.allclose(s_b, s_g, atol=1e-6)


# ---- after a gradient step, heads move away from zero and scores diverge ----
def test_block_head_learns_away_from_zero():
    torch.manual_seed(1)
    d_model = 8
    heads = BlockCorrectionHeads(d_model)
    opt = torch.optim.SGD(heads.parameters(), lr=1.0)
    z_q = torch.randn(2, d_model)
    out = heads(z_q, 0)
    loss = out.sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(heads.heads[0].weight, torch.zeros_like(heads.heads[0].weight))
    # untouched blocks stay exactly zero
    assert torch.allclose(heads.heads[1].weight, torch.zeros_like(heads.heads[1].weight))


# ---- KL gradient equals soft cross-entropy gradient (teacher entropy term
# is constant w.r.t. student params) ----
def test_kl_gradient_equals_soft_ce_gradient():
    torch.manual_seed(2)
    bsz, n = 5, 12
    valid_mask = torch.rand(bsz, n) > 0.2
    valid_mask[:, 0] = True  # avoid all-invalid rows
    d = torch.randn(bsz, n).abs()
    p_t = normalized_teacher_prob(d, valid_mask, tau_t=0.3)

    s1 = torch.randn(bsz, n, requires_grad=True)
    s2 = s1.detach().clone().requires_grad_(True)

    kl = kl_loss(p_t, s1, valid_mask, tau_s=0.2)
    kl.backward()

    logits = (s2 / 0.2).masked_fill(~valid_mask, float('-inf'))
    log_p_s = torch.log_softmax(logits, dim=-1).masked_fill(~valid_mask, 0.0)
    soft_ce = -(p_t.detach() * log_p_s).masked_fill(~valid_mask, 0.0).sum(-1).mean()
    soft_ce.backward()

    assert torch.allclose(s1.grad, s2.grad, atol=1e-5)


# ---- normalized_teacher_prob: masked entries get exactly zero probability,
# z-score computed over valid entries only, matches manual reference ----
def test_normalized_teacher_prob_masking_and_zscore():
    d = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    valid = torch.tensor([[True, True, True, False]])
    p = normalized_teacher_prob(d, valid, tau_t=1.0)
    assert p[0, 3].item() == pytest.approx(0.0, abs=1e-9)
    assert p[0].sum().item() == pytest.approx(1.0, abs=1e-6)
    # manual z-score over the 3 valid entries [1,2,3]: mean=2, std=sqrt(2/3)
    valid_d = d[0, :3]
    mean = valid_d.mean()
    std = valid_d.std(unbiased=False)
    normalized = (valid_d - mean) / std
    manual = torch.softmax(-normalized / 1.0, dim=-1)
    assert torch.allclose(p[0, :3], manual, atol=1e-5)
    # lower distance -> higher teacher probability
    assert p[0, 0] > p[0, 1] > p[0, 2]


# ---- teacher diagnostics: entropy/top1_mass/effective_positives on known
# distributions (uniform vs near-one-hot) ----
def test_teacher_diagnostics_uniform_vs_onehot():
    valid = torch.tensor([[True, True, True, True]])
    uniform = torch.full((1, 4), 0.25)
    diag_u = teacher_diagnostics(uniform, valid)
    assert diag_u['teacher_top1_mass'] == pytest.approx(0.25, abs=1e-6)
    assert diag_u['teacher_effective_positives'] == pytest.approx(4.0, abs=1e-4)

    onehot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    diag_o = teacher_diagnostics(onehot, valid)
    assert diag_o['teacher_top1_mass'] == pytest.approx(1.0, abs=1e-6)
    assert diag_o['teacher_effective_positives'] == pytest.approx(1.0, abs=1e-4)
    assert diag_o['teacher_entropy'] < diag_u['teacher_entropy']


# ---- structural leakage check: arm_score/encode_raw never take future/
# query_future as an argument at all ----
def test_score_functions_never_receive_future():
    import inspect
    from scripts.train_factorial_e2e01 import encode_raw
    sig_score = inspect.signature(arm_score)
    sig_encode = inspect.signature(encode_raw)
    for sig in (sig_score, sig_encode):
        for name in sig.parameters:
            assert 'future' not in name.lower() and 'query_y' not in name.lower()


# ---- global KL loss scale vs block-arm's weighted combination: neither
# term should be identically zero when teacher != student ----
def test_block_loss_combination_weights_all_terms_nonzero():
    torch.manual_seed(3)
    bsz, n = 6, 10
    valid = torch.ones(bsz, n, dtype=torch.bool)
    d_g = torch.rand(bsz, n)
    p_t_g = normalized_teacher_prob(d_g, valid, tau_t=0.1)
    s_g = torch.randn(bsz, n, requires_grad=True)
    kl_g = kl_loss(p_t_g, s_g, valid, tau_s=0.1)

    kl_blocks = []
    for _ in range(3):
        d_b = torch.rand(bsz, n)
        p_t_b = normalized_teacher_prob(d_b, valid, tau_t=0.1)
        s_b = torch.randn(bsz, n, requires_grad=True)
        kl_blocks.append(kl_loss(p_t_b, s_b, valid, tau_s=0.1))

    total = 0.5 * kl_g + sum(kl_blocks) / 6.0
    assert float(kl_g) > 0
    for kb in kl_blocks:
        assert float(kb) > 0
    assert float(total) > 0
    # global term's contribution is exactly half its own value, block terms
    # each contribute 1/6 -- verify the arithmetic directly (not re-deriving
    # gradient magnitude, just the documented weighting).
    assert torch.allclose(total, 0.5 * kl_g + (kl_blocks[0] + kl_blocks[1] + kl_blocks[2]) / 6.0)
