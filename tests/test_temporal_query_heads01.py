"""TRACK-TEMPORAL-QUERY-HEADS01 -- CPU unit tests for
`models/TemporalQueryHeads.py`. Uses a tiny stand-in MLP encoder with
the exact interface `model.encoder` (RelationEncoder) exposes --
`forward(relation_x: [B,F,L]) -> [B,D]`, L2-normalized -- so these
tests exercise the real masking/attention math without needing the
full RelationStage1 build pipeline or a GPU."""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.TemporalQueryHeads import (
    attention_entropy, build_patch_time_masks, compute_temporal_token_bank,
    pairwise_head_attention_cosine, query_attention_diversity, temporal_attention_heads,
)
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads

torch.manual_seed(0)


class _FakeMLPEncoder(nn.Module):
    """Same contract as RelationEncoder in mlp mode: [B,F,L] -> [B,D], unit-norm."""

    def __init__(self, feat, seq_len, d_model=8, d_ff=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat * seq_len, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, relation_x):
        bsz = relation_x.size(0)
        h = self.net(relation_x.reshape(bsz, -1))
        return F.normalize(h, dim=-1)


def test_patch_masks_cover_every_position_exactly_once():
    masks = build_patch_time_masks(seq_len=96, patch_len=16)
    assert masks.shape == (6, 96)
    assert torch.equal(masks.sum(dim=0), torch.ones(96, dtype=torch.long))
    assert (masks.sum(dim=1) == 16).all()


def test_patch_masks_edge_case_remainder_patch():
    masks = build_patch_time_masks(seq_len=100, patch_len=16)
    assert masks.shape[0] == math.ceil(100 / 16) == 7
    assert torch.equal(masks.sum(dim=0), torch.ones(100, dtype=torch.long))
    assert masks[-1].sum() == 4  # 100 - 6*16


def test_token_bank_shape_and_unit_norm():
    bsz, feat, seq_len, d_model = 4, 1, 96, 8
    enc = _FakeMLPEncoder(feat, seq_len, d_model)
    masks = build_patch_time_masks(seq_len, 16)
    relation_x = torch.randn(bsz, feat, seq_len)
    t_p, z_p_raw, z_0 = compute_temporal_token_bank(enc, relation_x, masks)
    assert t_p.shape == (bsz, 6, d_model)
    assert z_p_raw.shape == (bsz, 6, d_model)
    assert z_0.shape == (bsz, d_model)
    norms = t_p.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.isfinite(t_p).all()


def test_token_bank_no_nan_even_with_zero_patch_content():
    """If a patch's masked input happens to be identical to the all-zero
    baseline (degenerate case: patch content itself is exactly zero),
    z_p_raw - z_0 is the zero vector and normalize() must not produce NaN
    (F.normalize has an eps floor; this just confirms it holds here)."""
    bsz, feat, seq_len, d_model = 2, 1, 32, 8
    enc = _FakeMLPEncoder(feat, seq_len, d_model)
    masks = build_patch_time_masks(seq_len, 16)
    relation_x = torch.zeros(bsz, feat, seq_len)  # every patch == zero baseline
    t_p, z_p_raw, z_0 = compute_temporal_token_bank(enc, relation_x, masks)
    assert torch.isfinite(t_p).all()
    assert torch.allclose(z_p_raw - z_0.unsqueeze(1), torch.zeros_like(t_p), atol=1e-6)


def test_temporal_attention_sums_to_one_and_unit_norm_output():
    bsz, n_slots, n_patches, d_model = 4, 5, 6, 8
    g = F.normalize(torch.randn(bsz, n_slots, d_model), dim=-1)
    t_p = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)
    q, a = temporal_attention_heads(g, t_p)
    assert q.shape == (bsz, n_slots, d_model)
    assert a.shape == (bsz, n_slots, n_patches)
    sums = a.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    q_norms = q.norm(dim=-1)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)


def test_zero_added_trainable_parameters_vs_v5_slotheads():
    """TQH-5-Masked must add ZERO new trainable parameters beyond V5's
    own SlotHeads -- attention is a parameter-free function."""
    d_model = 8
    slot_heads = SlotHeads(d_model, n_slots=5, std=1e-3)
    v5_param_count = sum(p.numel() for p in slot_heads.parameters())

    bsz, n_patches = 4, 6
    z_q = F.normalize(torch.randn(bsz, d_model), dim=-1)
    g = slot_heads(z_q)
    t_p = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)
    before = {id(p) for p in slot_heads.parameters()}
    q, a = temporal_attention_heads(g, t_p)
    after = {id(p) for p in slot_heads.parameters()}
    assert before == after, 'temporal_attention_heads must not register new parameters'
    assert sum(p.numel() for p in slot_heads.parameters()) == v5_param_count


def test_query_attention_diversity_differs_across_random_queries():
    """Different (random, hence almost surely distinct) queries should NOT
    produce identical attention for the same head -- diversity should be
    well below 1.0 (perfect identity)."""
    bsz, n_slots, n_patches, d_model = 16, 5, 6, 8
    torch.manual_seed(1)
    g = F.normalize(torch.randn(bsz, n_slots, d_model), dim=-1)
    t_p = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)
    q, a = temporal_attention_heads(g, t_p)
    div = query_attention_diversity(a)
    assert div == div  # not NaN
    assert div < 0.99, f'attention looks query-invariant (diversity={div}), expected real variation'


def test_pairwise_head_collapse_detection():
    """If all heads share the SAME query vector g_m, their attention
    distributions must be (near-)identical -- collapse correctly detected
    (cosine ~ 1). With independent random heads, cosine should be well
    below 1."""
    bsz, n_slots, n_patches, d_model = 8, 5, 6, 8
    torch.manual_seed(2)
    t_p = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)

    g_single = F.normalize(torch.randn(bsz, 1, d_model), dim=-1).expand(bsz, n_slots, d_model)
    _, a_collapsed = temporal_attention_heads(g_single, t_p)
    cos_collapsed = pairwise_head_attention_cosine(a_collapsed)
    assert cos_collapsed > 0.999

    g_diverse = F.normalize(torch.randn(bsz, n_slots, d_model), dim=-1)
    _, a_diverse = temporal_attention_heads(g_diverse, t_p)
    cos_diverse = pairwise_head_attention_cosine(a_diverse)
    assert cos_diverse < 0.99


def test_attention_entropy_bounds():
    bsz, n_slots, n_patches, d_model = 4, 5, 6, 8
    g = F.normalize(torch.randn(bsz, n_slots, d_model), dim=-1)
    t_p = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)
    _, a = temporal_attention_heads(g, t_p)
    ent = attention_entropy(a)
    assert ent.shape == (bsz, n_slots)
    assert (ent >= 0).all()
    assert (ent <= math.log(n_patches) + 1e-4).all()


def test_gradient_flows_to_encoder_and_slotheads():
    bsz, feat, seq_len, d_model = 4, 1, 32, 8
    enc = _FakeMLPEncoder(feat, seq_len, d_model)
    slot_heads = SlotHeads(d_model, n_slots=5, std=1e-3)
    masks = build_patch_time_masks(seq_len, 16)
    relation_x = torch.randn(bsz, feat, seq_len)

    z_q = enc(relation_x)
    t_p, _, _ = compute_temporal_token_bank(enc, relation_x, masks)
    g = slot_heads(z_q)
    q, a = temporal_attention_heads(g, t_p)
    loss = q.pow(2).sum()
    loss.backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert slot_heads.W.grad is not None and slot_heads.W.grad.abs().sum() > 0


def test_no_future_leakage_in_token_bank_signature():
    """compute_temporal_token_bank/temporal_attention_heads take only
    relation_x/encoder/masks/g/t_p -- no batch_y, no query_future, no
    teacher/candidate-future tensor can reach them by construction
    (inspected at the call-site; this test just pins the function
    signatures so a future edit can't quietly add one)."""
    import inspect
    sig1 = inspect.signature(compute_temporal_token_bank)
    sig2 = inspect.signature(temporal_attention_heads)
    assert list(sig1.parameters) == ['encoder', 'relation_x', 'patch_masks']
    assert list(sig2.parameters) == ['g', 't_p']
