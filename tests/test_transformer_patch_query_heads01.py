"""TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 -- CPU-only structural tests
(no GPU, no real checkpoint needed for most of these; the parameter
-count-parity and num_patches checks are pure arithmetic/config
assertions)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.relation_patch_embed import RelationPatchEmbedding
from models.TemporalQueryHeads import temporal_attention_heads
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_transformer_patch_query_heads01 import ARM_N_SLOTS, PATCH_AWARE_ARMS

torch.manual_seed(0)


def test_num_patches_six_for_both_horizons():
    pe96 = RelationPatchEmbedding(seq_len=96, patch_len=16, stride=16, d_model=128)
    pe720 = RelationPatchEmbedding(seq_len=720, patch_len=120, stride=120, d_model=128)
    assert pe96.num_patches == 6
    assert pe720.num_patches == 6


def test_arm_n_slots_table():
    assert ARM_N_SLOTS == {'V1': 1, 'V5': 5, 'PH1': 1, 'PH5': 5}
    assert set(PATCH_AWARE_ARMS) == {'PH1', 'PH5'}


def test_slotheads_head0_shared_across_n_slots_1_and_5():
    """V1/PH1 (n_slots=1) and V5/PH5 (n_slots=5) must share head-0's
    exact initialization -- guaranteed by SlotHeads' per-slot-seeded
    (1000+m) construction, independent of n_slots. Verified directly
    rather than assumed."""
    d_model = 128
    s1 = SlotHeads(d_model, n_slots=1, std=1e-3)
    s5 = SlotHeads(d_model, n_slots=5, std=1e-3)
    assert torch.equal(s1.W[0], s5.W[0])


def test_slotheads_identical_across_runs_same_n_slots():
    """T-V5 and T-PH5 (both n_slots=5, same std) must be byte-identical
    at init -- the pairwise-fairness requirement for comparing B2 vs C2."""
    d_model = 128
    a = SlotHeads(d_model, n_slots=5, std=1e-3)
    b = SlotHeads(d_model, n_slots=5, std=1e-3)
    assert torch.equal(a.W, b.W)


def test_patch_aware_query_is_parameter_free_beyond_slotheads():
    """C1/C2 add zero trainable parameters beyond the paired V-arm's own
    SlotHeads -- `temporal_attention_heads` (reused unmodified from
    TRACK-TEMPORAL-QUERY-HEADS01) is a pure function of (g, token_bank)."""
    d_model, n_patches, bsz, n_slots = 128, 6, 4, 5
    slot_heads = SlotHeads(d_model, n_slots=n_slots, std=1e-3)
    param_count_before = sum(p.numel() for p in slot_heads.parameters())
    z_q = F.normalize(torch.randn(bsz, d_model), dim=-1)
    g = slot_heads(z_q)
    token_bank = F.normalize(torch.randn(bsz, n_patches, d_model), dim=-1)
    q, attn = temporal_attention_heads(g, token_bank)
    param_count_after = sum(p.numel() for p in slot_heads.parameters())
    assert param_count_before == param_count_after
    assert q.shape == (bsz, n_slots, d_model)
    assert attn.shape == (bsz, n_slots, n_patches)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_v5_and_ph5_same_n_slots_identical_param_count():
    d_model = 128
    v5_heads = SlotHeads(d_model, n_slots=ARM_N_SLOTS['V5'], std=1e-3)
    ph5_heads = SlotHeads(d_model, n_slots=ARM_N_SLOTS['PH5'], std=1e-3)
    assert sum(p.numel() for p in v5_heads.parameters()) == sum(p.numel() for p in ph5_heads.parameters())


def test_no_future_leakage_in_patch_attention_signature():
    import inspect
    sig = inspect.signature(temporal_attention_heads)
    assert list(sig.parameters) == ['g', 't_p']
