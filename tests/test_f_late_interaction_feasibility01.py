import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import RelationEncoder
from scripts.diag_f_late_interaction_feasibility01 import score_f1_aligned, score_f2_local_lse, score_f3_maxsim
from scripts.train_f_late_interaction_probe01 import LateInteractionHead


def _tiny_configs(seq_len=720, patch_len=120, d_model=16, pooling='cls'):
    return SimpleNamespace(
        relation_encoder_type='transformer', relation_pooling=pooling,
        relation_value_space='delta_last', relation_self_fill='zero',
        seq_len=seq_len, patch_len=patch_len, stride=patch_len,
        d_model=d_model, n_heads=2, e_layers=1, d_ff=32, dropout=0.0,
        relation_input_dim=1, retrieval_similarity='cosine',
    )


def _build_encoder():
    torch.manual_seed(0)
    cfg = _tiny_configs()
    enc = RelationEncoder(cfg)
    enc.eval()
    return enc


def test_forward_pooled_output_unchanged_by_return_tokens_flag():
    enc = _build_encoder()
    x = torch.randn(3, 1, 720)
    with torch.no_grad():
        z1 = enc(x)
        z2, tokens = enc(x, return_tokens=True)
    assert torch.equal(z1, z2), 'pooled output must be byte-identical whether return_tokens is requested or not'


def test_patch_token_count_matches_expected():
    enc = _build_encoder()
    x = torch.randn(2, 1, 720)
    with torch.no_grad():
        _, tokens = enc(x, return_tokens=True)
    expected_patches = 720 // 120
    assert tokens.shape[1] == expected_patches == 6


def test_token_output_shape_is_B_P_D():
    enc = _build_encoder()
    x = torch.randn(4, 1, 720)
    with torch.no_grad():
        z, tokens = enc(x, return_tokens=True)
    assert tokens.dim() == 3
    assert tokens.shape[0] == 4
    assert tokens.shape[2] == z.shape[-1]


def test_tokens_are_l2_normalized():
    enc = _build_encoder()
    x = torch.randn(3, 1, 720)
    with torch.no_grad():
        _, tokens = enc(x, return_tokens=True)
    norms = tokens.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_mean_pooling_returns_all_positions_as_tokens():
    torch.manual_seed(0)
    cfg = _tiny_configs(pooling='mean')
    enc = RelationEncoder(cfg)
    enc.eval()
    x = torch.randn(2, 1, 720)
    with torch.no_grad():
        _, tokens = enc(x, return_tokens=True)
    assert tokens.shape[1] == 6  # no CLS slot to exclude under mean pooling


def test_w0_local_lse_matches_aligned_ranking():
    torch.manual_seed(1)
    B, N, P, D = 2, 5, 4, 8
    tok_q = F.normalize(torch.randn(B, P, D), dim=-1)
    tok_i = F.normalize(torch.randn(N, P, D), dim=-1)
    s_aligned = score_f1_aligned(None, None, tok_q, tok_i)
    s_lse_w0 = score_f2_local_lse(None, None, tok_q, tok_i, w=0, lam=0.0, tau_a=0.1)
    # w=0 -> only j=p allowed -> LSE degenerates to sim[p,p]/tau_a, a positive
    # monotonic rescaling of s_aligned -> identical Top-K ordering.
    order_aligned = s_aligned.argsort(dim=-1)
    order_lse = s_lse_w0.argsort(dim=-1)
    assert torch.equal(order_aligned, order_lse)


def test_p1_identity_projection_equals_cosine():
    torch.manual_seed(2)
    B, N, D = 3, 4, 8
    tok_q = F.normalize(torch.randn(B, 1, D), dim=-1)
    tok_i = F.normalize(torch.randn(N, 1, D), dim=-1)
    s_aligned = score_f1_aligned(None, None, tok_q, tok_i)
    s_cosine = tok_q[:, 0, :] @ tok_i[:, 0, :].T
    assert torch.allclose(s_aligned, s_cosine, atol=1e-5)


def test_unordered_maxsim_invariant_to_candidate_patch_order_position_aware_is_not():
    torch.manual_seed(3)
    B, N, P, D = 2, 3, 4, 8
    tok_q = F.normalize(torch.randn(B, P, D), dim=-1)
    tok_i = F.normalize(torch.randn(N, P, D), dim=-1)
    perm = torch.randperm(P)
    tok_i_shuffled = tok_i[:, perm, :]

    s_maxsim = score_f3_maxsim(None, None, tok_q, tok_i)
    s_maxsim_shuf = score_f3_maxsim(None, None, tok_q, tok_i_shuffled)
    assert torch.allclose(s_maxsim, s_maxsim_shuf, atol=1e-6), 'MaxSim must be invariant to patch permutation'

    s_aligned = score_f1_aligned(None, None, tok_q, tok_i)
    s_aligned_shuf = score_f1_aligned(None, None, tok_q, tok_i_shuffled)
    assert not torch.allclose(s_aligned, s_aligned_shuf, atol=1e-4), \
        'position-aware aligned score must change when candidate patch order is shuffled'


def test_frozen_trunk_no_gradient_head_has_gradient():
    torch.manual_seed(4)
    head = LateInteractionHead(d_model=8, w=1)
    tok_q_raw = torch.randn(2, 3, 8, requires_grad=False)
    tok_i_raw = torch.randn(4, 3, 8, requires_grad=False)
    z_q = head.project_q(tok_q_raw)
    z_i = head.project_k(tok_i_raw)
    s = head.score(z_q, z_i)
    s.sum().backward()
    assert head.w_q.weight.grad is not None and head.w_q.weight.grad.abs().sum() > 0
    assert head.w_k.weight.grad is not None and head.w_k.weight.grad.abs().sum() > 0
    assert not tok_q_raw.requires_grad and not tok_i_raw.requires_grad


def test_head_deterministic_given_seed():
    torch.manual_seed(7)
    h1 = LateInteractionHead(d_model=8, w=1)
    torch.manual_seed(7)
    h2 = LateInteractionHead(d_model=8, w=1)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)


def test_local_lse_no_nan_or_inf():
    torch.manual_seed(5)
    B, N, P, D = 2, 6, 6, 8
    tok_q = F.normalize(torch.randn(B, P, D), dim=-1)
    tok_i = F.normalize(torch.randn(N, P, D), dim=-1)
    for w in (0, 1, 2):
        s = score_f2_local_lse(None, None, tok_q, tok_i, w=w, lam=0.1, tau_a=0.05)
        assert torch.isfinite(s).all()


def test_individual_utility_memsafe_signature_has_no_scorer_or_encoder_arg():
    """Structural check that the teacher distance function this script
    reuses cannot see any scorer/encoder object -- it only takes
    memory/offset/query tensors and a chunk size."""
    import inspect
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    params = list(inspect.signature(individual_utility_memsafe).parameters)
    assert params == ['memory_c', 'offset_c', 'query_future', 'chunk_size']
