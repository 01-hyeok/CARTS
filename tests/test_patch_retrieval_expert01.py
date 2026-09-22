"""TRACK-A-PATCH-RETRIEVAL-EXPERT01 sanity tests."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.relation_patch_embed import RelationPatchEmbedding
from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k


# ---- non-overlap patch sizing: num_patches == seq_len / patch_len exactly
# for the three spec-mandated sizes (24, 48, 120) and the native (16) ----
@pytest.mark.parametrize('patch_len', [16, 24, 48, 120])
def test_nonoverlap_patch_count_matches_seq_len_division(patch_len):
    seq_len = 720
    assert seq_len % patch_len == 0
    embed = RelationPatchEmbedding(seq_len=seq_len, patch_len=patch_len, stride=patch_len,
                                   d_model=8, dropout=0.0)
    assert embed.num_patches == seq_len // patch_len
    x = torch.randn(2, 1, seq_len)
    out = embed(x)
    assert out.shape == (2, embed.num_patches, 8)


# ---- recall@k: perfect overlap -> 1.0, disjoint -> 0.0, partial -> exact fraction ----
def test_recall_at_k_known_cases():
    model_idx = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
    oracle_same = model_idx.clone()
    assert torch.allclose(recall_at_k(model_idx, oracle_same, 5), torch.ones(2))

    oracle_disjoint = torch.tensor([[5, 6, 7, 8, 9], [20, 21, 22, 23, 24]])
    assert torch.allclose(recall_at_k(model_idx, oracle_disjoint, 5), torch.zeros(2))

    oracle_partial = torch.tensor([[0, 1, 8, 9, 7], [10, 11, 20, 21, 22]])
    got = recall_at_k(model_idx, oracle_partial, 5)
    assert torch.allclose(got, torch.tensor([2 / 5, 2 / 5]))


# ---- ndcg@k: model order == ideal (ascending distance) order -> ndcg == 1.0 ----
def test_ndcg_at_k_perfect_ranking_is_one():
    torch.manual_seed(0)
    bsz, n, k = 3, 20, 10
    d = torch.rand(bsz, n)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    model_idx = stable_topk_indices(d, k, largest=False)  # perfect: lowest distance first
    ndcg = ndcg_at_k(model_idx, d, valid, k)
    assert torch.allclose(ndcg, torch.ones(bsz), atol=1e-5)


# ---- ndcg@k: worst possible order (reversed) scores strictly below perfect ----
def test_ndcg_at_k_worst_order_is_lower():
    torch.manual_seed(1)
    bsz, n, k = 3, 20, 10
    d = torch.rand(bsz, n)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    best_idx = stable_topk_indices(d, k, largest=False)
    worst_idx = stable_topk_indices(d, k, largest=True)
    ndcg_best = ndcg_at_k(best_idx, d, valid, k)
    ndcg_worst = ndcg_at_k(worst_idx, d, valid, k)
    assert (ndcg_worst <= ndcg_best + 1e-6).all()
    assert (ndcg_worst < ndcg_best - 1e-4).any()


# ---- candidate-side gradient: E = encode_raw(model, memory_x, c) must NOT
# be detached -- gradients from the candidate side must reach the encoder,
# even when the query side is fixed. Isolated by freezing z_q via detach and
# checking model.parameters() still receive nonzero grad through E alone.
class _TinyEncoder(torch.nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.proj = torch.nn.Linear(4, d_model)

    def _relation_tensor(self, x, c, c2):
        return x  # x already [B, 4] in this isolated test

    def encoder(self, relation_x):
        return self.proj(relation_x)


def test_candidate_side_gradient_flows():
    torch.manual_seed(2)
    model = _TinyEncoder()
    x_q = torch.randn(3, 4)
    x_cand = torch.randn(5, 4)

    z_q = encode_raw(model, x_q, c=0).detach()  # freeze query side entirely
    E = encode_raw(model, x_cand, c=0)  # NOT detached -- this is what the trainer does
    assert E.requires_grad

    s = arm_score(z_q, E, None)
    loss = s.sum()
    loss.backward()

    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0, 'candidate-side encoding produced zero gradient -- E must not be detached/cached'
