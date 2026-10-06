"""Unit tests for `utils/candidate_pool.py` -- the common full/coarse-topk
candidate-pool abstraction. Covers spec items 2, 3, 4, 6, 7, 10, 15, 16,
17 of the candidate-pool refactor (the rest are covered in
`tests/test_retriever_pool01.py`, which needs the trainer wiring too)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.candidate_pool import (
    CandidatePoolConfig, build_coarse_pool, compute_coarse_delta_last_cosine_scores,
    encode_pooled_candidates, gather_candidate_histories, gather_candidate_values,
    local_to_global, pooled_future_mse,
)


def test_config_defaults_preserve_legacy_behavior():
    cfg = CandidatePoolConfig()
    assert cfg.mode == 'full'
    assert cfg.size == 100
    assert cfg.metric == 'delta_last_cosine'


def test_config_rejects_unsupported_mode_and_metric():
    with pytest.raises(ValueError):
        CandidatePoolConfig(mode='bogus')
    with pytest.raises(ValueError):
        CandidatePoolConfig(metric='bogus')
    with pytest.raises(ValueError):
        CandidatePoolConfig(mode='coarse_topk', size=0)


# -------------------- item 6: hand-computable delta-last cosine --------------------

def test_delta_last_cosine_hand_computable():
    # one query, two candidates, L=3, single channel slot (channel 0 of a C=1 tensor)
    batch_x = torch.tensor([[[1.0], [2.0], [4.0]]])         # [B=1, L=3, C=1], delta-last from 4 -> [-3,-2,0]
    memory_x = torch.tensor([
        [[1.0], [2.0], [4.0]],    # identical to query -> delta [-3,-2,0], cosine with itself = 1.0
        [[0.0], [0.0], [0.0]],    # delta [0,0,0] -> degenerate (zero vector)
    ])
    scores = compute_coarse_delta_last_cosine_scores(batch_x, memory_x, channel=0)
    assert scores.shape == (1, 2)
    assert torch.isclose(scores[0, 0], torch.tensor(1.0), atol=1e-5)


def test_delta_last_cosine_matches_manual_normalize():
    torch.manual_seed(0)
    B, L, N, C = 3, 5, 7, 2
    batch_x = torch.randn(B, L, C)
    memory_x = torch.randn(N, L, C)
    ch = 1
    scores = compute_coarse_delta_last_cosine_scores(batch_x, memory_x, channel=ch)
    q = batch_x[:, :, ch]
    k = memory_x[:, :, ch]
    q_delta = q - q[:, -1:]
    k_delta = k - k[:, -1:]
    expected = F.normalize(q_delta, dim=-1) @ F.normalize(k_delta, dim=-1).T
    assert torch.allclose(scores, expected, atol=1e-6)


# -------------------- item 7: future-blindness --------------------

def test_coarse_score_function_has_no_future_argument():
    import inspect
    sig = inspect.signature(compute_coarse_delta_last_cosine_scores)
    names = list(sig.parameters)
    assert names == ['batch_x', 'memory_x', 'channel']
    for forbidden in ('batch_y', 'memory_y', 'query_future'):
        assert forbidden not in names


# -------------------- item 2/3: pool size == M, all pool candidates valid --------------------

def test_pool_size_is_exactly_m_and_all_valid():
    torch.manual_seed(1)
    B, N, M = 4, 50, 10
    scores = torch.randn(B, N)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    cand_mask[:, :5] = False  # 45 valid per query, plenty >= M
    pool = build_coarse_pool(scores, cand_mask, M)
    assert pool.shape == (B, M)
    for b in range(B):
        assert cand_mask[b, pool[b]].all()


def test_pool_valid_count_too_small_raises():
    B, N, M = 2, 20, 15
    scores = torch.randn(B, N)
    cand_mask = torch.zeros(B, N, dtype=torch.bool)
    cand_mask[:, :10] = True  # only 10 valid, asking for 15
    with pytest.raises(ValueError):
        build_coarse_pool(scores, cand_mask, M)


# -------------------- item 4: determinism --------------------

def test_pool_is_deterministic():
    torch.manual_seed(2)
    B, N, M = 3, 30, 8
    scores = torch.randn(B, N)
    cand_mask = torch.ones(B, N, dtype=torch.bool)
    pool1 = build_coarse_pool(scores, cand_mask, M)
    pool2 = build_coarse_pool(scores, cand_mask, M)
    assert torch.equal(pool1, pool2)


# -------------------- item 10: local -> global mapping correctness --------------------

def test_local_to_global_mapping():
    pool_idx_global = torch.tensor([[7, 3, 9, 1], [2, 8, 0, 5]])  # [B=2, M=4]
    local_idx = torch.tensor([[2, 0], [3, 1]])                    # [B=2, K=2], values in [0,4)
    global_idx = local_to_global(pool_idx_global, local_idx)
    assert torch.equal(global_idx, torch.tensor([[9, 7], [5, 8]]))


# -------------------- gather helpers: shape + content correctness --------------------

def test_gather_candidate_histories_shape_and_content():
    N, L, C = 20, 6, 3
    memory_x = torch.arange(N * L * C, dtype=torch.float32).reshape(N, L, C)
    pool_idx_global = torch.tensor([[0, 5, 19], [2, 2, 7]])  # [B=2, M=3], duplicates allowed in this toy test
    gathered = gather_candidate_histories(memory_x, pool_idx_global)
    assert gathered.shape == (2, 3, L, C)
    assert torch.equal(gathered[0, 0], memory_x[0])
    assert torch.equal(gathered[0, 2], memory_x[19])
    assert torch.equal(gathered[1, 1], memory_x[2])


def test_gather_candidate_values_and_pooled_future_mse():
    N, H = 10, 4
    memory_c = torch.arange(N * H, dtype=torch.float32).reshape(N, H)
    pool_idx_global = torch.tensor([[0, 1], [2, 3]])  # [B=2, M=2]
    pooled = gather_candidate_values(memory_c, pool_idx_global)
    assert pooled.shape == (2, 2, H)
    assert torch.equal(pooled[0, 0], memory_c[0])
    assert torch.equal(pooled[1, 1], memory_c[3])

    offset_c = torch.zeros(2)
    query_future = pooled[:, 0, :].clone()  # make candidate 0 of each query a perfect match
    d_pool = pooled_future_mse(pooled, offset_c, query_future)
    assert d_pool.shape == (2, 2)
    assert torch.allclose(d_pool[:, 0], torch.zeros(2), atol=1e-6)
    assert (d_pool[:, 1] > 0).all()


# -------------------- item 8: encoder sees exactly B*M candidates, not B*N --------------------

def test_encode_pooled_candidates_calls_encoder_with_exactly_bm_rows():
    B, M, L, C, D = 2, 5, 4, 3, 8
    pooled_x = torch.randn(B, M, L, C)
    calls = []

    def fake_encode(x, channel):
        calls.append(x.shape)
        return torch.randn(x.size(0), D)

    z = encode_pooled_candidates(fake_encode, pooled_x, channel=0)
    assert z.shape == (B, M, D)
    assert len(calls) == 1
    assert calls[0] == (B * M, L, C)  # NOT (B*N, L, C) -- the whole point of this module
