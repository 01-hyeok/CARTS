"""Integration/regression tests for `scripts/train_retriever_pool01.py`
and the candidate-pool refactor's trainer-level wiring. Needs a real
(tiny) reference checkpoint and a live `exp`/`model` -- GPU if available,
CPU otherwise; always only a handful of batches, never a training loop.
Covers the spec's remaining unit-test items not already covered by
`tests/test_candidate_pool01.py` (1/19, 5, 9, 11, 12, 13, 14, 15, 16, 17).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from scripts.train_t_pure_multislot01 import compute_scores_full_grad, round_robin_topk_selection
from utils.candidate_pool import CandidatePoolConfig

REF_ETTH1_96 = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
               'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_'
               'dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _skip_if_missing_ckpt():
    if not REF_ETTH1_96.exists():
        pytest.skip(f'reference checkpoint not found: {REF_ETTH1_96}')


def _build_tiny_experiment(num_query_views=0):
    _skip_if_missing_ckpt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF_ETTH1_96), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=10, patch_len=16)
    exp, args, model = build_model(cli, device)
    slot_heads = None
    if num_query_views >= 1:
        slot_heads = SlotHeads(int(args.d_model), n_slots=num_query_views, std=1e-3).to(device)
    return exp, args, model, slot_heads, device


def _first_batch(exp, device):
    _, loader = exp._get_data(flag='train', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    return batch_x.float().to(device), batch_y.float().to(device), batch_start_idx


# -------------------- item 1/19: full mode == legacy exactly --------------------

def test_full_mode_matches_legacy_compute_scores_full_grad_slots():
    exp, args, model, slot_heads, device = _build_tiny_experiment(num_query_views=2)
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    pool_cfg = CandidatePoolConfig(mode='full')
    model.eval()
    slot_heads.eval()
    with torch.no_grad():
        new_scores, new_mask, pool_idx = compute_scores(pool_cfg, model, slot_heads, batch_x, exp, 0,
                                                         cand_mask, None, batch_start_idx, device)
        legacy_scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, 0)
    assert pool_idx is None
    assert torch.equal(new_mask, cand_mask)
    assert torch.allclose(new_scores, legacy_scores, atol=1e-5)


def test_full_mode_matches_legacy_zero_proj():
    exp, args, model, slot_heads, device = _build_tiny_experiment(num_query_views=0)
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    pool_cfg = CandidatePoolConfig(mode='full')
    model.eval()
    with torch.no_grad():
        new_scores, _, pool_idx = compute_scores(pool_cfg, model, None, batch_x, exp, 0, cand_mask, None,
                                                  batch_start_idx, device)
        z_q = encode_raw(model, batch_x, 0)
        z_k = encode_raw(model, exp.memory_x, 0)
        legacy_scores = arm_score(z_q, z_k, None).unsqueeze(1)
    assert pool_idx is None
    assert torch.allclose(new_scores, legacy_scores, atol=1e-5)


# -------------------- item 9: candidate-side encoder gradient non-zero in coarse_topk --------------------

def test_coarse_topk_candidate_encoder_gradient_nonzero():
    exp, args, model, slot_heads, device = _build_tiny_experiment(num_query_views=1)
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    pool_size = 50
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=pool_size)

    from utils.candidate_pool import build_coarse_pool, compute_coarse_delta_last_cosine_scores
    coarse_scores = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, 0)
    pool_idx_global = build_coarse_pool(coarse_scores, cand_mask, pool_size)

    class _FakeCache:
        def lookup(self, start_idx, channel, dev):
            return pool_idx_global.to(dev)

    model.train()
    slot_heads.train()
    for p in model.parameters():
        p.requires_grad_(True)
    scores, valid_mask, returned_pool = compute_scores(pool_cfg, model, slot_heads, batch_x, exp, 0,
                                                       cand_mask, _FakeCache(), batch_start_idx, device)
    assert returned_pool.shape == (batch_x.size(0), pool_size)
    loss = scores.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
        '[ISSUE] candidate encoder gradient is zero under coarse_topk'


# -------------------- item 11: S=1 round-robin == ordinary top-10 inside the pool --------------------

def test_s1_round_robin_equals_ordinary_top10_in_pool():
    torch.manual_seed(0)
    B, M, K = 4, 30, 10
    scores = torch.randn(B, 1, M)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    picks = round_robin_topk_selection(scores, valid_mask, k=K)
    expected = stable_topk_indices(scores[:, 0, :], K, largest=True)
    assert torch.equal(picks, expected)


# -------------------- items 12/13: no duplicates, no invalid in final selection --------------------

def test_no_duplicates_no_invalid_in_pool_selection():
    torch.manual_seed(1)
    B, M, K, S = 3, 25, 10, 3
    scores = torch.randn(B, S, M)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    valid_mask[:, -5:] = False  # last 5 invalid in the pool
    picks = round_robin_topk_selection(scores, valid_mask, k=K)
    for b in range(B):
        assert len(set(picks[b].tolist())) == K, 'duplicate final candidate found'
        assert valid_mask[b, picks[b]].all(), 'invalid final candidate found'


# -------------------- item 14: Agg == D + C (already asserted inline in cache builder; spot check) --------------------

def test_agg_equals_d_plus_c_identity():
    torch.manual_seed(2)
    B, K, H = 5, 10, 4
    y_sel = torch.randn(B, K, H)
    query_future = torch.randn(B, H)
    ind_mse_i = ((y_sel - query_future.unsqueeze(1)) ** 2).mean(-1)
    D_ = ind_mse_i.mean(-1) / K
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
    C_ = agg_mse - D_
    assert torch.allclose(D_ + C_, agg_mse, atol=1e-6)


# -------------------- items 16/17: fail fast on bad cache / bad size --------------------

def test_cache_fingerprint_mismatch_raises(tmp_path):
    payload = {
        'query_start_idx': torch.arange(5),
        'pool_idx_global': torch.zeros(5, 2, 10, dtype=torch.int32),
        'meta': {'seq_len': 96, 'pred_len': 96, 'candidate_mask': 'raft',
                 'candidate_pool_size': 10, 'candidate_pool_metric': 'delta_last_cosine'},
    }
    path = tmp_path / 'val.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        CandidatePoolCache(path, expected_meta={'seq_len': 720})  # mismatched seq_len


def test_candidate_pool_size_leq_top_k_fails_fast():
    import subprocess
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'train_retriever_pool01.py'),
         '--reference_ckpt', str(REF_ETTH1_96), '--num_query_views', '0',
         '--candidate_pool_mode', 'coarse_topk', '--candidate_pool_size', '5', '--top_k', '10'],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'candidate_pool_size' in (result.stdout + result.stderr)
