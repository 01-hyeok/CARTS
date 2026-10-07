"""TRACK-V-MEANMIX-INFERENCE01 -- required unit tests (spec section 10).
Tests 1-7 use synthetic tensors; tests 8-9 are integration tests
against the real ETTh1_96 checkpoints and live in
tests/test_v_meanmix_cache01.py (checkpoint-loading smoke tests)."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


def _toy(bsz=4, s=5, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(bsz, s, n, generator=g)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    return scores, cand_mask


# -------------------- test 1: S=1 mean-mixture == round-robin --------------------

def test_s1_meanmix_equals_round_robin():
    scores, cand_mask = _toy(s=1)
    rr_idx = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
    mm_idx, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    assert torch.equal(rr_idx.sort(dim=-1).values, mm_idx.sort(dim=-1).values)


# -------------------- test 2: V0-style single-head score, ordinary Top10 == mean-mixture --------------------

def test_v0_single_head_ordinary_top10_equals_meanmix():
    bsz, n = 3, 25
    g = torch.Generator().manual_seed(1)
    s_flat = torch.randn(bsz, n, generator=g)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    ordinary_idx = stable_topk_indices_ref(s_flat, cand_mask, TOP_K)
    mm_idx, _, _ = mean_mixture_topk_selection(s_flat.unsqueeze(1), cand_mask, tau_s=0.1, k=TOP_K)
    assert torch.equal(ordinary_idx.sort(dim=-1).values, mm_idx.sort(dim=-1).values)


def stable_topk_indices_ref(scores, cand_mask, k):
    from models.RelationStage1 import stable_topk_indices
    return stable_topk_indices(scores.masked_fill(~cand_mask, float('-inf')), k, largest=True)


# -------------------- test 3/4: probabilities sum to 1 over valid candidates --------------------

def test_per_head_probability_sums_to_one():
    scores, cand_mask = _toy()
    cand_mask[:, -5:] = False  # some invalid candidates too
    _, p_heads, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    valid_sum = (p_heads * cand_mask.unsqueeze(1).float()).sum(dim=-1)
    assert torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-5)


def test_mix_probability_sums_to_one():
    scores, cand_mask = _toy()
    cand_mask[:, -5:] = False
    _, _, p_mix = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    valid_sum = (p_mix * cand_mask.float()).sum(dim=-1)
    assert torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-5)


# -------------------- test 5: masked candidates never selected --------------------

def test_masked_candidates_never_selected():
    scores, cand_mask = _toy(bsz=2, s=3, n=20)
    cand_mask[0, :15] = False  # only 5 valid for query 0
    cand_mask[1, :12] = False  # only 8 valid for query 1
    picks, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=5)
    for b in range(2):
        valid_set = set(torch.nonzero(cand_mask[b]).flatten().tolist())
        assert set(picks[b].tolist()).issubset(valid_set)


# -------------------- test 6: deterministic --------------------

def test_deterministic_selection():
    scores, cand_mask = _toy()
    picks1, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    picks2, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    assert torch.equal(picks1, picks2)


# -------------------- test 8: selection never touches batch_y/query future --------------------

def test_selection_signature_has_no_future_inputs():
    import inspect
    sig = inspect.signature(mean_mixture_topk_selection)
    params = set(sig.parameters)
    assert params == {'scores', 'cand_mask', 'tau_s', 'k'}, \
        f'selection must only take past-score + mask inputs, got {params}'


# -------------------- V2/V5 sanity: mean-mixture and round-robin generally differ --------------------

def test_v5_meanmix_generally_differs_from_round_robin():
    scores, cand_mask = _toy(bsz=8, s=5, n=50, seed=7)
    rr_idx = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
    mm_idx, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    overlap = torch.tensor([
        len(set(rr_idx[b].tolist()) & set(mm_idx[b].tolist())) / TOP_K for b in range(8)
    ])
    assert overlap.mean() < 1.0, 'round-robin and mean-mixture should generally pick different sets for S=5'


def test_no_nan_or_inf_in_outputs():
    scores, cand_mask = _toy()
    picks, p_heads, p_mix = mean_mixture_topk_selection(scores, cand_mask, tau_s=0.1, k=TOP_K)
    assert torch.isfinite(p_heads[cand_mask.unsqueeze(1).expand_as(p_heads)]).all()
    assert torch.isfinite(p_mix[cand_mask]).all()
    assert torch.isfinite(picks.float()).all()
