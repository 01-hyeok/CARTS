import torch

from models.RelationStage1 import stable_topk_indices
from utils.soft_topk_retrieval01 import (
    soft_topk_select_and_forecast, soft_topk_weighted_forecast, soft_topk_weights,
)


def test_weights_sum_to_one():
    torch.manual_seed(0)
    B, N, K = 4, 20, 10
    logits = torch.randn(B, N)
    p = torch.softmax(logits, dim=-1)
    picks = stable_topk_indices(p, K)
    w = soft_topk_weights(p, picks)
    assert w.shape == (B, K)
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-6)


def test_topk_index_matches_ordinary_topk():
    torch.manual_seed(1)
    B, N, K = 6, 30, 10
    logits = torch.randn(B, N)
    p = torch.softmax(logits, dim=-1)
    picks = stable_topk_indices(p, K)
    # Ordinary numpy-style top-k on the same p, independently computed.
    ref_vals, ref_idx = torch.sort(p, dim=-1, descending=True, stable=True)
    ref_picks = ref_idx[:, :K]
    assert torch.equal(torch.sort(picks, dim=-1).values, torch.sort(ref_picks, dim=-1).values)
    # The gathered probabilities at `picks` must be the K largest values of p.
    picked_vals = p.gather(1, picks)
    assert torch.allclose(torch.sort(picked_vals, dim=-1, descending=True).values,
                           torch.sort(ref_vals[:, :K], dim=-1, descending=True).values, atol=1e-7)


def test_uniform_topk_probability_matches_uniform_mean():
    torch.manual_seed(2)
    B, N, K, T = 3, 15, 5, 4
    # Construct p so the TOP-K entries are exactly equal (uniform within Top-K),
    # and strictly larger than everything else, so TopK selection is unambiguous.
    base = torch.rand(B, N) * 0.01
    p = base.clone()
    topk_val = 1.0  # large, will dominate after we fix the rest small
    chosen = torch.arange(K)
    p[:, chosen] = topk_val
    p = p / p.sum(dim=-1, keepdim=True)
    picks = stable_topk_indices(p, K)
    y_sel = torch.randn(B, K, T)
    r_soft, w = soft_topk_select_and_forecast(p, picks, y_sel)
    uniform_mean = y_sel.mean(dim=1)
    assert torch.allclose(w, torch.full((B, K), 1.0 / K), atol=1e-5)
    assert torch.allclose(r_soft, uniform_mean, atol=1e-5)


def test_gradient_flows_from_forecast_to_probability_logits():
    torch.manual_seed(3)
    B, N, K, T = 2, 12, 5, 3
    logits = torch.randn(B, N, requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    with torch.no_grad():
        picks = stable_topk_indices(p, K)
    y_sel = torch.randn(B, K, T)
    r, w = soft_topk_select_and_forecast(p, picks, y_sel)
    target = torch.randn(B, T)
    loss = ((r - target) ** 2).mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_weighted_forecast_matches_manual_formula():
    torch.manual_seed(4)
    B, N, K, T = 3, 20, 10, 6
    logits = torch.randn(B, N)
    p = torch.softmax(logits, dim=-1)
    picks = stable_topk_indices(p, K)
    y_sel = torch.randn(B, K, T)
    w = soft_topk_weights(p, picks)
    r = soft_topk_weighted_forecast(w, y_sel)
    manual = torch.zeros(B, T)
    for b in range(B):
        p_topk = p[b, picks[b]]
        w_manual = p_topk / p_topk.sum()
        manual[b] = (y_sel[b] * w_manual.unsqueeze(-1)).sum(dim=0)
    assert torch.allclose(r, manual, atol=1e-6)
