"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- shared NOQA/QA fusion step used
by every method's (V0/V1/V2/V5/Hard/Joint) end-to-end SoftTopK trainer.
Each method script builds its own `p_full`/`picks`/`y_sel` (canonical,
method-specific retrieval scoring -- never reimplemented here) and then
calls `fuse` to get the final forecast + the quantities needed for the
combined loss and QA diagnostics. This file owns ONLY the part that is
identical across all six methods: SoftTopK aggregation, NOQA/QA fusion,
and the combined loss.
"""
import torch
import torch.nn as nn

from utils.quality_aware_gate01 import QualityGate, compute_qa_features
from utils.soft_topk_retrieval01 import soft_topk_select_and_forecast


class GlobalLambda(nn.Module):
    """NOQA arm: one trainable scalar lambda_global = sigmoid(a), a=0 at
    init (lambda_global = 0.5), same as the Original CARTS Trainable
    Global Lambda Stage-2 convention used throughout this project."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))

    def forward(self, batch_size):
        lam = torch.sigmoid(self.a)
        return lam.expand(batch_size)


def soft_topk_forecast(p_full, picks, y_sel):
    """Shared Soft Top-K aggregation step. Returns (R, weights)."""
    return soft_topk_select_and_forecast(p_full, picks, y_sel)


def fuse_noqa(base_forecast, soft_forecast, global_lambda):
    """base_forecast, soft_forecast: [B, T]. global_lambda: GlobalLambda
    module. Returns (y_hat, lambda_used [B])."""
    lam = global_lambda(base_forecast.size(0))
    y_hat = base_forecast + lam.unsqueeze(-1) * (soft_forecast - base_forecast)
    return y_hat, lam


def fuse_qa(base_forecast, soft_forecast, y_sel, weights, p_full, picks, valid_mask, gate, normalizer):
    """QA arm: computes the 7 common quality features (inference-only,
    no query future), TRAIN-normalizes them, runs the QualityGate, and
    fuses. Returns (y_hat, lambda_used [B], raw_features [B,7])."""
    feats = compute_qa_features(base_forecast, soft_forecast, y_sel, weights, p_full, picks, valid_mask)
    feats_norm = normalizer.transform(feats)
    lam = gate(feats_norm)
    y_hat = base_forecast + lam.unsqueeze(-1) * (soft_forecast - base_forecast)
    return y_hat, lam, feats


def combined_loss(y_hat, y, retrieval_loss, gamma=1.0):
    """L_total = L_forecast + gamma * L_retrieval. gamma is fixed at 1.0
    for this track (no sweep) -- see spec section 11-1."""
    l_forecast = ((y_hat - y) ** 2).mean()
    l_total = l_forecast + gamma * retrieval_loss
    return l_total, l_forecast
