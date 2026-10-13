"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- Quality-Aware dynamic fusion gate.

Two pieces:
  1. `compute_qa_features(...)` -- 7 common, inference-available-only
     features (no query true future `Y` is ever an argument here, by
     construction -- see `test_qa_gate_leakage01.py`).
  2. `QualityGate` -- the fixed architecture
     Linear(7,16) -> GELU -> Linear(16,1) -> Sigmoid, producing
     lambda(q,c) in [0,1], initialized so lambda ~= 0.5 at step 0 (same
     as NOQA's lambda_global = sigmoid(0) = 0.5), by zeroing the final
     layer's weight and bias.

Fusion (both arms): `Y_hat = B + lambda * (R - B)`. NOQA uses one
trainable scalar `lambda_global = sigmoid(a)`; QA uses
`lambda(q,c) = QualityGate(features(q,c))`. This module only ever
produces the QA side -- the NOQA global scalar is a single
`nn.Parameter` the training script owns directly.
"""
import torch
import torch.nn as nn

QA_FEATURE_NAMES = (
    'd_mae', 'd_rmse', 'd_cand_dispersion', 'entropy_norm',
    'top1_top2_margin', 'top10_mass', 'ess',
)
QA_FEATURE_DIM = len(QA_FEATURE_NAMES)


def compute_qa_features(base_forecast, soft_forecast, y_sel, weights, p_full, picks, valid_mask, eps=1e-12):
    """All arguments are quantities available at inference time, using
    only the query's own PAST (base_forecast, candidate probabilities)
    and the CANDIDATES' historical futures (y_sel, which are aligned
    past windows from the memory bank -- never the query's own true
    future). No argument here is the query's ground-truth future `Y`.

    base_forecast, soft_forecast: [B, T]
    y_sel: [B, K, T]           -- Top-K candidates' future values
    weights: [B, K]             -- renormalized Top-K weights (sum to 1)
    p_full: [B, N]              -- full candidate probability distribution
    picks: [B, K] long          -- indices of the Top-K into the N axis
    valid_mask: [B, N] bool      -- which of the N candidates are valid

    Returns features: [B, 7], order = QA_FEATURE_NAMES.
    """
    diff = base_forecast - soft_forecast
    d_mae = diff.abs().mean(dim=-1)
    d_rmse = (diff ** 2).mean(dim=-1).clamp_min(0).sqrt()

    dispersion_per_cand = ((y_sel - soft_forecast.unsqueeze(1)) ** 2).mean(dim=-1)  # [B, K]
    d_cand_dispersion = (weights * dispersion_per_cand).sum(dim=-1)

    p_valid = p_full.masked_fill(~valid_mask, 0.0)
    n_valid = valid_mask.sum(dim=-1).clamp_min(2).float()
    entropy = -(p_valid * (p_valid.clamp_min(eps)).log()).sum(dim=-1)
    entropy_norm = entropy / n_valid.log()

    top2_vals, _ = torch.topk(p_full.masked_fill(~valid_mask, float('-inf')), 2, dim=-1)
    top1_top2_margin = top2_vals[:, 0] - top2_vals[:, 1]

    top10_mass = p_full.gather(1, picks).sum(dim=-1)

    ess = 1.0 / (weights ** 2).sum(dim=-1).clamp_min(eps)

    feats = torch.stack(
        [d_mae, d_rmse, d_cand_dispersion, entropy_norm, top1_top2_margin, top10_mass, ess],
        dim=-1,
    )
    return feats


class TrainFeatureNormalizer:
    """Mean/std computed ONCE from TRAIN-split features and frozen.
    `fit` must be called only on TRAIN; `transform` is used identically
    for TRAIN/VAL/TEST afterward -- normalizing VAL/TEST with their own
    statistics is exactly the leakage this class exists to prevent."""

    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, features):
        with torch.no_grad():
            self.mean = features.mean(dim=0).clone()
            self.std = features.std(dim=0).clamp_min(1e-6).clone()
        self._fitted = True
        return self

    def to(self, device):
        """Move the fitted mean/std to `device` -- needed when `fit` was
        called on CPU-collected features (e.g. accumulated via `.cpu()`
        during a diagnostic pass) but `transform` is later called on
        GPU tensors during training."""
        if self._fitted:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

    def transform(self, features):
        if not self._fitted:
            raise RuntimeError('TrainFeatureNormalizer.transform called before fit() -- '
                                'must fit on TRAIN features first, never on VAL/TEST.')
        return (features - self.mean) / self.std

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, sd):
        self.mean = sd['mean']
        self.std = sd['std']
        self._fitted = True


class QualityGate(nn.Module):
    """Fixed architecture, no sweep: Linear(7,16) -> GELU -> Linear(16,1)
    -> Sigmoid. Final layer zero-initialized so lambda(q,c) = 0.5 for
    every input at step 0, matching NOQA's lambda_global = sigmoid(0)."""

    def __init__(self, input_dim=QA_FEATURE_DIM, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        logit = self.net(features).squeeze(-1)
        return torch.sigmoid(logit)


def validation_oracle_lambda(base_forecast, soft_forecast, query_future):
    """VALIDATION-ONLY diagnostic oracle (never used in training, never
    computed on TEST before the final method is locked):
    lambda*_oracle = clip((R-B)^T(Y-B) / ||R-B||^2, 0, 1), per (q,c)."""
    r_minus_b = soft_forecast - base_forecast
    y_minus_b = query_future - base_forecast
    num = (r_minus_b * y_minus_b).sum(dim=-1)
    den = (r_minus_b ** 2).sum(dim=-1).clamp_min(1e-12)
    return (num / den).clamp(0.0, 1.0)
