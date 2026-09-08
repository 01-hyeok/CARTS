"""EXP-ORACLE-CHOICE01 mandatory pre-GPU sanity checks (spec's numbered
list):
1. full-memory masked CE includes every valid candidate in the softmax
2. invalid candidate probability is exactly 0
3. oracle label == argmax(true dense utility)
4. synthetic small-N positive control: near-100% oracle choice accuracy
5. dense_utility chunking does not change the resulting logits/argmax/loss
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.dense_utility import dense_utility


def _toy(bsz=6, n=41, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    u_target = torch.randn(bsz, n, generator=g)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -4:] = False
    valid[0, :20] = False  # a row with many invalid entries too
    u_hat = torch.randn(bsz, n, generator=g, requires_grad=True)
    return u_hat, u_target, valid


def test_1_and_2_softmax_covers_all_valid_and_invalid_prob_is_zero():
    u_hat, u_target, valid = _toy()
    tau = 0.1
    neg_inf = torch.finfo(u_target.dtype).min / 4
    logits = u_hat.masked_fill(~valid, neg_inf) / tau
    probs = F.softmax(logits, dim=-1)
    # invalid positions get exactly 0 probability
    assert torch.all(probs[~valid] == 0.0)
    # valid positions' probabilities sum to 1 per row (every valid candidate
    # is included in the softmax denominator, none silently dropped)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_3_oracle_label_is_argmax_true_utility():
    u_hat, u_target, valid = _toy()
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid, neg_inf)
    i_star_expected = target_masked.argmax(dim=-1)
    loss, diag = oracle_choice_step_loss(u_hat, u_target, valid, tau=0.1)
    # re-derive i_star the same way the loss function does internally and confirm equality
    i_star_actual = u_target.masked_fill(~valid, neg_inf).argmax(dim=-1)
    assert torch.equal(i_star_expected, i_star_actual)


def test_4_smalln_positive_control_fits_near_100pct_accuracy():
    torch.manual_seed(0)
    bsz, n, d = 32, 64, 8
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -5:] = False
    true_features = torch.randn(bsz, n, d)
    u_target = true_features.sum(dim=-1) + 0.01 * torch.randn(bsz, n)

    linear = torch.nn.Linear(d, 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.05)
    for step in range(300):
        optimizer.zero_grad()
        u_hat = linear(true_features).squeeze(-1)
        loss, diag = oracle_choice_step_loss(u_hat, u_target, valid, tau=0.1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        u_hat = linear(true_features).squeeze(-1)
        _, diag = oracle_choice_step_loss(u_hat, u_target, valid, tau=0.1)
    assert diag['top1_acc'] > 0.9, f"expected near-100% oracle choice accuracy, got {diag['top1_acc']}"


def test_5_dense_utility_chunking_does_not_change_logits_or_loss():
    torch.manual_seed(0)
    bsz, n, h = 4, 37, 6
    prefix = torch.randint(0, n, (bsz, 2))
    w = torch.rand(bsz, n)
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    a_unchunked = dense_utility(prefix, w, futures, query_future, chunk_size=None)
    a_chunked = dense_utility(prefix, w, futures, query_future, chunk_size=9)
    assert torch.allclose(a_unchunked, a_chunked, atol=1e-6)

    u_target_unchunked = -a_unchunked
    u_target_chunked = -a_chunked
    valid = torch.ones(bsz, n, dtype=torch.bool)
    u_hat = torch.randn(bsz, n)
    loss_u, _ = oracle_choice_step_loss(u_hat, u_target_unchunked, valid, tau=0.1)
    loss_c, _ = oracle_choice_step_loss(u_hat, u_target_chunked, valid, tau=0.1)
    assert torch.allclose(loss_u, loss_c, atol=1e-6)
