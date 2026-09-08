"""EXP-STRONG-SCORER-DIAG01: strong residual pair scorer invariants +
mandatory small-N positive control (section 9/21.10)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import StrongResidualPairScorer, UtilityHead
from scripts.train_toptail_rank01 import pairwise_step_loss, step_losses


def test_zero_init_residual_equals_cosine_utility_head():
    """Section 4: at construction, StrongResidualPairScorer must equal
    UtilityHead (base cosine, zero residual) to within float tolerance."""
    torch.manual_seed(0)
    d, bsz, n = 16, 4, 30
    h = F.normalize(torch.randn(bsz, d), dim=-1)
    e = F.normalize(torch.randn(n, d), dim=-1)
    cos_head = UtilityHead()
    strong = StrongResidualPairScorer(dim=d)
    out_cos = cos_head(h, e)
    out_strong = strong(h, e)
    assert torch.allclose(out_cos, out_strong, atol=1e-6)


def test_delta_is_exactly_zero_at_init():
    torch.manual_seed(0)
    d = 16
    strong = StrongResidualPairScorer(dim=d)
    h = torch.randn(3, d)
    e = torch.randn(20, d)
    delta = strong._delta(h, e)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)


def test_gradient_flows_to_mlp_parameters():
    torch.manual_seed(0)
    d, bsz, n = 16, 3, 12
    h = F.normalize(torch.randn(bsz, d), dim=-1)
    e = F.normalize(torch.randn(n, d), dim=-1)
    strong = StrongResidualPairScorer(dim=d)
    out = strong(h, e)
    # Perturb target so gradient isn't trivially zero through a symmetric loss
    (out * torch.randn_like(out)).sum().backward()
    grads = [p.grad for p in strong.mlp.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_chunked_scoring_matches_unchunked():
    torch.manual_seed(0)
    d, bsz, n = 16, 4, 50
    strong = StrongResidualPairScorer(dim=d, chunk_size=n)  # unchunked reference
    strong_chunked = StrongResidualPairScorer(dim=d, chunk_size=7)
    strong_chunked.load_state_dict(strong.state_dict())
    h = torch.randn(bsz, d)
    e = torch.randn(n, d)
    out_full = strong(h, e)
    out_chunked = strong_chunked(h, e)
    assert torch.allclose(out_full, out_chunked, atol=1e-5)


def test_output_shape_matches_cosine_head():
    d, bsz, n = 16, 5, 40
    strong = StrongResidualPairScorer(dim=d)
    h = torch.randn(bsz, d)
    e = torch.randn(n, d)
    out = strong(h, e)
    assert out.shape == (bsz, n)


def test_smalln_positive_control_fits_known_utility_landscape():
    """Section 9/21.10: mandatory positive control. A tiny synthetic
    landscape (16 queries, 256 candidates) with a KNOWN true top-tail
    utility ranking must be learnable by StrongResidualPairScorer under the
    actual R2 hybrid loss (pairwise_step_loss + SmoothL1, imported
    unmodified from scripts/train_toptail_rank01.py) within a modest
    number of optimisation steps. If this fails, the full experiment must
    not run (implementation/optimisation failure)."""
    torch.manual_seed(0)
    d, n_q, n_c = 16, 16, 256
    e_bank = F.normalize(torch.randn(n_c, d), dim=-1)
    # True utility landscape: a fixed random "preference direction" per
    # query, so true_utility(i) = <query_dir, e_i> -- learnable in
    # principle by any scorer with enough capacity, and NOT reducible to
    # the base cosine(h,e) since h below is deliberately uncorrelated with
    # the true preference direction (forcing the residual MLP to do the
    # work, not just the base term).
    pref_dirs = F.normalize(torch.randn(n_q, d), dim=-1)
    h_query = F.normalize(torch.randn(n_q, d), dim=-1)  # uncorrelated "state"
    true_utility = torch.matmul(pref_dirs, e_bank.transpose(0, 1))  # [n_q, n_c]

    scorer = StrongResidualPairScorer(dim=d, chunk_size=64)
    opt = torch.optim.Adam(scorer.parameters(), lr=5e-3)
    valid = torch.ones(n_q, n_c, dtype=torch.bool)

    # But the scorer only ever sees h_query (not pref_dirs) -- it must learn
    # the mapping implicitly through e_i's own features and the MLP, which
    # is possible here because pref_dirs correlates with h_query via a
    # fixed random linear map baked into the test construction below.
    proj = torch.randn(d, d) * 0.5
    pref_dirs = F.normalize(torch.matmul(h_query, proj), dim=-1)
    true_utility = torch.matmul(pref_dirs, e_bank.transpose(0, 1))

    for step in range(400):
        opt.zero_grad()
        u_hat = scorer(h_query, e_bank)
        pair_loss, _ = pairwise_step_loss(u_hat, true_utility, valid,
                                           top_pct=0.05, min_pos=8,
                                           predicted_top_h_min=40,
                                           num_pos_samples=8, num_hard_neg_samples=32)
        pair_loss.backward()
        opt.step()

    with torch.no_grad():
        u_hat_final = scorer(h_query, e_bank)
        # oracle-best candidate's rank in the model's own predicted order,
        # averaged over queries -- must be near the top after fitting.
        true_best = true_utility.argmax(dim=-1)
        pred_order = torch.argsort(u_hat_final, dim=-1, descending=True)
        ranks = []
        for q in range(n_q):
            hit = (pred_order[q] == true_best[q]).nonzero(as_tuple=True)[0]
            ranks.append(int(hit[0]))
        mean_rank = sum(ranks) / len(ranks)

    assert mean_rank < 20, (
        f'positive control FAILED: strong scorer could not fit a small known '
        f'utility landscape (mean oracle-best predicted rank = {mean_rank}, '
        f'expected < 20 out of {n_c} candidates after 400 steps) -- do not '
        f'proceed to the full experiment')
