"""EXP-SEQDIAG01: Frozen-B0 control arm + metric/chance-baseline definition
tests. Complements tests/test_exp_seqfull01.py (masking, duplicate/invalid
impossibility, teacher-forcing, forced-selection reuse -- all still apply
unchanged to the frozen arm since it shares the same run_sequence/step_logits
code path).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_seqfull01 import overlap_at_k, run_sequence, step_losses


# ---------- frozen-encoder semantics ----------

def test_frozen_encoder_params_receive_no_gradient():
    """Mirrors the exact assertion train_seqfull01.py makes for
    --frozen_encoder: with requires_grad=False, autograd must never assign
    .grad to an encoder parameter, not merely leave it at zero."""
    torch.manual_seed(0)
    encoder = torch.nn.Linear(8, 8)
    for p in encoder.parameters():
        p.requires_grad = False
    x = torch.randn(4, 8)
    out = encoder(x)
    assert not out.requires_grad, 'frozen encoder output must not require grad'
    # Nothing to backward through the encoder; simulate the loss path used in
    # training by feeding `out` into a trainable head instead.
    head = torch.nn.Linear(8, 1)
    loss = head(out).sum()
    loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_frozen_encoder_weights_are_bit_identical_before_and_after_training_step():
    """A frozen encoder's weights must not move even after an optimizer step
    that updates other parameters in the same forward/backward graph."""
    torch.manual_seed(0)
    encoder = torch.nn.Linear(8, 8)
    for p in encoder.parameters():
        p.requires_grad = False
    before = [p.clone() for p in encoder.parameters()]
    head = torch.nn.Linear(8, 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.1)
    for _ in range(5):
        optimizer.zero_grad()
        loss = head(encoder(torch.randn(4, 8))).sum()
        loss.backward()
        optimizer.step()
    after = list(encoder.parameters())
    for b, a in zip(before, after):
        torch.testing.assert_close(b, a)


def test_set_conditioner_gradient_nonzero_with_a_frozen_encoder():
    """The actual EXP-SEQDIAG01 configuration: E has no gradient path at all
    (frozen encoder), but the SetConditioner must still receive a real
    gradient from the sequential CE loss."""
    torch.manual_seed(0)
    n, dim, bsz, k = 12, 6, 2, 4
    with torch.no_grad():
        E = F.normalize(torch.randn(n, dim), dim=-1)
    assert not E.requires_grad, 'this test simulates the frozen-encoder case'
    q = F.normalize(torch.randn(bsz, dim), dim=-1)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    teacher_idx = torch.randint(0, n, (bsz, k))
    set_cond = SetConditioner(dim)
    empty = EmptySetToken(dim)
    logits_per_step, _ = run_sequence(q, E, valid, set_cond, empty, k=k, teacher_idx=teacher_idx)
    query_valid = torch.ones(bsz, dtype=torch.bool)
    losses, _ = step_losses(logits_per_step, teacher_idx, query_valid)
    (sum(losses) / k).backward()
    sc_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in set_cond.parameters() if p.grad is not None
    ) ** 0.5
    assert sc_grad_norm > 0, 'SetConditioner must learn even with a frozen encoder'


# ---------- metric / chance-baseline definitions (section 10 audit) ----------

def test_overlap_at_k_is_the_normalized_intersection_not_the_raw_count():
    """Pins the exact definition in use: |S_pred ∩ S_teacher| / K, not the
    raw intersection count. This is the formula whose chance baseline is
    K/N -- confusing it with the raw count (chance K^2/N) or with a
    per-candidate probability (chance 1/N) is exactly the labeling bug this
    test guards against a repeat of (see EXPERIMENT_LOG.md's ERRATUM)."""
    k = 5
    picks = torch.tensor([[1, 2, 3, 4, 5]])
    teacher = torch.tensor([[3, 4, 5, 6, 7]])  # 3 overlap
    valid = torch.tensor([True])
    result = overlap_at_k(picks, teacher, valid)
    raw_intersection = 3
    assert result == pytest.approx(raw_intersection / k)
    assert result != pytest.approx(raw_intersection)  # would be the raw-count reading


@pytest.mark.parametrize('n,k', [(8449, 10), (240, 10), (36696, 10)])
def test_normalized_overlap_chance_baseline_is_k_over_n(n, k):
    """Monte Carlo check (small trial count, seeded) that E[|A∩B|/K] for two
    independent random K-subsets of N items is K/N, not K^2/N (raw count
    expectation) or 1/N (single-candidate-match probability) -- the two
    wrong formulas that were actually used in EXP-SEQFULL01's original
    write-up before this experiment's mandatory audit caught it."""
    torch.manual_seed(0)
    trials = 4000
    total = 0.0
    for _ in range(trials):
        a = torch.randperm(n)[:k]
        b = torch.randperm(n)[:k]
        total += len(set(a.tolist()) & set(b.tolist())) / k
    empirical = total / trials
    correct = k / n
    wrong_raw_count = (k * k) / n
    wrong_per_candidate = 1 / n
    # Empirical must land near the correct formula and clearly away from the
    # two wrong ones whenever they are numerically distinguishable at this N/K.
    assert empirical == pytest.approx(correct, abs=max(3 * (correct * (1 - correct) / trials) ** 0.5, 1e-4))
    if abs(wrong_raw_count - correct) > 1e-6:
        assert abs(empirical - wrong_raw_count) > abs(empirical - correct)
    if abs(wrong_per_candidate - correct) > 1e-6:
        assert abs(empirical - wrong_per_candidate) > abs(empirical - correct)


def test_raw_intersection_count_chance_baseline_is_k_squared_over_n():
    """The OTHER valid metric definition (raw count, not normalized) has its
    own, different chance baseline -- pinned so the two are never conflated
    again."""
    torch.manual_seed(0)
    n, k, trials = 500, 10, 4000
    total = 0
    for _ in range(trials):
        a = torch.randperm(n)[:k]
        b = torch.randperm(n)[:k]
        total += len(set(a.tolist()) & set(b.tolist()))
    empirical = total / trials
    assert empirical == pytest.approx(k * k / n, rel=0.25)
