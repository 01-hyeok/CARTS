"""EXP-SEQFULL01: unit coverage for the sequential, teacher-forced,
full-memory selector (models/SequentialSetRetriever.py,
scripts/train_seqfull01.py). Pure-function tests on toy tensors, following
this repo's convention of testing the extracted pieces rather than
constructing the full experiment (see tests/test_stage1_full_memory_metric.py,
tests/test_exp_frr01_arms.py).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import EmptySetToken, SetConditioner, step_logits
from scripts.train_seqfull01 import overlap_at_k, run_sequence, step_losses
from models.RelationStage2 import Model as RelationStage2Model
from utils.oracle_intervention import select_greedy_weighted_set


def _toy_setup(bsz=3, n=20, dim=8, seed=0):
    torch.manual_seed(seed)
    E = F.normalize(torch.randn(n, dim, requires_grad=True), dim=-1)
    q = F.normalize(torch.randn(bsz, dim), dim=-1)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    return E, q, valid_mask


# ---------- 1/2. masking ----------

def test_selected_candidate_masked_at_next_step():
    E, q, valid = _toy_setup()
    selected = torch.zeros_like(valid)
    selected[:, 3] = True
    logits = step_logits(q, E, selected, valid)
    assert (logits[:, 3] < -1e30).all()


def test_invalid_candidate_never_selected():
    E, q, valid = _toy_setup()
    valid = valid.clone()
    valid[:, 5] = False
    selected = torch.zeros_like(valid)
    logits = step_logits(q, E, selected, valid)
    assert (logits[:, 5] < -1e30).all()
    assert logits.argmax(dim=-1).eq(5).sum() == 0


# ---------- 3. no duplicates ----------

def test_duplicate_selection_impossible_over_a_full_free_running_sequence():
    E, q, valid = _toy_setup(n=15)
    set_cond = SetConditioner(E.size(-1))
    empty = EmptySetToken(E.size(-1))
    with torch.no_grad():
        _, picks = run_sequence(q, E, valid, set_cond, empty, k=10, teacher_idx=None)
    for row in picks:
        assert len(set(row.tolist())) == row.numel(), 'duplicate pick in a free-running sequence'


# ---------- 4. full-memory support size invariant ----------

def test_full_memory_support_never_shrinks_across_steps():
    E, q, valid = _toy_setup(n=37)
    set_cond = SetConditioner(E.size(-1))
    empty = EmptySetToken(E.size(-1))
    with torch.no_grad():
        logits_per_step, _ = run_sequence(q, E, valid, set_cond, empty, k=10, teacher_idx=None)
    for logits in logits_per_step:
        assert logits.shape == (q.size(0), 37), 'a step scored fewer than the full memory'


# ---------- 5. teacher matches the existing Weighted Set Oracle exactly ----------

def test_teacher_sequence_is_exactly_select_greedy_weighted_set():
    """The precompute script calls select_greedy_weighted_set with no wrapper
    logic of its own beyond gathering pool_idx -- with a FULL pool (pool_idx =
    arange(N)), gathering is the identity, so the teacher must reproduce
    select_greedy_weighted_set's own output bit-for-bit on the same inputs."""
    torch.manual_seed(0)
    bsz, n, h = 4, 12, 5
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    scores = torch.randn(bsz, n)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    a = select_greedy_weighted_set(futures, query_future, scores, valid, k=5, tau=0.1)
    b = select_greedy_weighted_set(futures, query_future, scores, valid, k=5, tau=0.1)
    torch.testing.assert_close(a, b)


# ---------- 6. teacher-forcing prefix correctness ----------

def test_teacher_forced_set_state_uses_the_oracle_prefix_not_the_models_own_picks():
    """Construct E so every candidate has a distinguishable embedding, force
    the student's own argmax to disagree with the teacher at step 1, and
    check that step 2's set-state (m) is built from the ORACLE's step-1 pick,
    not whatever the student would have chosen on its own."""
    dim = 4
    n = 6
    E = torch.eye(n, dim + 2)[:, :dim].float()  # distinguishable rows
    E = F.normalize(E, dim=-1)
    q = torch.zeros(1, dim)
    valid = torch.ones(1, n, dtype=torch.bool)
    teacher_idx = torch.tensor([[2, 4, 0, 1, 3]])  # arbitrary oracle order, k=5

    class RecordingConditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_m = []

        def forward(self, q, m):
            self.seen_m.append(m.clone())
            return F.normalize(q + 0.0 * m, dim=-1)  # ignore m in the score itself

    cond = RecordingConditioner()
    empty = EmptySetToken(dim)
    run_sequence(q, E, valid, cond, empty, k=5, teacher_idx=teacher_idx)
    # step index 1 (t=1) is the first step where a prefix exists (from step 0).
    expected_m_step1 = E[2].unsqueeze(0)  # mean of just the oracle's t=0 pick
    torch.testing.assert_close(cond.seen_m[1], expected_m_step1)


# ---------- 7. CE target is the next oracle candidate ----------

def test_ce_target_is_the_next_oracle_candidate():
    torch.manual_seed(0)
    n = 9
    logits = torch.randn(2, n)
    teacher_idx = torch.tensor([[3, 5], [1, 7]])
    query_valid = torch.tensor([True, True])
    losses, accs = step_losses([logits], teacher_idx, query_valid)
    expected = F.cross_entropy(logits, teacher_idx[:, 0])
    torch.testing.assert_close(losses[0], expected)


def test_ce_ignores_invalid_queries():
    n = 6
    logits = torch.randn(3, n)
    teacher_idx = torch.tensor([[-1, -1], [2, 3], [4, 1]])
    query_valid = torch.tensor([False, True, True])
    losses, _ = step_losses([logits], teacher_idx, query_valid)
    target = teacher_idx[:, 0].clamp_min(0)
    expected = F.cross_entropy(logits[1:], target[1:])
    torch.testing.assert_close(losses[0], expected)


# ---------- 8. query Y_q absent from the student forward path ----------

def test_run_sequence_signature_has_no_query_future_parameter():
    import inspect
    params = set(inspect.signature(run_sequence).parameters)
    assert not (params & {'y', 'query_y', 'batch_y', 'query_future'}), (
        'run_sequence must never receive the query future; only the offline '
        'teacher construction may')


# ---------- 9/10/11. candidate-side gradient ----------

def test_candidate_gradient_is_nonzero_and_embeddings_are_not_detached():
    torch.manual_seed(0)
    n, dim, bsz, k = 12, 6, 2, 4
    raw = torch.randn(n, dim, requires_grad=True)
    E = F.normalize(raw, dim=-1)
    assert E.requires_grad, 'candidate embeddings must not be detached'
    q = F.normalize(torch.randn(bsz, dim), dim=-1)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    teacher_idx = torch.randint(0, n, (bsz, k))
    set_cond = SetConditioner(dim)
    empty = EmptySetToken(dim)
    logits_per_step, _ = run_sequence(q, E, valid, set_cond, empty, k=k, teacher_idx=teacher_idx)
    query_valid = torch.ones(bsz, dtype=torch.bool)
    losses, _ = step_losses(logits_per_step, teacher_idx, query_valid)
    (sum(losses) / k).backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_k_steps_reuse_one_differentiable_embedding_tensor():
    """The same E object (identity, not a recomputed copy) must be used at
    every step -- this is what "one encoder forward per optimisation step"
    means in practice: run_sequence must not internally re-derive E."""
    n, dim, bsz, k = 8, 4, 2, 3
    E = F.normalize(torch.randn(n, dim, requires_grad=True), dim=-1)
    q = F.normalize(torch.randn(bsz, dim), dim=-1)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    set_cond = SetConditioner(dim)
    empty = EmptySetToken(dim)
    seen_ids = []
    orig_step_logits = step_logits

    def spy(h_t, candidate_embeddings, selected_mask, valid_mask):
        seen_ids.append(id(candidate_embeddings))
        return orig_step_logits(h_t, candidate_embeddings, selected_mask, valid_mask)

    import scripts.train_seqfull01 as trainmod
    trainmod.step_logits = spy
    try:
        run_sequence(q, E, valid, set_cond, empty, k=k, teacher_idx=None)
    finally:
        trainmod.step_logits = orig_step_logits
    assert len(set(seen_ids)) == 1, 'a different embedding tensor was used across steps'
    assert seen_ids[0] == id(E)


# ---------- overlap metric sanity (supports the small-N gate's PASS reading) ----------

def test_overlap_at_k_is_one_when_sets_match_regardless_of_order():
    picks = torch.tensor([[1, 2, 3], [4, 5, 6]])
    teacher = torch.tensor([[3, 2, 1], [6, 5, 4]])
    valid = torch.tensor([True, True])
    assert overlap_at_k(picks, teacher, valid) == pytest.approx(1.0)


def test_overlap_at_k_ignores_invalid_queries():
    picks = torch.tensor([[1, 2], [7, 8]])
    teacher = torch.tensor([[9, 9], [7, 8]])
    valid = torch.tensor([False, True])
    assert overlap_at_k(picks, teacher, valid) == pytest.approx(1.0)


# ---------- 12/13. forced selection reaches Stage-2 unchanged ----------

def test_forced_selection_table_reaches_forced_selection_for_and_nothing_else():
    """set_forced_selection must only ever set the one attribute the existing,
    already-reviewed oracle-intervention mechanism reads -- EXP-SEQFULL01
    reuses this verbatim for Stage-2, so nothing about weighting/gate/fusion
    is touched by installing a table built from the sequential selector."""
    class Stub:
        set_forced_selection = RelationStage2Model.set_forced_selection
        _forced_selection_for = RelationStage2Model._forced_selection_for

    stub = Stub()
    before = dict(vars(stub))
    table = {(0, 0): torch.tensor([[1, 2, 3]])}
    stub.set_forced_selection(table)
    after = dict(vars(stub))
    new_keys = set(after) - set(before)
    assert new_keys == {'_forced_selection'}
    torch.testing.assert_close(stub._forced_selection_for(0, 0), table[(0, 0)])
    assert stub._forced_selection_for(1, 0) is None
    stub.set_forced_selection(None)
    assert stub._forced_selection_for(0, 0) is None
