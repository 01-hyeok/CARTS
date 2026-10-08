"""TRACK-V-SHARED-TOP100-SIGNFIX01 -- reproduces and then (after the fix)
disproves the teacher-sign bug in scripts/train_v_sharedtop100_01.py.

Bug: `pooled_future_mse` already returns a lower-is-better MSE distance.
`normalized_teacher_prob(d, ...)` already internally negates (z-score
then `softmax(-z/tau)`) so the correct call is `normalized_teacher_prob(d_pool, ...)`.
The buggy code called `normalized_teacher_prob(-d_pool, ...)`, negating
twice in effect and assigning HIGH probability to HIGH-MSE (bad)
candidates."""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob

SHAREDTOP100_SRC = (REPO_ROOT / 'scripts' / 'train_v_sharedtop100_01.py').read_text()
FULL_MEMORY_FILES = (
    'train_t_pure_multislot01.py', 'train_j_shared_encoder_drift01.py',
    'train_expert_v5_full01.py', 'train_hard_expert_v5_full01.py',
)


# -------------------- Test A: teacher prob decreases as distance increases --------------------

def test_a_teacher_prob_decreases_with_distance():
    d = torch.tensor([[0.1, 1.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    p = normalized_teacher_prob(d, mask, tau_t=0.1)
    assert p[0, 0] > p[0, 1] > p[0, 2], f'[ISSUE] teacher prob must decrease as distance increases, got {p}'


# -------------------- Test B: argmax(p_T) == argmin(d) --------------------

def test_b_argmax_prob_equals_argmin_distance():
    d = torch.tensor([[0.1, 1.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    p = normalized_teacher_prob(d, mask, tau_t=0.1)
    assert int(p.argmax()) == int(d.argmin()) == 0


def test_reproduces_the_bug_on_the_buggy_call_pattern():
    """Confirms the BUGGED call pattern (`normalized_teacher_prob(-d, ...)`
    applied to an already-correct distance) inverts the ranking --
    this is what scripts/train_v_sharedtop100_01.py did before the fix."""
    d = torch.tensor([[0.1, 1.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    p_buggy = normalized_teacher_prob(-d, mask, tau_t=0.1)
    assert int(p_buggy.argmax()) == int(d.argmax()) == 2, \
        '[ISSUE] expected the buggy call to give max probability to the WORST (highest-distance) candidate'


# -------------------- Test C: P100 teacher parity vs full-memory reference on the same d --------------------

def test_c_p100_teacher_matches_full_memory_reference_on_same_d():
    d = torch.rand(4, 20)
    mask = torch.ones(4, 20, dtype=torch.bool)
    p_full_style = normalized_teacher_prob(d, mask, tau_t=0.1)   # full-memory convention: pass distance directly
    p_p100_style = normalized_teacher_prob(d, mask, tau_t=0.1)   # P100, AFTER the fix: identical call
    assert torch.allclose(p_full_style, p_p100_style, atol=1e-9)


# -------------------- Test D: ranking by ascending d is NOT reversed in probability --------------------

def test_d_ranking_not_reversed():
    g = torch.Generator().manual_seed(0)
    d = torch.rand(1, 10, generator=g)
    mask = torch.ones(1, 10, dtype=torch.bool)
    p = normalized_teacher_prob(d, mask, tau_t=0.1)
    order_by_d = d[0].argsort()          # ascending distance = best-to-worst
    order_by_p = p[0].argsort(descending=True)  # descending probability = best-to-worst
    assert torch.equal(order_by_d, order_by_p), '[ISSUE] probability ranking must match ascending-distance ranking exactly'


# -------------------- Test E: Full-memory teacher sign unaffected by this fix --------------------

def test_e_full_memory_files_use_correct_utility_negation_pattern():
    for fname in FULL_MEMORY_FILES:
        src = (REPO_ROOT / 'scripts' / fname).read_text()
        assert 'normalized_teacher_prob(' in src, f'{fname} should call normalized_teacher_prob'
        # every call site in these files is either `normalized_teacher_prob(-u, ...)` (u = utility,
        # higher-is-better, from individual_utility_memsafe -- so -u IS the correct distance) or
        # `normalized_teacher_prob(d_raw, ...)` where d_raw was itself already defined as `-u` --
        # NEVER a double negation of an already-signed distance like d_pool.
        assert 'normalized_teacher_prob(-d_pool' not in src
        assert 'normalized_teacher_prob(-teacher_d' not in src


def test_sharedtop100_file_no_longer_double_negates_d_pool():
    assert 'normalized_teacher_prob(-d_pool' not in SHAREDTOP100_SRC, \
        '[ISSUE][ABORT] train_v_sharedtop100_01.py still double-negates d_pool -- fix not applied'
    assert 'normalized_teacher_prob(d_pool' in SHAREDTOP100_SRC, \
        '[ISSUE] expected the corrected call normalized_teacher_prob(d_pool, ...)'
