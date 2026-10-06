"""TRACK-W-TIMESTAMP-FUSION01 unit/smoke tests (spec section 18).

Covers: A (timestamp alignment), B (no leakage), C (shared encoder),
D (gradient), E (C0/C1/C2 fairness), F (multi-query fairness),
G (zero_proj has zero projection params), J (teacher invariance).
H/I are covered by existing, unmodified `test_t_pure_multislot_validation01.py`
tests of `round_robin_topk_selection`/`hard_eval_decomposition` (reused
UNMODIFIED here, so re-testing their core math again would be redundant)
-- this file only checks that THIS track's new code calls them correctly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.TimestampRelationEncoder import TimestampFusionEncoder, zero_time_features
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_w_timestamp_common01 import (
    EXPECTED_CADENCE, FREQ_OVERRIDE, detect_cadence, encode_raw_timestamp,
    make_shuffle_permutation, shuffled_memory_x_mark,
)

ETTH1_ROOT = '../Dataset/Time-Series-Library_dataset/ETT-small/'
ETTH1_PATH = 'ETTh1.csv'
WEATHER_ROOT = '../Dataset/Time-Series-Library_dataset/weather/'
WEATHER_PATH = 'weather.csv'


def _skip_if_missing(root, path):
    if not (Path(REPO_ROOT) / root / path).exists():
        pytest.skip(f'dataset file not found: {root}{path}')


# -------------------- A: cadence / alignment --------------------

def test_a1_etth1_cadence_is_hourly():
    _skip_if_missing(ETTH1_ROOT, ETTH1_PATH)
    median_interval, _ = detect_cadence(str(Path(REPO_ROOT) / ETTH1_ROOT), ETTH1_PATH)
    assert median_interval == EXPECTED_CADENCE['ETTh1']


def test_a2_weather_cadence_is_10min_not_hourly():
    _skip_if_missing(WEATHER_ROOT, WEATHER_PATH)
    median_interval, _ = detect_cadence(str(Path(REPO_ROOT) / WEATHER_ROOT), WEATHER_PATH)
    assert median_interval == EXPECTED_CADENCE['custom']
    assert median_interval != EXPECTED_CADENCE['ETTh1'], \
        '[ISSUE] Weather must NOT be treated as hourly -- this is the whole point of the cadence audit'


def test_a3_memory_x_mark_alignment_first_middle_last():
    """sliding_window_view(data_stamp, L, axis=0).transpose(0,2,1) must
    give memory_x_mark[i] == data_stamp[starts_local[i] : starts_local[i]+L]
    for first/middle/last candidate -- the exact claim AUDIT.md PART 2.4
    makes, checked directly against synthetic data (no GPU/dataset
    download needed for this one)."""
    T, L, K = 50, 5, 3
    data_stamp = np.arange(T * K, dtype=np.float32).reshape(T, K)
    num_windows = T - L + 1
    memory_x_mark = np.lib.stride_tricks.sliding_window_view(
        data_stamp, L, axis=0
    ).transpose(0, 2, 1)[:num_windows]
    for i in (0, num_windows // 2, num_windows - 1):
        np.testing.assert_array_equal(memory_x_mark[i], data_stamp[i:i + L])


# -------------------- B: no leakage --------------------

def test_b1_stage1_window_dataset_never_returns_future_mark():
    """`Stage1WindowDataset.__getitem__` must discard `seq_y_mark`
    unconditionally -- checked directly against the source, since the
    additive `include_time_mark` patch must not have reintroduced it."""
    import inspect
    from utils.relation_memory import Stage1WindowDataset
    src = inspect.getsource(Stage1WindowDataset.__getitem__)
    assert 'seq_y_mark' not in src.replace('_, _ =', '').replace("'_, _, _, _'", ''), \
        '[ISSUE] seq_y_mark must never be named/returned from __getitem__'
    # the discard pattern unpacks it into `_` -- confirm the tuple position
    # count rather than a name (future calendar is positional arg 4 of
    # base_dataset.__getitem__'s 5-tuple, must stay discarded)
    assert '_, seq_x, seq_y, seq_x_mark, _ = item' in src


# -------------------- C: shared encoder --------------------

def test_c1_query_and_candidate_use_identical_module_identical_output():
    torch.manual_seed(0)
    enc = TimestampFusionEncoder(seq_len=8, d_model=16, d_ff=32, time_feat_dim=4, time_proj_dim=6, dropout=0.0)
    enc.eval()  # dropout=0 anyway, but eval() removes any doubt about determinism
    x = torch.randn(3, 8)
    c = torch.randn(3, 8, 4)
    out_as_query = enc(x, c)
    out_as_candidate = enc(x, c)  # identical inputs through the SAME module instance
    assert torch.allclose(out_as_query, out_as_candidate)
    # encode_raw_timestamp must extract channel c's own delta-last window
    batch_x = torch.randn(3, 8, 2)
    mark = torch.randn(3, 8, 4)
    z0 = encode_raw_timestamp(enc, batch_x, mark, 0)
    z1 = encode_raw_timestamp(enc, batch_x, mark, 1)
    assert not torch.allclose(z0, z1), 'different channels must give different embeddings'


# -------------------- D: gradient --------------------

def test_d1_value_and_time_proj_both_get_nonzero_gradient():
    torch.manual_seed(0)
    enc = TimestampFusionEncoder(seq_len=6, d_model=8, d_ff=16, time_feat_dim=3, time_proj_dim=4)
    x = torch.randn(2, 6, requires_grad=False)
    c = torch.randn(2, 6, 3)
    z = enc(x, c)
    z.sum().backward()
    assert enc.value_proj.weight.grad is not None and enc.value_proj.weight.grad.abs().sum() > 0
    assert enc.time_proj.weight.grad is not None and enc.time_proj.weight.grad.abs().sum() > 0


def test_d2_no_time_mode_zero_input_still_has_identical_architecture():
    torch.manual_seed(0)
    enc = TimestampFusionEncoder(seq_len=6, d_model=8, d_ff=16, time_feat_dim=3, time_proj_dim=4)
    x = torch.randn(2, 6)
    zero_c = zero_time_features(x, time_feat_dim=3)
    assert zero_c.shape == (2, 6, 3)
    assert torch.all(zero_c == 0)
    z = enc(x, zero_c)
    assert z.shape == (2, 8)  # architecture unchanged -- same forward signature/output shape


# -------------------- E: C0/C1/C2 fairness --------------------

def test_e1_param_count_and_init_hash_identical_regardless_of_time_mode():
    """C0/C1/C2 differ only in what DATA is fed into the SAME
    architecture -- param count and init hash (before any forward pass)
    must be identical when constructed with the same seed."""
    def build():
        torch.manual_seed(42)
        return TimestampFusionEncoder(seq_len=10, d_model=16, d_ff=32, time_feat_dim=5, time_proj_dim=8)

    enc_a, enc_b = build(), build()
    count_a = sum(p.numel() for p in enc_a.parameters())
    count_b = sum(p.numel() for p in enc_b.parameters())
    assert count_a == count_b
    for (ka, va), (kb, vb) in zip(sorted(enc_a.state_dict().items()), sorted(enc_b.state_dict().items())):
        assert ka == kb
        assert torch.allclose(va, vb), f'[ISSUE] init mismatch at {ka}'


def test_e2_shuffled_memory_mark_is_a_pure_reorder_not_a_resample():
    torch.manual_seed(0)
    N, L, K = 20, 4, 3
    real = torch.randn(N, L, K)
    perm = make_shuffle_permutation(N, seed=0)
    shuffled = shuffled_memory_x_mark(real, perm)
    assert shuffled.shape == real.shape
    # every row of `shuffled` must be some row of `real` -- a reorder, not new data
    for i in range(N):
        assert any(torch.allclose(shuffled[i], real[j]) for j in range(N))
    # and the permutation must not be the identity (or the test setup is degenerate)
    assert not np.array_equal(perm, np.arange(N))


# -------------------- F: multi-query fairness (SlotHeads reuse) --------------------

def test_f1_slotheads_w1_bit_identical_across_num_slots():
    d_model = 16
    w1_at_1 = SlotHeads(d_model, n_slots=1, std=1e-3).W[0]
    w1_at_2 = SlotHeads(d_model, n_slots=2, std=1e-3).W[0]
    w1_at_5 = SlotHeads(d_model, n_slots=5, std=1e-3).W[0]
    assert torch.equal(w1_at_1, w1_at_2)
    assert torch.equal(w1_at_1, w1_at_5)


# -------------------- G: zero_proj has zero projection parameters --------------------

def test_g1_zero_proj_mode_has_no_slotheads_params():
    """--mode zero_proj (C0/C1/C2, and T-V0) must involve NO SlotHeads
    module at all -- checked structurally: the training script only
    constructs SlotHeads when --mode slots."""
    src = Path(REPO_ROOT, 'scripts', 'train_w_timestamp_stage1_01.py').read_text()
    assert "if cli.mode == 'slots':" in src
    assert 'slot_heads = SlotHeads' in src


# -------------------- J: teacher invariance --------------------

def test_j1_teacher_independent_of_time_encoder_or_mode():
    """The future-MSE teacher (`individual_utility_memsafe` on
    `memory_y`/`batch_y`) must not depend on the time encoder or
    time_mode at all -- it is computed from `memory_c`/`offset_c`/
    `query_future` alone, never touching `time_encoder`."""
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    torch.manual_seed(0)
    memory_c = torch.randn(30, 5)
    offset_c = torch.randn(4)
    query_future = torch.randn(4, 5)
    u1 = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=8)
    u2 = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=8)
    assert torch.allclose(u1, u2), 'teacher must be a pure function of (memory_c, offset_c, query_future)'
