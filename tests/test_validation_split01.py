from utils.validation_split01 import chronological_select_holdout_indices, verify_no_window_overlap


def test_split_is_chronological_select_before_holdout():
    sel, hold = chronological_select_holdout_indices(2000, seq_len=96, pred_len=96, select_fraction=0.5)
    assert sel == list(range(0, 1000))
    assert hold  # must be non-empty for this realistic window count
    assert min(hold) > max(sel)


def test_purge_gap_prevents_any_window_overlap():
    for seq_len, pred_len, n_windows in [(96, 96, 300), (192, 192, 500), (336, 336, 800), (96, 720, 1000)]:
        sel, hold = chronological_select_holdout_indices(n_windows, seq_len, pred_len, 0.6)
        assert verify_no_window_overlap(sel, hold, seq_len, pred_len), \
            f'overlap found for seq_len={seq_len} pred_len={pred_len} n_windows={n_windows}'


def test_too_few_windows_yields_empty_holdout_not_a_silent_overlap():
    """If the purge gap would consume everything, holdout must be empty
    -- never a non-empty holdout that actually overlaps select."""
    sel, hold = chronological_select_holdout_indices(50, seq_len=96, pred_len=96, select_fraction=0.5)
    assert hold == []


def test_detects_actual_overlap_when_indices_given_without_gap():
    """verify_no_window_overlap must be able to catch a real overlap, not
    just rubber-stamp whatever is passed in."""
    sel = list(range(0, 100))
    hold_overlapping = list(range(50, 150))  # overlaps sel directly, let alone the span gap
    assert not verify_no_window_overlap(sel, hold_overlapping, seq_len=96, pred_len=96)


def test_empty_inputs_are_trivially_non_overlapping():
    assert verify_no_window_overlap([], [1, 2, 3], 96, 96)
    assert verify_no_window_overlap([1, 2, 3], [], 96, 96)
