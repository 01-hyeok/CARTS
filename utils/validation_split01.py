"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A5.

The original track selected every checkpoint using the FULL validation
split's forecast MSE. Any later attempt to carve a sub-range out of
that same validation split and call it an "untouched holdout" would be
invalid -- the checkpoint was already chosen using all of it. This
module provides a genuine chronological Val-Select / Val-Holdout split,
for NEW controlled runs in this causal-validation track, with a purge
gap sized to the window length so no VAL-SELECT window's input or
target range can overlap any VAL-HOLDOUT window's input or target
range.

Window `i` (0-indexed into the validation split) covers input indices
`[i, i+seq_len)` and target indices `[i+seq_len, i+seq_len+pred_len)`,
so its full span is `seq_len+pred_len` steps. Two windows `i < j`
overlap (input or target) iff `j < i + seq_len + pred_len`. Using a
purge gap of exactly `seq_len + pred_len` windows between the last
Val-Select window and the first Val-Holdout window is therefore
sufficient (and keeps one extra window of margin beyond the minimum
`seq_len+pred_len-1`).
"""


def chronological_select_holdout_indices(n_windows, seq_len, pred_len, select_fraction=0.5):
    """Returns (select_indices, holdout_indices), both plain `range`-like
    lists of window indices into the validation split, in chronological
    order, Val-Select first (used for checkpoint/hyperparameter
    selection), Val-Holdout last (used ONLY for the final reported
    comparison, never for any selection decision), separated by a purge
    gap of `seq_len + pred_len` windows so no window's input/target
    range crosses the split boundary.

    If the purge gap consumes the entire remaining range, `holdout_indices`
    is empty -- callers must check for this (it means `n_windows` is too
    small for this `seq_len+pred_len` at this `select_fraction` and the
    split should not be used, not silently proceed with zero holdout
    queries)."""
    if n_windows <= 0:
        return [], []
    purge_gap = seq_len + pred_len
    n_select = int(round(n_windows * select_fraction))
    n_select = max(0, min(n_select, n_windows))
    select_indices = list(range(0, n_select))
    holdout_start = n_select + purge_gap
    holdout_indices = list(range(holdout_start, n_windows)) if holdout_start < n_windows else []
    return select_indices, holdout_indices


def verify_no_window_overlap(select_indices, holdout_indices, seq_len, pred_len):
    """Returns True iff no (select window, holdout window) pair's
    [i, i+seq_len+pred_len) spans intersect. O(1) given the indices are
    sorted ranges from `chronological_select_holdout_indices` (checks
    only the boundary), but implemented as an explicit pairwise check
    over the boundary-adjacent windows so it also catches a caller
    passing arbitrary (not necessarily gap-respecting) index lists."""
    if not select_indices or not holdout_indices:
        return True
    span = seq_len + pred_len
    last_select = max(select_indices)
    first_holdout = min(holdout_indices)
    # also guard against a holdout index appearing BEFORE a select index
    # (caller error), not just the chronological-gap case.
    if first_holdout >= last_select:
        return first_holdout >= last_select + span
    first_select = min(select_indices)
    last_holdout = max(holdout_indices)
    return first_select >= last_holdout + span
