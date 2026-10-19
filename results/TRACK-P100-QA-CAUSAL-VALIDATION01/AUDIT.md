# TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A Audit

Status: Phase A code fixes complete and unit-tested (20/20 new tests
pass; full existing suite re-run for regressions, see below). GPU-based
Phase B/C runs have NOT started yet -- they queue behind the still-running
original TRACK-P100-SOFTTOPK-QUALITY-AWARE01 grid on GPU1 (no parallel
GPU jobs, per standing rule). This document records exactly what was
found, how it was verified against real code/artifacts (not assumed),
and what was fixed.

Principle followed throughout: **original_protocol artifacts are never
deleted, modified, or silently overwritten.** Every fix below is a NEW
file; the original scripts/results this track audits remain byte-for-byte
as they were.

---

## A1. QA/NOQA batch-order confound -- CONFIRMED, FIXED

**Claim**: the QA arm's one-time feature-normalizer fitting pass
iterates the SAME `train_loader` object used for real training (`for
bi, (...) in enumerate(train_loader): if bi >= 50: break`), before the
real training loop's first `enumerate(train_loader)` call.

**Verification**: read `train_softtopk_e2e_v01.py` lines 199-235 (and
the structurally identical code in `train_softtopk_e2e_hard01.py` lines
177-207, `train_softtopk_e2e_joint01.py` lines 176-206). Confirmed the
mechanism: `train_loader` is built once with `generator=train_gen`
(`torch.Generator().manual_seed(loader_seed)`); each
`enumerate(train_loader)` call triggers a fresh
`RandomSampler.__iter__()`, which draws exactly one new permutation
from `train_gen`'s state, advancing it, regardless of how many batches
are actually consumed before `break`. QA's fitting pass therefore
burns one permutation draw that NOQA (which has no fitting pass) never
burns -- so QA's real epoch-1 training loop sees `train_gen`'s SECOND
draw while NOQA's epoch-1 sees its FIRST, and this one-draw offset
persists for every subsequent epoch (each arm's `enumerate(train_loader)`
calls stay offset by exactly one from then on).

**Confirmed against real artifacts**, not just reasoned about: pulled
`epoch_metrics.json`'s saved `batch_order_sha256` field for epoch 1 of
all 4 originally-flagged cells --

| Cell/Method | NOQA epoch1 hash | QA epoch1 hash | Match? |
|---|---|---|---|
| ETTh1_96/V5 | `25cf11ab788f822d` | `0785eb6d1169cd3e` | No |
| ETTh1_192/V5 | `1e67d4a11234dd81` | `43af1bde7e52e39f` | No |
| Weather_720/Joint | `6af96801af06168a` | `0492e2023f796281` | No |
| Solar_96/V0 | `bd4af217432b086a` | `df1d624a01d62b91` | No |

All four mismatch, exactly as predicted by the mechanism. Since the
mechanism is universal (identical code path in every V-family/Hard/
Joint QA run), **this confound affects every one of the original
track's 72 QA runs**, not just these 4 -- they were checked only
because the spec named them as representative.

**Fix**: `scripts/train_softtopk_e2e_v02_rngfixed.py`,
`train_softtopk_e2e_hard02_rngfixed.py`,
`train_softtopk_e2e_joint02_rngfixed.py`. The fitting pass now builds
its OWN throwaway `DataLoader`/`Generator`
(`loader_seed + 10_000_003`), never touching `train_gen`. `train_loader`
is therefore byte-identical in content and iteration order between
NOQA and QA for every epoch. Also newly recorded: `slot_heads_init_hash`
(all 3 trainers) and `router_init_hash` (Joint only), so the *other*
half of A1's instruction -- "the common retriever parameters must be
verified identically initialized, not just assumed" -- is checkable,
not just asserted.

**Verification tooling**: `scripts/verify_rngfixed_pairs01.py` asserts,
for one (cell, method) pair's rng-corrected run: `init_hash` match,
`slot_heads_init_hash` match, `router_init_hash` match, and
`batch_order_sha256` match at **every** epoch (not just epoch 1 -- a
checker that only looked at epoch 1 could miss a confound that
happened to still match there by chance at later epochs under a
different mechanism). 5 unit tests in
`tests/test_verify_rngfixed_pairs01.py`, including a deliberate
epoch-2-only mismatch to prove the full-epoch scan actually runs.

---

## A2. Base Forecaster TEST leakage -- CONFIRMED, scope WIDER than Solar

**Claim (as given)**: the Solar Base Forecaster prep reads the TEST
loader and records test MSE/MAE.

**Verification**: read `scripts/train_r_base_forecaster01.py` line 62
(`_, test_loader = exp._get_data(flag='test', shuffle=False)`) and
lines 116-125 (loads the best-val checkpoint back, evaluates it on
`test_loader`, writes `test_mse`/`test_mae` into `metrics.json`).
Confirmed checkpoint SELECTION itself (`if val_mse < best['val_mse']:
... torch.save(...)`, lines 108-110) happens strictly before the TEST
eval and uses only `val_mse` -- so no selection decision was ever made
using TEST. But TEST data WAS read and a TEST metric WAS computed and
persisted for every checkpoint this script ever produced.

**Scope correction**: traced `scripts/run_softtopk_quality_aware_grid01.sh`'s
`BASE_*` path assignments for all 12 cells, not just Solar:

- `BASE_ETTH1_96`, `BASE_WEATHER_96` -> `checkpoints/track_r_final_method_generalization01/.../base/checkpoint.pth`
- `BASE_ETTH1_{192,336,720}`, `BASE_WEATHER_{192,336,720}` -> `checkpoints/track_v_multiquery_generalization01/.../base/checkpoint.pth`
- `BASE_SOLAR_{96,192,336,720}` -> `checkpoints/track_p100_softtopk_quality_aware01/Solar_base/.../checkpoint.pth`

Opened `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H192/seed0/base/metrics.json`
(one of the non-Solar checkpoints) and confirmed it ALSO has a
populated `test_mse: 0.44721854165196157` field with the identical
`metrics.json` schema `train_r_base_forecaster01.py` produces --
confirming the Solar-built base checkpoints were trained by the SAME
script used (at some earlier point, reused unmodified) for every other
cell too. **This is a track-wide property of every frozen Base
Forecaster used by TRACK-P100-SOFTTOPK-QUALITY-AWARE01, not a
Solar-specific issue.** Checkpoint selection was not contaminated
(val-only), but the claim "TEST completely untouched" is not accurate
for the Base Forecaster component of the original track and is
corrected here rather than repeated.

**Fix**: `scripts/train_base_forecaster_trainval_only01.py` -- identical
architecture/hyperparameters to `train_r_base_forecaster01.py`, but
NEVER constructs a `flag='test'` loader. Available for any future
Base Forecaster checkpoint this causal-validation track needs. Per the
audit instruction, existing Base checkpoints are NOT retrained solely
because of this finding (their selection was val-only and therefore
valid) -- this is a forward-looking fix plus an honest disclosure, not
a retroactive correction of Phase B/C's reused checkpoints.

---

## A3. GO/STOP decision can return a verdict on incomplete data -- CONFIRMED, FIXED

**Claim**: `compute_qa_go_decision01.py` can return `QA_GO=True` even
with fewer than 72/72 combos or a never-computed oracle criterion.

**Verification**: read the script's final line,
`qa_go = criterion_1 and criterion_2 and criterion_3_5 and (criterion_6
is not False)`. `None is not False` evaluates to `True` in Python, so
an unset `criterion_6` (never computed) does not block GO. The script
also never checks `len(rows) == 72` anywhere. **Reproduced live**: ran
the unmodified original script against the real, still-in-progress
grid (125/144 complete, 62/72 combos available, oracle never computed)
on 2026-10-19 --

```
[compute_qa_go_decision] WARNING: 10 (cell,method) combos missing...
[compute_qa_go_decision] QA_GO=True overall_avg_delta=-0.007575 cells_with_qa_win=10/12
  criterion 6_positive_dynamic_oracle_headroom: None
```

Confirmed exactly as claimed: a definite `QA_GO=True` on 62/72 combos
with criterion 6 `None`.

**Fix**: `scripts/compute_qa_go_decision02_strict01.py`. Requires BOTH
all 72/72 combos present AND criterion 6 actually computed
(`is not None`) before returning any `true`/`false` verdict; otherwise
returns `{"status": "INCOMPLETE", "QA_GO": null}`. An
`--allow_incomplete_preview` flag still surfaces the numeric preview
(deltas, averages) for progress-tracking, but `QA_GO` stays `null` even
with that flag -- there is no code path in this script that can return
a `true`/`false` verdict on incomplete data. The original script is
left completely unmodified (it still legitimately backs the original
track's own explicitly-labeled PRELIMINARY interim reporting). 6 unit
tests in `tests/test_compute_qa_go_decision02_strict01.py`, including a
direct regression test reproducing the exact 62/72-partial scenario
above and asserting `status == 'INCOMPLETE'`.

---

## A4. "Per-channel" oracle lambda is actually global -- CONFIRMED, FIXED

**Claim**: `validation_diagnostics01.py`'s `per_channel_oracle_static_lambda`
collapses the channel dimension via an unqualified `.sum()` on `[N, T]`
input, so it is not actually computing a value per channel.

**Verification**: read the function -- `r_minus_b = soft_forecast -
base_forecast` (docstring says shape `[N, T]`), then `num =
(r_minus_b * y_minus_b).sum()` and `den = (r_minus_b ** 2).sum()`, both
with no `dim=` argument, summing over every element of whatever `[N,
T]` tensor is passed. Its only caller, `oracle_headroom_report`, calls
it exactly once on the whole pooled (query x channel) tensor -- so as
written and as used, it returns one global scalar, not one value per
channel, despite the name and docstring.

**Fix**: `utils/validation_diagnostics02_perchannel01.py`. Requires
genuinely `[C, N, T]`-shaped input (raises `ValueError` on anything
else, so the channel axis can never again be silently summed away by
omission) and provides `global_oracle_lambda` (the function the
original ACTUALLY computes, correctly named) alongside a real
`per_channel_oracle_lambda` (reduces only over `(N, T)`, returns one
value per channel, `[C]`) and `oracle_headroom_report_v2` (4-way
comparison: NOQA / Global Oracle / Per-channel Oracle / Dynamic
Oracle). The per-query/channel dynamic oracle
(`validation_oracle_lambda` in `quality_aware_gate01.py`) was already
correct for this shape (reduces only over the last dim, `T`) and is
reused unmodified. 5 unit tests in
`tests/test_validation_diagnostics02_perchannel01.py`, including a
constructed 2-channel example with deliberately opposite optimal
lambdas (channel 0 wants 1.0, channel 1 wants 0.0) that the per-channel
function recovers correctly (`>0.9` / `<0.1`) while the global function
necessarily blends them into one value strictly between the two
(demonstrating the ORIGINAL function's real, non-per-channel,
behavior) -- and a monotonicity check (Dynamic MSE <= Per-channel MSE
<= Global MSE, since each oracle has access to a superset of the
previous one's degrees of freedom).

The original `validation_diagnostics01.py` is left completely
unmodified.

---

## A5. Validation split -- design principle adopted, utility built

**Claim**: the original track selected every checkpoint using the FULL
validation split's forecast MSE; any later claim of an "untouched
holdout" carved from that same split after the fact would be invalid.

This is a correct methodological point, not a code bug to grep for.
Agreed and adopted: new controlled runs in this track (Phase B/C) will
report BOTH the Val-Select number (used for checkpoint selection, same
convention as the original track) and, separately, a genuine
chronological Val-Holdout number that no selection decision ever saw.

**Built**: `utils/validation_split01.py` --
`chronological_select_holdout_indices(n_windows, seq_len, pred_len,
select_fraction)` returns disjoint, chronologically-ordered
(Val-Select, Val-Holdout) window-index lists separated by a purge gap
of `seq_len + pred_len` windows -- sized so that no window's
input-or-target span (`[i, i+seq_len+pred_len)`) can cross the
boundary between the two splits, verified by
`verify_no_window_overlap`. 5 unit tests in
`tests/test_validation_split01.py`, including a direct overlap-detection
check (feeding genuinely overlapping indices and confirming the
verifier catches it, not just rubber-stamping) and an edge case where
too few windows exist for the requested split (purge gap consumes
everything -> holdout is empty, not silently non-empty-and-overlapping).

This utility is not yet wired into a training script (the original
track's checkpoint-selection convention -- select on the full
validation split -- is kept for Phase B's RNG-corrected reproduction,
so Phase B's own result is comparable to the original track's); it
will be used by Phase C's final-comparison reporting, where it matters
that the reported Val-Holdout number was never used for any choice.

---

## Status of each audit item

| Item | Status |
|---|---|
| A1 batch-order confound | Confirmed against real artifacts, fixed, unit-tested |
| A2 Base Forecaster TEST read | Confirmed (scope: all 12 cells, not just Solar), fixed going forward, existing checkpoints kept (val-only selection was valid) |
| A3 GO/STOP incomplete-data bug | Confirmed by live reproduction, fixed, unit-tested |
| A4 mislabeled per-channel oracle | Confirmed by reading code+caller, fixed, unit-tested |
| A5 validation split design | Adopted, utility built and unit-tested, not yet wired into a trainer |

## Full existing test-suite regression check

Ran immediately after writing all Phase A code (no existing file was
modified -- only new files added):
`python -m pytest tests/ -q` -- result logged separately once complete
(see `results/TRACK-P100-QA-CAUSAL-VALIDATION01/full_suite_regression_check.txt`).

## Next steps (Phase B)

Phase B's 10 RNG-corrected paired runs (5 representative cells x 2
arms) use the new `*_rngfixed.py` trainers, write to
`results/TRACK-P100-QA-CAUSAL-VALIDATION01/rng_corrected/`, and queue
on GPU1 behind the still-running original grid (no parallel GPU jobs).
Representative cells per the spec: ETTh1_192/V5, ETTh1_336/Hard,
Weather_720/V5, Weather_336/Joint, Solar_96/V5.
