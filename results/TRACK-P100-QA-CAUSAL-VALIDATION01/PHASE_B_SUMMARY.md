# Phase B summary -- RNG-corrected reproduction, COMPLETE (5/5 pairs)

All 10 runs (5 representative cells x 2 arms) complete. All 5 pairs
pass RNG parity verification (`scripts/verify_rngfixed_pairs01.py`):
identical `init_hash`/`slot_heads_init_hash`/`router_init_hash` and
identical `batch_order_sha256` at every one of the 10 epochs, for
every pair -- the A1 fix works as designed.

## Original (RNG-confounded) vs RNG-corrected, VALIDATION MSE

| Cell/Method | Orig NOQA | Orig QA | Orig delta | Corr NOQA | Corr QA | Corr delta | Direction preserved? |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1_192/V5 | 0.955703 | 0.962265 | +0.006562 | 0.955703 | 0.961763 | +0.006060 | **YES** (NOQA wins both) |
| ETTh1_336/Hard | 1.241653 | 1.224178 | -0.017475 | 1.241653 | 1.228451 | -0.013203 | **YES** (QA wins both) |
| Weather_720/V5 | 0.647742 | 0.618717 | -0.029026 | 0.647742 | 0.624463 | -0.023279 | **YES** (QA wins both) |
| Weather_336/Joint | 0.546094 | 0.535591 | -0.010503 | 0.546094 | 0.537567 | -0.008527 | **YES** (QA wins both) |
| Solar_96/V5 | 0.183358 | 0.182407 | -0.000951 | 0.183358 | 0.180586 | -0.002772 | **YES** (QA wins both) |

**NOQA's own val_mse is byte-identical between Original and
RNG-corrected in all 5 pairs** (0.955703, 1.241653, 0.647742,
0.546094, 0.183358) -- expected and a good internal sanity check: NOQA
never ran the QA-only feature-fitting pass that caused the confound,
so its own training trajectory is untouched by the fix; only QA's
numbers move.

## Reading

The A1 batch-order confound is real and now fixed, but **it is not the
explanation for QA's apparent effect** -- not in direction, at least.
All 5 pairs preserve their original winner (4/5 QA, 1/5 NOQA,
unchanged). Magnitudes shrink somewhat after correction (e.g.
Weather_720's QA margin: -0.029 -> -0.023; Solar_96's: -0.001 -> -0.003,
actually grew slightly) -- the confound appears to have modestly
inflated some margins, not fabricated them. This is evidence (not yet
proof -- 5 of 72 cells, one seed) that QA's improvement survives this
particular confound check.

This does NOT yet answer the deeper causal question this track exists
to investigate: whether QA's benefit comes from genuine query-dependent
quality estimation, versus a simpler global-calibration or
per-channel-trust effect. That requires Phase C (fusion ablation --
F0-F4, Global/Per-channel/Dynamic lambda comparison, shuffled-feature
control, retriever-checkpoint-frozen fusion diagnostic) and Phase D
(seed robustness), neither of which has started yet.

## Status of the causal-validation track

- Phase A (code audit + fixes): COMPLETE.
- Phase B (RNG-corrected reproduction, 5 representative pairs): **COMPLETE**, this document.
- Phase C (fusion ablation, shuffle control, oracle headroom): NOT STARTED.
- Phase D (seed robustness): NOT STARTED.
- Phase E (final causal determination): NOT STARTED, blocked on C/D.
