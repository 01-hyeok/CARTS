# EXP-TEACHER-FORCING-DIAG01 — Report

**No new training.** Reuses the existing C0/R2 checkpoint
(`EXP-TOPTAIL-RANK01/R2`) verbatim. Question: before committing to
on-policy retraining (`EXP-ONPOLICY-PREFIX01`), does the SAME frozen R2
checkpoint reproduce the Set Oracle's actual next choice well WHEN GIVEN
the correct oracle prefix (Mode A), and how does that compare to its own
free-running behavior (Mode B)?

**Scope limitation (documented, affects only the Stage-2 rows below, not
the rank/NDCG/regret/divergence diagnostics):** this diagnostic forces
only channel 0's selections into Stage-2 for both arms (unlike this
project's official Stage-2 evaluations, which force all 7 channels). Its
Stage-2 MSE numbers are therefore **not directly comparable in absolute
terms** to C0's canonical 0.39526 or any other experiment's Stage-2
number — only the RELATIVE comparison between this diagnostic's own two
arms (same single-channel scope for both) is valid, and is treated as
such throughout.

---

## Comparison table (test split, 200-query subsample for detailed
diagnostics, full test split for Stage-2 within the single-channel scope
above)

| Metric | Oracle-prefix | Free-running |
|---|---:|---:|
| Oracle next-choice rank mean | 1148.8 | N/A (no single "oracle rank" defined for free-running's own picks) |
| Top-1 accuracy | 0.25% | — |
| Top-5 containment | 1.8% | — |
| Top-10 containment | 3.25% | — |
| Top-50 containment | 12.2% | — |
| NDCG@10 | 0.768 | — |
| NDCG@50 | 0.790 | — |
| Spearman (true top-1%) | 0.279 | — |
| mean step regret | 0.254 | 0.376 |
| mean divergence rate (vs. oracle's own picks) | — | 99.5% |
| first divergence step (mean) | — | 0.01 (diverges essentially immediately) |
| mean prefix overlap (set, vs. oracle) | — | 2.7% |
| final A_weighted | 0.246 | 0.869 |
| Stage-2 Final MSE (single-channel scope only) | 0.322 | 0.377 |

## Per-step detail (selected steps, oracle-prefix mode)

| step | rank_mean | rank_median | NDCG@10 | top1_acc | regret_mean |
|---|---:|---:|---:|---:|---:|
| t=1 | 120.3 | 70 | 0.965 | 1.0% | 0.990 |
| t=2 | 488.5 | 236 | 0.756 | 1.0% | 0.249 |
| t=5 | 931.9 | 500 | 0.764 | 0.0% | 0.150 |
| t=10 | 2299.2 | 1271 | 0.712 | 0.0% | 0.161 |

## NDCG vs. exact-choice-rank divergence (explicitly required by spec)

**This diagnostic reproduces the EXP-FIRSTANCHOR-DIAG pattern exactly**:
NDCG@10 stays high (0.71–0.96 across all steps) while the Oracle's exact
next choice sits at a poor predicted rank (mean 120–2299 depending on
step, growing worse at later steps) and Top-1 accuracy is near 0
throughout. Per the pre-registered interpretation rule: **the model finds
a generally-good CANDIDATE POOL (high NDCG — it is not ranking randomly)
but cannot resolve the fine-grained distinction needed to identify the
Set Oracle's SPECIFIC greedy choice within that pool.** This is consistent
with — and independently corroborates — `EXP-ORACLE-CHOICE01`'s finding
that a more direct choice-prediction objective (rather than R2's
SmoothL1+pairwise) recovers real value: the underlying representation
already carries a "good pool" signal (visible in NDCG here), and D1's
result shows that directly training toward the Oracle's choice can
extract more of that signal into a better decision than R2's original
surrogate did.

## Case classification

**Not a clean Case A, not a clean Case B, not a clean Case C — but
elements of A, B, and C all present, reported honestly rather than forced
into one.**

- **Case C is clearly and strongly present**: the NDCG-vs-exact-rank
  divergence documented above matches the spec's Case C description
  precisely (good top-region utility quality, poor exact greedy-choice
  resolution) — this is a genuine, well-evidenced finding independent of
  teacher forcing.
- **Case A has partial support**: Oracle-prefix performance is not
  actually "good" in an absolute sense — Top-1 accuracy is ~0.25%, rank
  mean balloons past 1000 by mid-sequence, and this happens EVEN WITH the
  correct prefix — so it would be an overstatement to say "given the
  correct state, the model picks well." This is evidence against
  teacher-forcing mismatch being a SUFFICIENT explanation on its own.
- **Case B also has real support**: within this diagnostic's own paired
  comparison, oracle-prefix is measurably better than free-running on
  every metric that can be compared (regret 0.254 vs. 0.376; final
  A_weighted 0.246 vs. 0.869 — lower is better; single-channel-scope
  Stage-2 0.322 vs. 0.377). Free-running diverges from the oracle's own
  trajectory almost immediately (99.5% divergence rate, first divergence
  at step ~0, prefix overlap collapsing to 2.7%) — the model's own
  self-generated states are visited far less accurately than the oracle's
  states, and behave measurably worse once there. This is exactly the
  state-distribution-mismatch signature `EXP-ONPOLICY-PREFIX01` is
  designed to test.

**Net reading**: teacher-forcing/exposure-bias mismatch is a real,
independently-evidenced contributor (oracle-prefix beats free-running
within this same checkpoint), but it operates ALONGSIDE a
surrogate/decision-resolution weakness (Case C) that is at least as
severe — the model doesn't reproduce the Oracle's exact choice well even
under ideal state. Per the spec, this mixed reading still supports
proceeding with `EXP-ONPOLICY-PREFIX01` (Case B's rationale holds even
if not the SOLE explanation), while noting the Case C finding as
additional context that a future combined objective (on-policy prefix +
a more direct choice-aligned loss, not approved to test this round) may
be needed to address both factors together.

---

Source files: `oracle_prefix_summary_test.json`,
`free_running_summary_test.json`, `full_result_test.json`,
`step_metrics_test.csv`, `metrics.csv`, `checkpoint_fingerprints.txt`,
`working_tree.diff`, `command.txt`, `env.txt`.
