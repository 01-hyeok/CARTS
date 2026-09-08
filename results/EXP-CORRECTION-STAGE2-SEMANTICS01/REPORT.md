# EXP-CORRECTION-STAGE2-SEMANTICS01 (Track B2) — Report

No new retrieval selector trained. B0/base_forecast and the retrieval
encoder/score stay frozen throughout — only a small per-channel
`RetrievalGate` (existing class, unmodified) is newly trained for the C1
arm. Ran in parallel with, independently of, Track A
(`EXP-ONPOLICY-CHOICE01`) and Track B1 (`EXP-CORRECTION-ORACLE-DIAG02`).

**Question**: given the SAME reference retrieval (Top-K membership + alpha
weights, from B0's own existing production score — critically, NOT a new
selector), does fusing the historical CORRECTION aggregate
(`C_ret = Σ alpha_i * r_i`, `r_i = Y_i - B_i`) beat the existing FUTURE
aggregate fusion Stage-2 already uses?

## Arms

| Arm | Top-K selection | Value | Fusion |
|---|---|---|---|
| B0 | none | none | `B_q` |
| F0 | reference retrieval | `Y_i` | current production Stage-2 (unforced) |
| C0 | SAME reference retrieval | `Y_i - B_i` | `B_q + 1.0*C_ret` (fixed γ=1, no training) |
| C1 | SAME reference retrieval | `Y_i - B_i` | `B_q + γ*C_ret`, γ a newly-trained per-channel gate |

## Per-channel results (test split)

| channel | B0 | F0 | C0 | C1 | γ mean (test) | C1 beats F0? | Outcome |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| 0 | 1.4496 | **0.7487** | 0.7557 | 0.7549 | 0.972 | ✗ | B2-D |
| 1 | 0.3245 | **0.2191** | 0.2416 | 0.2311 | 0.666 | ✗ | B2-D |
| 2 | 1.5371 | **0.7717** | 0.7870 | 0.7868 | 0.983 | ✗ | B2-D |
| 3 | 0.2701 | **0.1763** | 0.1923 | 0.1922 | 0.670 | ✗ | B2-D |
| 4 | 0.7304 | **0.5005** | 0.5224 | 0.5198 | 0.661 | ✗ | B2-D |
| 5 | 0.1585 | **0.1333** | 0.1542 | 0.1384 | 0.462 | ✗ | B2-D |
| 6 | 0.0570 | 0.0622 | 0.0806 | **0.0603** | 0.263 | ✓ | B2-B |

## Aggregate (mean across all 7 channels)

| Arm | MSE |
|---|---:|
| F0 | **0.37312** |
| C1 | 0.38337 |
| C0 | 0.39056 |

## Answering the required outcome question

**On 6 of 7 channels, and on the cross-channel aggregate, F0 (existing
future-value fusion) beats BOTH C0 and C1 — Outcome B2-D.** Only channel
6 (the channel with by far the lowest B0 MSE, 0.057 — already the easiest
to forecast) shows C1 beating F0, and even there F0 itself already beats
C0. The learned gate (C1) consistently improves over the fixed γ=1 arm
(C0) on every channel — i.e. adaptive correction strength IS better than
a naive full correction — but this improvement is not enough to close the
gap to F0 on 6 of 7 channels.

Per the pre-registered interpretation rule: **this does NOT reject the
Correction retrieval hypothesis overall** — Track B1's own diagnostic
found the Correction Set Oracle (a genuine upper bound, using the true
query future) DOES beat the Future Set Oracle on 4/7 channels and on the
cross-channel mean. The gap between that finding and this experiment's
negative result is consistent with the spec's own framing: **the
reference Top-K here comes from the EXISTING future-oriented production
retrieval, not from a retrieval process aligned with the correction
objective** — Track B1 already showed Future and Correction Oracles select
different candidate sets. Reusing a future-oriented Top-K for a
correction-value fusion may simply hand the correction fusion the WRONG
candidates to work with, independent of whether the correction VALUE
itself has merit.

## Gate behavior

γ (the learned gate) settled at genuinely different values per channel
(0.263–0.983), not collapsing to a trivial 0 or 1 — `c1_gamma_near_zero_frac`
was 0.0 for the channels checked, meaning the gate is doing real,
per-query-varying work, not defaulting to "ignore the correction" as a
degenerate solution. This, combined with C1 consistently beating C0, is
evidence the gate training itself worked correctly (also independently
confirmed by the positive-control sanity test on synthetic data) — the
negative result is about the retrieval-target mismatch, not a broken gate.

## Conclusion

**Outcome B2-D** (both C0 and C1 worse than F0) on the cross-channel
aggregate and on 6 of 7 individual channels; the single exception (channel
6) is the channel where retrieval contributes least overall. Per the
spec's own interpretation guidance, this is read together with Track B1:
the Correction VALUE has real oracle-level headroom (B1), but this
experiment's FIXED, future-oriented reference retrieval does not let
Stage-2 realize that headroom — the open question this leaves for any
future work (not started automatically, per the STOP rule) is whether a
retrieval process specifically aligned with the correction objective
(rather than reusing the existing future-oriented Top-K) would change this.

---

Source files: `summary.json`, `metrics.csv`, `channel{0..6}_gate_history.json`,
`command.txt`, `config.json`, `sanity_summary.json`, `env.txt`,
`checkpoint_fingerprints.txt`, `working_tree.diff`, `logs/`.
