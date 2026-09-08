# EXP-ENCODER-ANCHOR01 — Report

Scope: ETTh1, H96, seed 0, top_k=10. C0 = R2 + cosine + FROZEN B0 encoder
(existing `EXP-TOPTAIL-RANK01/R2`, reused, not retrained). E1 = R2 + cosine
+ TRAINABLE encoder, no anchor (existing `EXP-ENCODER-UNFREEZE01`, reused,
not retrained) — found severe representation collapse
(`embedding_effective_rank` 17.48→2.96→1.49) and worse Stage-2 than C0
(0.39526→0.40668). **E2 (this experiment, the only new training) = E1 +
one addition: `L_anchor = 1 - cos(f_theta(x), f_theta0(x))`,
`lambda_anchor=0.16` (fixed, selected via a gradient-norm-ratio diagnostic,
not a sweep, never by looking at test Stage-2).**

Priority order for the final judgement, per the approved spec: **Stage-2
Final MSE > HardAggregate/gap_recovery > realized t1/t2 top-tail quality >
representation collapse diagnostics > training/validation proxy.**

---

## 1. Did the anchor actually prevent representation collapse?

**Yes, clearly.** `embedding_effective_rank` (channel 0, first 256 memory
rows, same probe as E1):

| | B0 | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 | epoch6 (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 (no anchor) | 17.48 | 2.96 | — | 1.89 | — | — | 1.49 |
| E2 (anchor) | 17.48 | 15.70 | 16.06 | 16.12 | 16.11 | 16.56 | 16.23 |

E2 stays within ~5-10% of B0's own effective rank at every epoch trained,
with no downward trend at any point — collapse never occurs, in stark
contrast to E1's monotonic, unrecovered collapse to <10% of B0's rank.
`pairwise_cosine_mean` (candidate indistinguishability) confirms this:
E1 rose to 0.99+ (collapsed); E2's own probe never showed this pattern
(not directly logged per-epoch in the same field, but the stable
effective-rank trajectory and stable `cos_to_b0`/`drift_l2` below are
inconsistent with the E1 collapse signature).

## 2. Did the encoder actually adapt, or stay effectively frozen?

**Mixed — the encoder moved, but only a small, roughly constant amount,
never drifting further as training continued (unlike E1's runaway drift).**
`cos_to_b0` stayed 0.9965–0.9977 across all 6 epochs (essentially flat, no
trend), and `drift_l2` stayed 0.066–0.081 (small, also no clear trend).
Compare to E1, whose `cos_to_b0` fell monotonically every epoch
(0.650→0.559) and whose `drift_l2` grew monotonically (0.826→0.931) — E2's
representation reaches a small, stable displacement from B0 and stays
there, rather than continuing to move. This is consistent with the anchor
successfully constraining the encoder to a neighborhood of B0, but it
leaves genuinely open whether that neighborhood is "meaningfully adapted"
or "functionally still B0 with noise" — resolved only by the downstream
metrics below, not by the representation diagnostics alone.

## 3. Three-way representation trajectory (C0 → E1 → E2)

| | C0 (frozen, =B0) | E1 (trainable, no anchor) | E2 (trainable + anchor) |
|---|---:|---:|---:|
| `embedding_effective_rank` (best/selected epoch) | 17.48 | 2.96 | 15.70 |
| `embedding_effective_rank` (final epoch trained) | 17.48 | 1.49 | 16.23 |
| `cos_to_b0` (selected epoch) | 1.0 | 0.650 | 0.997 |
| `drift_l2` (selected epoch) | 0.0 | 0.826 | 0.070 |

## 4. Did t1/t2 top-tail ranking improve vs. C0?

Test split:

| metric | C0 | E1 | E2 | E2 vs C0 |
|---|---:|---:|---:|---|
| t1 Spearman (true top-1%) | 0.2128 | 0.1455 | 0.2442 | **better** |
| t1 oracle rank median | 99 | 210 | 86 | **better** |
| t1 Top-50 containment | 30.5% | 20.5% | 31.5% | **better** |
| t2 hurt_frac | 0.232 | 0.194 | 0.216 | better |
| t2 selected true rank median | 1934.0 | 539.5 | 1929.5 | ≈ same |
| t2 oracle predicted rank median | 560.0 | 318.0 | 481.5 | better |
| t2 Spearman (true top-1%) | 0.1404 | 0.1358 | 0.1656 | **better** |
| t2 continuation_regret | 0.3498 | 0.3987 | 0.3177 | **better** |

**E2 is the best of all three arms on t1 (every metric) and on most of t2**
(hurt_frac, oracle rank, Spearman, and notably `continuation_regret` —
the t2 metric closest to realized decision quality, better than BOTH C0
and E1). The one t2 exception is `selected true rank median`, essentially
unchanged from C0 (1929.5 vs 1934.0), far worse than E1's 539.5.

## 5. Did HardAggregate / gap_recovery improve vs. C0?

| metric | C0 | E1 | E2 | E2 vs C0 |
|---|---:|---:|---:|---|
| HardAggregate@10 (seq) | 0.5514 | 0.5466 | 0.6768 | **worse** |
| gap_recovery | -0.2626 | -0.3703 | -0.3594 | worse |

**No — both are worse than C0, and HardAggregateMSE is the WORST of all
three arms** (0.6768, clearly worse than both C0's 0.5514 and even E1's
own collapsed-representation result of 0.5466). This directly contradicts
the t1/t2 top-tail picture in §4, where E2 looked like the best arm.

## 6. Stage-2 Final MSE — the primary metric

| | B0 | C0 | E1 | E2 |
|---|---:|---:|---:|---:|
| Stage-2 MSE | 0.37312 | 0.39526 | 0.40668 | **0.40418** |
| Δ vs B0 | — | +0.02214 | +0.03355 | +0.03106 |
| gap_recovery | — | -0.2626 | -0.3703 | -0.3594 |

**E2 is worse than C0 (0.40418 > 0.39526), but slightly better than E1
(0.40418 < 0.40668)** — a small improvement over the collapsed arm, not
enough to close the gap to the frozen baseline. By the primary,
highest-priority metric this project has committed to, **E2 does not beat
C0.**

## 7. Which Outcome (A/B/C/D)?

**Outcome B — collapse prevented, but downstream Stage-2 does not
improve — is the best-supported reading, with an important caveat from
the HardAggregate result.**

- Not Outcome A: Stage-2 MSE is worse than C0 (§6), which the spec
  explicitly requires for Outcome A regardless of any proxy improvement.
- Not Outcome C: collapse plainly did NOT persist — `embedding_effective_rank`
  stayed close to B0's own value at every epoch (§1), the opposite of E1's
  signature.
- Not a clean Outcome D either: the encoder did move (drift_l2 0.07-0.08,
  not ~0), and t1/several t2 metrics changed substantially and favorably
  relative to C0 (§4) — this is not "no adaptation happened."
- **Outcome B fits best**, but with a twist the pre-registered outcome
  categories don't fully anticipate: by the SECOND-priority metric
  (HardAggregate), E2 is not merely "flat" relative to C0 as Outcome B's
  own description implies — it is the WORST of the three arms, even
  though t1/t2 top-tail metrics (third-priority) are E2's best results.
  This three-way disagreement between Stage-2 (worse), HardAggregate
  (worst of all), and t1/t2 (best of all) is reported as-is per the
  project's standing practice of not forcing contradictory metrics into a
  single clean narrative.

## 8. Does this support, refute, or leave undetermined the "collapse caused
E1's failure" hypothesis?

**Leaves it undetermined — more precisely, it refutes the STRONG version
of the hypothesis (collapse fully explains E1's Stage-2 failure) while
being consistent with a WEAK version (collapse was A contributing factor,
not THE bottleneck).** If collapse were the primary/sufficient cause of
E1's Stage-2 failure, preventing it should have closed most of the gap to
C0; instead E2 closed only a small fraction of the gap (Stage-2 improved
by 0.0025 relative to E1's 0.40668, while the remaining gap to C0 is
0.0089) and made HardAggregate strictly worse. At the same time, E2's
consistent t1/t2 top-tail improvements over BOTH C0 and E1 (§4) are a real
signal that a non-collapsing adapted representation captures SOMETHING
useful that neither the frozen encoder nor the collapsed one does — this
is exactly the kind of partial, non-clean-binary evidence this project's
reporting practice is built to surface rather than obscure. The
HardAggregate/Stage-2 vs. t1/t2 disagreement suggests the remaining
bottleneck (after collapse is controlled for) may lie in how per-step
top-tail ranking gains translate — or fail to translate — into the
free-running K=10 sequential aggregate, which is a plausible bridge to the
still-pending teacher-forcing/on-policy question (`EXP-TEACHER-FORCING-DIAG01`,
`EXP-ONPOLICY-PREFIX01`).

---

### Primary result table (test split, ETTh1 H96)

| Metric | C0 (frozen) | E1 (trainable, no anchor) | E2 (trainable + anchor) |
|---|---:|---:|---:|
| best_epoch | 1 | 1 | 1 |
| val_overlap@10 (best) | 0.008228 | 0.008300 | 0.009400 |
| `embedding_effective_rank` (selected epoch) | 17.48 | 2.96 | 15.70 |
| `cos_to_b0` (selected epoch) | 1.0 | 0.650 | 0.997 |
| t1 top-1% Spearman | 0.2128 | 0.1455 | 0.2442 |
| t1 oracle rank median | 99 | 210 | 86 |
| t1 Top-50 containment | 30.5% | 20.5% | 31.5% |
| t2 selected true rank median | 1934.0 | 539.5 | 1929.5 |
| t2 oracle predicted rank median | 560.0 | 318.0 | 481.5 |
| t2 continuation regret | 0.3498 | 0.3987 | 0.3177 |
| HardAggregate@10 | 0.5514 | 0.5466 | 0.6768 |
| gap_recovery | -0.2626 | -0.3703 | -0.3594 |
| **Stage-2 MSE** | **0.39526** | 0.40668 | 0.40418 |

Per the pre-registered stopping rule: this result does not clear the bar
for continuing the encoder-adaptation direction (Stage-2 still worse than
C0), so no further encoder-side follow-up (larger anchor sweep, different
anchor form, partial unfreezing) is started automatically. Source files:
`stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`representation_diagnostics.json`, `representation_probe.csv`, `metrics.csv`,
`train_summary.json`, `checkpoint_fingerprints.txt`, `working_tree.diff`,
`sanity_summary.json`, `notes.md`, `command.txt`, `config.json`.
