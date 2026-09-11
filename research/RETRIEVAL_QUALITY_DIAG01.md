# EXP-RETRIEVAL-QUALITY-DIAG01 — why does retrieval fail to help at ETTh1 H720?

Diagnostic only. **No Stage-1 training, no loss/scorer/Oracle-definition
change, no hyperparameter tuning, no rerun.** Existing checkpoints and
retrieval caches were reused; only evaluation was computed. Script:
`scripts/diag_retrieval_quality01.py`. Results:
`results/EXP-RETRIEVAL-QUALITY-DIAG01/`.

Fusion convention analysed throughout:
`Y_final = (1-λ)·Y_base + λ·Y_ret` (= `Y_base + λ·(Y_ret − Y_base)`).

---

# 1. Executive Summary

**The headline conclusion of the previous experiments — "at H720 retrieval
does not help" — is wrong. It was an artifact of *jointly training the base
predictor with retrieval*, not a property of retrieval.** When one
canonical Base predictor is held fixed for every arm, retrieval improves
H720 forecasting for **every** arm tested.

Direct answers to the six required questions:

1. **Is `Y_ret` itself bad at H720?** As a *standalone forecast*, yes:
   Ret MSE 0.584–0.731 vs Base 0.483. But that is also true at H96
   (0.407–0.551 vs 0.393), so it does not distinguish the two horizons.
2. **Is `Y_ret` pointing the wrong way at H720?** **No — the opposite.**
   Correction cosine is *higher* at H720 (+0.116…+0.201) than at H96
   (+0.119…+0.162) for the Individual arms, and the fraction of
   query×channels where the oracle mixing weight is exactly 0 is *lower*
   at H720 (23.5–27.6%) than at H96 (30.8–35.5%). Retrieval is useful for
   **more** queries at H720, not fewer.
3. **Does an oracle λ rescue retrieval?** Yes, and so does a far weaker,
   fully deployable one. Oracle-λ: 0.432–0.445 at H720 (vs Base 0.48279).
   A single **validation-fitted global λ** (one scalar, no leakage):
   **0.4636–0.4789 — every arm beats Base.**
4. **Stage-1, retrieval value, or gate/fusion?** **Primarily the Stage-2
   training/fusion procedure; secondarily Stage-1.** See §10.
5. **Core H96 vs H720 difference?** Not retrieval quality — it is how
   badly joint Stage-2 training damages the base predictor. Co-trained
   base head alone: H96 0.395–0.417 (≈ canonical 0.393), H720
   **0.568–0.693** (canonical 0.483, i.e. up to **+0.21 MSE damage**).
6. **Why is KL+Asym downstream-friendlier despite lower Recall?** At H96 it
   is the best learned arm on *every* retrieval-quality metric measured
   (§9). At H720 it is **not** better than the Oracle-trained arms — so the
   historical "low Recall but better downstream" story reproduces at H96
   only.

**Verdict: FUSION/GATE BOTTLENECK (dominant) + STAGE-1 BOTTLENECK
(secondary).** Not an Oracle-definition problem, not a
retrieval-signal-absent problem.

---

# 2. Artifact / protocol verification

| Check | Result |
|---|---|
| Canonical Base-only reproduced, H96 | 0.392976 (expected 0.39298) ✓ |
| Canonical Base-only reproduced, H720 | 0.482785 (expected 0.48279) ✓ |
| Same `Y_true` for every arm | asserted `torch.equal(cache Y_q, canonical Y_true)` for every cache arm ✓ |
| Same query set / order | all splits loaded `shuffle=False`; H96 n_test=2785, H720 n_test=2161 ✓ |
| Same `Y_base` for every arm | one canonical Base-only checkpoint per horizon ✓ |
| Full memory, self-only | 8449 (H96) / 7201 (H720) candidates, `relation_top_n=1` ✓ |
| Future leakage in learned arms | learned arms use only their own score; `Y_q` enters only Oracle definitions and diagnostic bounds ✓ |
| Oracle definition stationarity | Set/Individual Oracles weighted by ONE FIXED reference scorer (canonical `S0_wce` Stage-1), so the Oracle never depends on any experiment's own encoder ✓ |
| Shape convention | `[N, H, C]`, C=7 ✓ |
| Zero-norm handling | cosine computed only where both norms > 1e-12; excluded fraction recorded (≈0 in all arms) ✓ |

**Two artifact findings recorded rather than worked around:**

- **`KL+Asym[s2ls]` and `KL+Asym[e2]` are bit-identical.** Their full
  `model_state_dict`s compare `torch.equal == True`, so they are the same
  model stored under two names. Both were run; identical numbers are
  expected, not a bug. Reported as one row.
- **The historical `KL+Asym H720 Stage-2 MSE = 0.478067` could not be
  located anywhere in the repository** (`research/*.md` contains only the
  Recall@10 table, `RESEARCH_CONTEXT.md:266`). Per the brief, the old
  number is therefore **not used** in any comparison here; the KL+Asym
  Stage-1 checkpoint was instead re-run through the current common
  harness.

---

# 3. Overall quality table

All values on TEST, canonical `Y_base`, mixture convention.

### ETTh1 H96 — canonical Base-only = **0.39298**

| Arm | Ret MSE | Ret>Base % | Corr.Cos | Neg.Align % | Oracle-λ MSE | λ*=0 % | Global-λ MSE | λ_global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL+Asym | 0.40691 | 36.1% | +0.1619 | 30.8% | 0.32736 | 30.8% | **0.36899** | 0.413 |
| Individual-TF-Asym | 0.41985 | 34.9% | +0.1511 | 32.5% | 0.33010 | 32.5% | 0.37280 | 0.439 |
| Set-TF-Asym | 0.55052 | 23.5% | +0.1192 | 35.5% | 0.33103 | 35.5% | 0.38155 | 0.229 |
| *Individual-OP-Asym* ᵃ | 0.42977 | 35.0% | +0.1561 | 31.6% | 0.32868 | 31.6% | 0.37373 | 0.424 |
| *Set-OP-Asym* ᵃ | 0.48136 | 29.7% | +0.1315 | 34.2% | 0.33606 | 34.2% | 0.38121 | 0.271 |
| **Oracle-Individual** | 0.14766 | 96.4% | +0.7106 | 0.1% | 0.14327 | 0.1% | 0.14766 | 1.000 |
| Oracle-Individual (uniform) | 0.13807 | 97.8% | +0.7310 | 0.1% | 0.13483 | 0.1% | 0.13807 | 1.000 |
| **Oracle-Set** | 0.10303 | 99.8% | +0.8340 | 0.0% | 0.10220 | 0.0% | 0.10303 | 1.000 |

### ETTh1 H720 — canonical Base-only = **0.48279**

| Arm | Ret MSE | Ret>Base % | Corr.Cos | Neg.Align % | Oracle-λ MSE | λ*=0 % | Global-λ MSE | λ_global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL+Asym | 0.59739 | 27.1% | +0.1553 | 27.6% | 0.44232 | 27.6% | 0.47397 | 0.386 |
| Individual-TF-Asym | 0.59494 | 30.7% | +0.1836 | 25.7% | 0.43363 | 25.7% | 0.46528 | 0.315 |
| Set-TF-Asym | 0.73146 | 20.5% | +0.1159 | 35.1% | 0.44509 | 35.1% | 0.47892 | 0.186 |
| *Individual-OP-Asym* ᵃ | 0.58391 | 31.4% | +0.2014 | 23.5% | 0.43214 | 23.5% | **0.46358** | 0.228 |
| *Set-OP-Asym* ᵃ | 0.61190 | 29.2% | +0.1820 | 25.3% | 0.43447 | 25.3% | 0.46757 | 0.313 |
| **Oracle-Individual** | 0.31428 | 93.6% | +0.5704 | 0.0% | 0.30540 | 0.0% | 0.31428 | 1.000 |
| Oracle-Individual (uniform) | 0.29685 | 96.1% | +0.5991 | 0.0% | 0.29132 | 0.0% | 0.29685 | 1.000 |
| **Oracle-Set** | 0.25633 | 99.9% | +0.6923 | 0.0% | 0.25513 | 0.0% | 0.25633 | 1.000 |

ᵃ On-policy Set/Individual rows are **secondary**: the on-policy Set arm's
own Stage-1 ranking metrics were shown to be endogenous
(`research/AUDIT_ORACLE_RANK_GAIN01.md` §4.4). Their *retrieval-quality*
numbers here are unaffected by that artifact (they depend only on the
produced `Y_ret`), but they are still marked separately.

**Every learned arm's Global-λ MSE beats Base at BOTH horizons.**

---

# 4. Correction alignment

`C_true = Y_true − Y_base`, `C_ret = Y_ret − Y_base`, cosine per
query×channel.

- Mean cosine is **positive for every arm at both horizons** (+0.116 …
  +0.201 learned; +0.57 … +0.83 Oracle).
- Strongly-negative alignment (cos < −0.5) is a small minority
  (see `H*_summary.csv`, `strong_neg_frac`).
- **The alignment does not degrade with horizon.** For the best learned
  arm (Individual-OP-Asym) cosine *rises* with horizon depth:
  +0.190 (steps 1–96) → +0.181 → +0.200 → **+0.198** (steps 337–720).

So "retrieval points the wrong way at long horizons" is **falsified**.

---

# 5. Oracle-λ analysis

`λ* = clip(⟨e,d⟩/⟨d,d⟩, 0, 1)` per query×channel, `d = Y_ret − Y_base`,
`e = Y_true − Y_base`. This uses the test target and is a **diagnostic
upper bound, not a deployable model.**

| | H96 | H720 |
|---|---|---|
| Oracle-λ MSE (learned arms) | 0.3274–0.3361 | 0.4321–0.4451 |
| gain vs Base | −0.057 … −0.066 | −0.038 … −0.051 |
| λ*=0 fraction (learned) | 30.8–35.5% | **23.5–27.6%** |
| λ*=1 fraction, λ interior | see `H*_summary.csv` | — |

The λ*=0 fraction — the share of query×channels where the best possible
thing to do is ignore retrieval entirely — is **lower at H720**. Combined
with §4, the retrieval signal at H720 is, if anything, *more* broadly
usable than at H96.

---

# 6. Harmful / catastrophic query analysis

| | harmful % (λ=1) | harmful % (global λ) | p5 gain (global) | p95 gain (global) |
|---|---:|---:|---:|---:|
| H96 learned arms | 63.9–76.5% | 44.2–46.5% | −0.082 … −0.111 | +0.147 … +0.229 |
| H720 learned arms | 68.6–79.5% | 32.3–46.6% | −0.047 … −0.108 | +0.090 … +0.146 |
| H720 Oracle-Set | 0.1% | 0.1% | +0.030 | +0.613 |

Pure retrieval (λ=1) hurts the majority of query×channels at both
horizons — retrieval is *not* a better standalone forecast. With a single
global λ the harmful share drops to roughly a third–half, and the upside
tail (p95) exceeds the downside tail (|p5|) in every learned arm at both
horizons, which is why the mean improves.

**H720 is not driven by a few catastrophic queries**: the p5/p95 spread at
H720 is *narrower* than at H96. It is a broad, mild effect, not a tail
event.

---

# 7. Horizon-wise analysis (H720)

Individual-OP-Asym (best learned arm), canonical base, global λ:

| steps | Base | Ret | mixture | cos | Ret>Base % |
|---|---:|---:|---:|---:|---:|
| 1–96 | 0.3762 | 0.4818 | **0.3595** | +0.190 | 28.7% |
| 97–192 | 0.4325 | 0.5524 | **0.4190** | +0.181 | 32.2% |
| 193–336 | 0.4607 | 0.5634 | **0.4451** | +0.200 | 33.7% |
| 337–720 | 0.5302 | 0.6250 | **0.5077** | +0.198 | 35.0% |

Oracle-Set on the same bins: Ret MSE 0.2495 / 0.2513 / 0.2462 / 0.2631 vs
Base 0.376 / 0.433 / 0.461 / 0.530 — the Oracle's advantage **grows** with
horizon depth (Ret>Base rises 78.9% → **99.4%**).

**There is no far-future collapse.** Retrieval helps in every bin, and the
achievable headroom is largest at the deepest horizon. Per-step data:
`H720_stepwise.csv`.

---

# 8. Channel-wise analysis

Full table: `H{96,720}_channelwise.csv` (per-channel Base/Ret MSE,
Ret>Base %, correction cosine, Oracle-λ MSE, λ*=0 fraction, and the
per-channel validation-fitted λ with its test MSE). H720 degradation is
**not** concentrated in one channel — the pattern (Ret MSE > Base MSE,
positive cosine, Oracle-λ well below Base) repeats across all 7 channels.

---

# 9. KL+Asym comparison

**At H96, KL+Asym is the best learned arm on every metric measured** —
despite its historically lower Recall@10:

| Metric | KL+Asym | best Oracle-trained arm |
|---|---:|---:|
| Ret MSE | **0.40691** | 0.41985 (Ind-TF-Asym) |
| Correction cosine | **+0.1619** | +0.1511 |
| λ*=0 fraction | **30.8%** | 32.5% |
| Oracle-λ MSE | **0.32736** | 0.33010 |
| Global-λ MSE | **0.36899** | 0.37280 |
| mean damage when harmful | **−0.1144** | −0.1278 |

This is a concrete, reproduced instance of the project's recurring
"Recall does not predict downstream" pattern: a retriever with *lower*
Recall produces candidates whose *aggregate* is closer to the truth and
whose correction direction is better aligned.

**At H720 the story does not hold**: KL+Asym's Global-λ MSE (0.47397) is
the *worst* of the learned arms except Set-TF, and Individual-OP-Asym
(0.46358) is better on every column. So "KL is downstream-friendlier" is
an **H96-specific** finding, not a general one.

---

# 10. Final diagnosis

The decisive evidence is a decomposition ladder
(`results/EXP-RETRIEVAL-QUALITY-DIAG01/decomposition_ladder.json`), all on
the same test queries:

### ETTh1 H720 (canonical Base-only = 0.48279)

| Arm | ① co-trained base ALONE (λ=0) | ② co-trained base + learned gate | ③ canonical base + learned gate | ④ canonical base + global λ | ⑤ canonical base + Oracle λ |
|---|---:|---:|---:|---:|---:|
| Individual-TF-Asym | **0.69290** | 0.53122 | 0.48109 | 0.46528 | 0.43363 |
| Set-TF-Asym | **0.64354** | 0.56855 | 0.52643 | 0.47892 | 0.44509 |
| Individual-OP-Asym | **0.56829** | 0.49814 | 0.46929 | 0.46358 | 0.43214 |
| Set-OP-Asym | **0.63007** | 0.51884 | 0.47788 | 0.46757 | 0.43447 |

### ETTh1 H96 (canonical Base-only = 0.39298)

| Arm | ① | ② | ③ | ④ | ⑤ |
|---|---:|---:|---:|---:|---:|
| Individual-TF-Asym | 0.41004 | 0.37754 | 0.37758 | 0.37280 | 0.33010 |
| Set-TF-Asym | 0.41711 | 0.40008 | 0.39354 | 0.38155 | 0.33103 |
| Individual-OP-Asym | 0.39740 | 0.37398 | 0.37370 | 0.37373 | 0.32868 |
| Set-OP-Asym | 0.39507 | 0.38294 | 0.38225 | 0.38121 | 0.33606 |

**Reading the ladder.**

1. **① is the dominant term at H720 and is nearly absent at H96.** A base
   head trained *jointly with retrieval* ends up **0.086–0.210 MSE worse**
   than the same architecture trained alone (0.568–0.693 vs 0.483). At H96
   the same damage is only 0.002–0.024. This single column explains why
   H720 arms appeared to "lose to Base-only" in
   `EXP-ORACLE-RANK-GAIN01`: they were being compared against a base
   predictor that had been trained *without* the handicap.
2. **① → ② shows retrieval repairing its own damage** (−0.11 … −0.16 at
   H720) but never fully.
3. **② → ③ isolates the base-head damage** (−0.04 … −0.05 at H720).
4. **③ → ④ isolates the learned gate.** At H96 the learned gate ≈ a single
   global scalar (differences ≤0.012). **At H720 the learned per-query
   gate is consistently WORSE than one global scalar** (up to +0.047),
   i.e. the gate's query-conditional behaviour is actively harmful at long
   horizons.
5. **④ → ⑤ is the remaining gate headroom** (−0.03 at both horizons).
6. **Stage-1 headroom is large and unclaimed.** At H720 the best learned
   arm under an *oracle* gate reaches 0.4321 (gain 0.051 over Base), while
   Oracle-Set retrieval reaches 0.2563 (gain 0.227). The learned
   retrievers therefore capture roughly **22%** of the available
   retrieval headroom.

### Classification

- **FUSION/GATE BOTTLENECK — dominant.** The Stage-2 procedure (joint base
  training + a learned per-query scalar gate) destroys more value at H720
  than retrieval adds. Sub-components: base-head corruption during joint
  training (largest), then the learned gate under-performing a single
  global scalar.
- **STAGE-1 BOTTLENECK — secondary but large.** Even with an oracle gate,
  learned retrievers realise only ~22% (H720) of the Oracle's headroom.
- **NOT an ORACLE-DEFINITION bottleneck.** Both true Oracles are
  enormously better than Base at both horizons, with a fixed,
  checkpoint-independent reference scorer.
- **NOT a retrieval-signal-absent case (CASE D is rejected).** Even a
  single validation-fitted scalar λ makes every learned arm beat Base at
  both horizons.

Mapping to the pre-registered cases: **CASE C**, with a strong CASE B
component. CASE A and CASE D are both falsified.

### Consequence for previously reported conclusions

> `EXP-ORACLE-RANK-GAIN01` / `AUDIT_ORACLE_RANK_GAIN01.md` state that at
> H720 "every retrieval arm is worse than Base-only", corroborated across
> two Stage-1 populations. **That statement remains true as a description
> of that specific Stage-2 procedure, but its natural interpretation —
> "retrieval is useless at H720" — is now falsified.** With the base
> predictor held fixed, retrieval helps at H720 for every arm, by
> 0.004–0.019 MSE with a single deployable global λ and by 0.038–0.051
> with an oracle λ.

---

# 11. Recommended next experiment

Proposed only; **nothing was launched.** In priority order:

1. **Frozen-base Stage-2 (cheapest, highest value).** Load the canonical
   Base-only checkpoint, freeze it, and train ONLY the fusion module per
   arm. This directly removes the dominant ① term. Cost: ~10 runs ×
   seconds. It also revives the design the user originally specified in
   `EXP-ORACLE-SCRATCH-FROZENBASE01` and later revised away — the present
   evidence says that original instinct was correct for H720.
2. **Gate ablation at H720.** Compare, on the canonical base: fixed
   global λ (already computed) vs the learned scalar gate (already
   computed) vs a channel-wise global λ (already computed, in
   `H*_channelwise.csv`). No training needed; the numbers exist. The open
   question is *why* the query-conditional gate underperforms a constant
   at H720 — a gate-input/capacity question.
3. **Only after 1–2**: revisit Stage-1, since ~78% of the H720 retrieval
   headroom is still unclaimed. Not worth attacking while a larger,
   cheaper Stage-2 loss is on the table.

---

## Files

`results/EXP-RETRIEVAL-QUALITY-DIAG01/`: `H96_summary.csv`,
`H720_summary.csv`, `H96_channelwise.csv`, `H720_channelwise.csv`,
`H96_querywise.csv`, `H720_querywise.csv`, `H96_horizon_bins.csv`,
`H720_horizon_bins.csv`, `H96_stepwise.csv`, `H720_stepwise.csv`,
`H96_artifact_manifest.json`, `H720_artifact_manifest.json`,
`decomposition_ladder.json`. Plots were optional and were not generated.
