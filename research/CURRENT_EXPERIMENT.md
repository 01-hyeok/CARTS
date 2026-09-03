# CURRENT_EXPERIMENT.md

Status: **PRIORITY REORDERED 2026-09-02 — implementation plan pending user approval; NO GPU RUN STARTED**

## Priority order (revised 2026-09-02)

1. **EXPERIMENT 1 — Test-matched Oracle Intervention** (ETTh1 H96, H720)
2. **EXPERIMENT 1-B — Full-memory Oracle Headroom** (after 1)
3. **EXPERIMENT 2 — Learn set-level utility in Stage-1** (ETTh1 H96) —
   *conditional*: only if Experiment 1 shows Set Oracle also lowers Stage-2 MSE
4. **A/B small-N memorization**, **cross-dataset feasibility**, **Track C E2E**
   — retained below, deprioritized, not deleted

Branch rule from the user: if Experiment 1 gives *Set A improves but Stage-2 does
not*, **skip Experiment 2 and run Track C E2E first**.

The ETTh1 Stage-2 wiring-fixed rerun requirement is unchanged and still blocking
for any ETTh1 downstream claim.

## New motivating observation (2026-09-02) — [repo] verified

ETTh1 H720, corrected-wiring Stage-2 (`logs/stage2_learned_score_corrected_selection/`):

| Arm | Stage-1 R@10 | Stage-2 test final MSE |
|---|---|---|
| KL + Asymmetric | 0.0608 | **0.478067** |
| WCE + Asymmetric | 0.0942 (+55%) | **0.505433** (worse) |

Higher recall, worse downstream MSE. So the question is not "can Recall@10 go
higher" but **"what Top-K target does Stage-2 actually want?"**

---

# Research Question

Is the future-derived Oracle retrieval target learnable at all from past-only
information, and if not, can retrieval instead be trained directly against
forecasting MSE?

This replaces the earlier question "how do we raise Recall@10 further?", which
`RESEARCH_CONTEXT.md` B2/B4/B12 showed to be the wrong target.

# Hypothesis

Three competing hypotheses, deliberately not pre-ranked:

- **Q1 (predictability):** future-derived Oracle utility does not generalize from
  past-only input. Phase C (B15) is direct evidence for this.
- **Q2 (capacity):** the asymmetric scorer is simply under-capacity.
- **Q3 (objective):** Oracle-membership matching is the wrong intermediate
  objective; forecasting-MSE-aligned end-to-end retrieval is the right one.

# Motivation

Phase C showed that direct Oracle imitation sits at random/uniform on validation
(TeacherSetRecall@10 0.0847–0.1363 vs random 0.10; imitation loss 4.589–4.621 vs
ln(100) ≈ 4.605) — **for the Individual target as well as the Set target**.
Because both failed, the failure cannot be attributed to the pointwise scorer's
inability to express a set objective. Before adding another loss, establish
whether the target is learnable at all.

---

# Experiment A/B — Feasibility / scorer capacity (small-N memorization)

Directory: `logs/memorization/`.

**Setup.** 32–64 fixed *train* queries, fixed P100, healthy WCE encoder frozen.

**Targets:** (1) Individual Oracle, (2) Set Oracle.

**Scorers:** (1) asymmetric, (2) `PairwiseScorer(feature_type='pair4')` from
`layers/pairwise_scorer.py`, whose pair4 feature is
`[z_q, z_k, z_q − z_k, |z_q − z_k|]`.

**Hard constraint:** reuse the scorer already wired into Stage-1. Do **not** wire
a new `UtilityPairScorer` into Stage-1 for this experiment.

**Reference baseline:** random TeacherSetRecall@10 in P100 = **0.10**.

## Decision Criterion (A/B)

| Outcome | Reading | Next action |
|---|---|---|
| **A** — train ≈ 0.1 too | Not a capacity result yet | Check target / index / gradient / update wiring **first**; only then suspect scorer functional capacity |
| **B** — train 0.8–1.0, VAL ≈ 0.1 | Not capacity — generalization / past-only predictability | Q1 becomes the leading hypothesis |
| **C** — asymmetric fails, pairwise MLP improves on train *and* VAL | Asymmetric scorer capacity bottleneck | Q2 supported |
| **D** — pairwise MLP memorizes train but fails VAL | Representation / predictability / generalization bottleneck | Q1 supported, Q2 rejected |

---

# Experiment — Cross-dataset feasibility

Is the failure ETTh1-specific? Same Individual-Oracle direct imitation on
**ETTh1 / ETTm1 / Weather** at **H96**, using the most expressive scorer whose
train fit was confirmed in A/B.

- Only ETTh1 low, others high → dataset-specific predictability / density issue.
- All near random → the problem formulation (predicting future-Oracle membership
  from past-only input) is itself the difficulty.

---

# Track C — End-to-End forecasting-aligned retrieval

Runs in parallel with, and independent of, Oracle imitation. Writes to its own
separate directory.

**Question.** Can retrieval that never matches Oracle candidates still lower
final forecasting MSE?

**Method.** Soft retrieval, no differentiable hard Top-K in this first pass:

```
p_i       = Softmax(s_i / tau_H)
Y_ret     = sum_i p_i * Y_i
L_E2E     = MSE(Y_final, Y_q)
```

`tau_H` uses the Phase 0 calibrated values: 0.015 (H96), 0.015 (H192),
0.01 (H336), 0.02 (H720).

## Required control arms

| Arm | Scorer | Retrieval | Stage-2 |
|---|---|---|---|
| E0 | — | base only | train |
| E1 | frozen | hard Top-10 | train |
| E1-soft | frozen | soft | train |
| E2 | trainable | soft | train |

The two contrasts that carry the result:

- `E1-soft − E1` = effect of the hard→soft retrieval representation change alone
- `E2 − E1-soft` = **pure effect of training the scorer with forecasting gradient**

**Mandatory across every arm:** same base forecaster checkpoint, same Stage-2
initialization, same seed, same split. Past experiments measured
`corr(base_mse, final_mse) ≈ +0.958`, so a differing base checkpoint contaminates
any retrieval-effect reading. This control is not optional.

## Decision Criterion (Track C)

| Case | Observation | Reading |
|---|---|---|
| 1 | soft MSE ↓ **and** hard Top-10 AggregateMSE ↓ **and** hard Stage-2 MSE ↓ | forecast-aligned E2E retrieval works |
| 2 | soft objective ↓ only, hard Top-10 unchanged | objective learns; soft→hard transfer is the bottleneck |
| 3 | no improvement | representation / predictability limit, or a limit on retrieval signal Stage-2 can exploit |
| 4 | Oracle imitation fails **but** E2E succeeds | **high-value result** — the Oracle-membership intermediate target was itself misaligned with forecasting |

---

# Required rerun — ETTh1 Stage-2 under fixed wiring

Independent of the above and **blocking for any downstream claim about ETTh1**.

Re-evaluate {KL, WCE} × {Cosine, Asymmetric, MLP} × {H96, H192, H336, H720} under
identical Stage-2 conditions with the wiring fix in place. Until then ETTh1
Stage-2 is a pre-fix historical diagnostic only; Weather is the valid downstream
evidence. See `RESEARCH_CONTEXT.md` → *Invalidated Results — Do Not Cite*.

---

# Controlled Variables

Held fixed unless named as the changed variable above:

- dataset and split protocol; `seq_len = pred_len`; `--features M`
- `--top_k 10`, `--relation_top_n 3`
- model size (`d_model 128`, `n_heads 4`, `e_layers 2`, `d_layers 1`, `d_ff 256`)
- frozen healthy WCE Stage-1 encoder; fixed P100 candidate pool
- per-horizon τ from Phase 0
- optimizer, LR schedule, epochs, patience, batch size, `--seed`
- checkpoint selection rule; evaluation protocol and metric definitions
- **Track C additionally:** identical base forecaster checkpoint and Stage-2 init

# Dataset

ETTh1 (A/B, Track C, rerun); ETTh1 + ETTm1 + Weather (cross-dataset, H96 only).

# Prediction Horizons

A/B: H96 and H720. Cross-dataset: H96. Track C and the rerun: 96 / 192 / 336 / 720.

# Metrics

- TeacherSetRecall@10 (against 0.10 random baseline) and imitation loss
  (against ln(100) ≈ 4.605)
- `I` individual MSE@10, `A` HardAggregateMSE@10, `V` candidate variance —
  reported together, since `I = A + V`
- Stage-2 final MSE, base MSE, retrieval gain, λ
- Recall@1/5/10 retained as a *diagnostic*, explicitly not as the objective

# Expected Outcomes

Enumerated per track in the decision-criterion tables above. Each outcome was
specified before running.

# Decision Criterion

Per the tables above. Reminders that apply to all tracks:

- Seed noise is ≈0.01 MSE; smaller differences are not evidence.
- A metric improvement does not establish a mechanism.
- Nothing in the *Do Not Conclude* table of `RESEARCH_CONTEXT.md` may be asserted
  on the strength of these runs.
