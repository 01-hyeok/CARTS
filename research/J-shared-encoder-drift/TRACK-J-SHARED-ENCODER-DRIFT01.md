# TRACK-J-SHARED-ENCODER-DRIFT01

**Status: COMPLETE (A0 diagnostic only, ETTh1_720, seed0). Verdict on the
causal question: a clear, robustly-reproduced pattern is observed, but it
is NOT the pattern the "candidate drift cancels useful query adaptation"
hypothesis (Case A) predicts — it is closer to, but distinct from, Case C
(coupled geometry interaction), and in the OPPOSITE direction the spec's
Case C description anticipated.** Both single-sided counterfactuals
(query-only adaptation with candidates frozen at init, St0; candidate-only
drift with the query frozen at init, S0t) are **worse** than the frozen
baseline S00 throughout training and at the final held-out test
evaluation. Yet the actual shared-encoder retrieval (Stt — both sides
moving together) is **better** than S00, St0, AND S0t, at every diagnostic
step from step 20 onward and on the final test set. Query and key
encoder-gradients are measurably, persistently **negatively correlated**
(cos(g_q, g_k) < 0 at 4 of 5 measured points after initialization) —
genuine gradient conflict is present — yet the joint update outcome is
beneficial, not cancelled. Representation collapse is present early
(effective rank drops from 23.9 to 4.5 by step 20) and partially recovers
but never returns to initialization levels. This combination of findings
does not cleanly fit any single pre-registered case; it is reported
as-is rather than forced into one.

## Q1. Was the corrected original MLP baseline exactly reproduced?

Not directly comparable to an existing external number (see
`research/J-shared-encoder-drift/AUDIT.md` item 20 — no prior MLP run
with this exact library exists in this repository). What IS verified:
(a) the step-0 four-way equivalence check (S00==St0==S0t==Stt within
1e-4) passed exactly, confirming the diagnostic machinery is correct at
initialization; (b) the **instrumentation ON/OFF equivalence check passed
exactly** (bit-identical per-batch training KL, bit-identical final
encoder hash) — but only after finding and fixing a real bug: the
gradient-conflict diagnostic originally ran with the model in `.train()`
mode (dropout active), consuming extra draws from the global RNG stream
and silently perturbing every subsequent training step's dropout pattern
between the ON and OFF configurations. Fixed by forcing `.eval()` inside
`gradient_conflict_diagnostic` (dropout off, zero RNG consumption) and
restoring the prior mode afterward. Confirmed with a full re-run: batch-
by-batch training KL identical to 13+ significant figures between ON and
OFF, final model state-dict hash identical.

## Q2. Training KL vs actual Stt full-memory retrieval

Both improve together over the run, but non-monotonically and with a
notable mid-training regression: `val_model_top10_individual_mse` (the
checkpoint-selection metric, which IS the Stt quantity on the val split)
goes 1.9964 (ep1) → **1.9946 (ep2, temporary best)** → 2.0208 (ep3,
worse) → 2.0122 (ep4) → 2.1996 (ep5, worst) → **1.9884 (ep6, new best)**
→ 1.9962 (ep7) → **1.9745 (ep8)** → 1.9745 (ep9) → **1.9689 (ep10,
final best)**. Trajectory-level correlation (62 diagnostic points, train
KL vs Stt retMSE): Pearson r=0.705 (p=1.6e-10), Spearman rho=0.720
(p=4e-11) — a real, statistically clear positive association (both trend
down together across the run as a whole), but the epoch-level
non-monotonicity above shows this is not a step-by-step guarantee: KL
decreased every single epoch while val Stt retMSE worsened in 3 of the 9
epoch-to-epoch transitions (ep1→ep2 barely, ep2→ep3, ep2→ep5 stretch).

## Q3. Does St0 (current query / initial key) improve?

**No — it gets worse and stays worse.** At step 0, St0 == S00 == 2.2317
(all four variants identical, by construction). By step 20, St0 =
2.9327 (retMSE, higher = worse) and it stays in the 2.7-2.98 range for
essentially the entire 2260-step run, never recovering to the step-0
baseline. Confirmed at the final test evaluation: St0 = 1.775 vs S00 =
1.167 (St0 is 52% worse than the frozen baseline). Query-side adaptation,
**evaluated against a candidate space frozen at initialization**, is
harmful in this run, not helpful.

## Q4. Does S0t (initial query / current key) improve or worsen?

**Worsens, and more severely than St0.** S0t rises from 2.2317 (step 0)
to 2.75-3.17 for essentially the whole run (peaking around step 220 at
3.17, the worst point observed for any variant at any step). Final test:
S0t = 1.446 vs S00 = 1.167 (24% worse). Candidate-side drift, evaluated
**against a query frozen at initialization**, is harmful, and more
harmful than query-side drift evaluated against a frozen candidate space.

## Q5. Is Stt worse than St0?

**No — the opposite.** Stt is better than St0 at every diagnostic step
from step 20 onward (e.g. step 2260: Stt=1.970 vs St0=2.848) and at the
final test evaluation (Stt=0.954 vs St0=1.775). This is the central,
unexpected finding of this track: the spec's Case A hypothesis
("candidate drift cancels useful query adaptation, so Stt should end up
worse than or no-better-than St0") predicts Stt <= St0 would NOT hold, or
that Stt would be dragged down toward S0t's level; instead Stt is clearly
the best of all four variants, throughout training and at test time.

## Q6. Does Top-K churn rise before or after retrieval degrades?

Churn is already severe at the very first diagnostic step (step 20):
`Stt` retains only 11.8% of its own step-0 (E0-based) Top-10 neighborhood
(`retention_vs_init` = 0.118), i.e. ~8.8 of the original 10 neighbors are
already gone. This happens in the SAME window (steps 0-20) where St0/S0t
first show their degradation and where effective rank collapses
(23.9→4.5). Churn and the St0/S0t degradation are concurrent from the
very first measurement, not sequential — the 20-step diagnostic
resolution in epoch 1 cannot establish which comes "first" at finer
grain; both are already in their degraded/churned state by the earliest
post-init measurement available.

## Q7. Is candidate displacement larger than query displacement?

**No — query displacement is consistently slightly LARGER.** At the
final step: query displacement mean = 0.795 (cosine distance from E0),
candidate displacement mean = 0.728. This holds throughout training, not
just at the end (e.g. step 60: query=0.558 vs candidate=0.542). The two
are close in magnitude and both grow together, but query-side movement is
the (modestly) larger of the two — the opposite of what a "runaway
candidate drift" story would predict as the dominant driver.

## Q8. Do Oracle Top-10/Top-100 candidates drift more than others?

Partially, and with a U-shape rather than a simple monotonic pattern. At
the final step, mean candidate displacement by utility group: bottom
(worst-utility candidates) = 0.795, oracle_top10 = 0.772,
oracle_top100_excl_top10 = 0.763, **middle = 0.688 (lowest of all four
groups)**. Oracle Top-10 and Top-100 candidates DO drift more than the
"middle" group (consistent with the spec's expectation), but they drift
about the same as — not more than — the worst-utility "bottom" group.
Drift is closer to utility-agnostic-at-the-extremes than "good candidates
move more."

## Q9. Does effective-rank collapse align in timing with retrieval deterioration?

Yes, closely, at the START of training: effective rank drops from 23.9
(step 0) to 4.5 (step 20) — an 81% collapse — in exactly the same window
where St0/S0t first show their degradation. It partially recovers
afterward (7.4 at step 40, 8.2 at step 60, stabilizing around 9.9-10.3 by
the end) but never returns to the step-0 level. Trajectory correlation
(effective rank vs Stt retMSE) is weak and inconsistent in sign between
Pearson (r=-0.222, p=0.08, not significant) and Spearman (rho=0.307,
p=0.015) — the aggregate relationship is not clean, consistent with
collapse being a real, early, partially-persistent phenomenon that
co-occurs with the St0/S0t degradation but does not have a simple
monotonic relationship with Stt's own (improving) trajectory over the
full run.

## Q10. Do query-side and key-side gradients conflict (cos(g_q,g_k) < 0)?

**Yes, persistently, after initialization.** Measured at 5 pre-registered
points (mean cosine over 7 channels, `gradient_conflict.csv`):

| Point | mean cos(g_q, g_k) | std | min | max |
|---|---:|---:|---:|---:|
| step0 (first batch) | **+0.352** | 0.211 | 0.087 | 0.701 |
| 25% of epoch 1 | **-0.312** | 0.259 | -0.786 | 0.040 |
| 50% of epoch 1 | -0.081 | 0.267 | -0.500 | 0.209 |
| end of epoch 1 | **-0.295** | 0.215 | -0.602 | 0.031 |
| best checkpoint (epoch 10) | **-0.177** | 0.192 | -0.374 | 0.073 |

Gradient conflict is positive only at the very first batch (before any
weight update), then goes negative and stays predominantly negative for
the rest of training (4 of 5 points, including the best checkpoint). This
is genuine evidence of query/key optimization conflict at the parameter
level — yet it coexists with Stt (the actual joint update) outperforming
both single-sided counterfactuals. The gradient-decomposition identity
`g_combined == g_q + g_k` was verified exactly (max relative difference
0.00e+00 across every channel and every measured point) — a pure
chain-rule check, not itself evidence for or against conflict, but
confirming the decomposition used to compute `cos(g_q,g_k)` is correct.

## Q11. Most conservative mechanism supported by the data

**None of the four pre-registered cases (A/B/C/D) fits cleanly; the data
support a distinct, specific pattern that should be named on its own
terms rather than forced into one of them.** Summary of what was
checked against each:

- **Case A (candidate drift cancels query gains)** predicts St0 improves
  and Stt is dragged toward S0t/worse. **Not supported** — St0 does not
  improve at all (Q3); Stt is the best variant, not a degraded compromise
  between St0 and S0t.
- **Case B (query update itself harmful, independent of candidate
  drift)** predicts KL down but St0 also up/flat. St0 IS worse throughout
  (consistent with part of this), but Case B additionally predicts Stt
  should also fail to improve — it does not (Stt clearly improves, both
  in-training and at test).
- **Case C (coupled geometry interaction)**, as specified, predicts St0
  and S0t stay roughly near baseline while Stt alone is uniquely bad. The
  data show the OPPOSITE polarity: St0 and S0t are uniquely BAD while Stt
  alone is GOOD. This is a coupling effect in the same general family as
  Case C (the joint trajectory behaves very differently from either
  single-sided counterfactual) but with the sign flipped from what the
  spec anticipated — worth naming distinctly rather than checking the
  Case C box.
- **Case D (representation collapse dominant)**: collapse is real and
  timing-aligned with the onset of St0/S0t degradation (Q9), but Stt
  keeps improving well past the point where collapse stabilizes, and the
  effective-rank-vs-Stt trajectory correlation is weak/inconsistent in
  sign — collapse looks like a contributing early-training phenomenon,
  not the dominant explanation for Stt's sustained improvement.

**Most defensible, conservative statement**: in this ETTh1_720, MLP,
seed0 diagnostic, the useful part of training is a **property of the
joint query+candidate update**, not decomposable into "useful query
adaptation, harmed by candidate drift" (Case A) or "harmful query
update alone" (Case B). Freezing either side alone (at initialization)
measurably hurts retrieval relative to letting both move together, even
though the two sides' gradients are measurably anti-correlated for most
of training. Whatever mechanism makes the coupled update net-beneficial
is not resolved by this diagnostic — it is established as a real,
reproduced (train + held-out test) phenomenon, not explained.

## Q12. Does this justify running A1 Frozen-Key?

**NO** — and this diagnostic actively argues against expecting A1 to
help, though it was not run. A1 (Frozen-Key: freeze candidates,
train only the query encoder) is, by construction, mathematically
equivalent to training under the St0 objective — and St0 is the
worst-performing of the three non-baseline variants observed here, both
during training and at final test. If this pattern reproduces on a
second seed, the evidence points AWAY from Frozen-Key as a fix, not
toward it. This is the opposite of the spec's anticipated "Case A implies
A1 is justified" logic — the pre-registered decision rule fires, but with
the opposite recommendation the spec text assumed as the likely outcome.
**Per the spec's explicit instruction, A1 is NOT run this round
regardless of this conclusion.**

## Baseline configuration (see AUDIT.md for full sourcing)

`relation_encoder_type=mlp`, `relation_self_fill=linear`,
`relation_input_space=relation_teacher_space=relation_value_space=delta_last`,
`tau_t=tau_s=0.1`, `top_k=10`, `batch_size=32`, `lr=1e-3`,
`train_epochs=10`, `patience=5`, `init_seed=loader_seed=0`, checkpoint
criterion = min val `model_top10_individual_mse`. N=7201 candidates
(ETTh1_720 train split), n_probe=256 (deterministic, evenly-spaced val
queries), geometry subset=500 (deterministic, fixed).

## Final result table (test split, best checkpoint = epoch 10)

| Variant | retMSE@10 | Recall@10 | NDCG@10 | Oracle regret | Uniform Agg MSE@10 |
|---|---:|---:|---:|---:|---:|
| S00 (init baseline) | 1.166906 | 0.011998 | 0.948239 | 0.671341 | 0.654366 |
| St0 (query adapts, key frozen) | 1.774666 | 0.004408 | 0.922627 | 1.279102 | 1.225827 |
| S0t (key adapts, query frozen) | 1.445952 | 0.005915 | 0.948735 | 0.950388 | 1.183069 |
| **Stt (actual, shared encoder)** | **0.953599** | **0.020257** | **0.965945** | **0.458034** | **0.641731** |

Every metric agrees: Stt is best, S00 second, S0t third, St0 worst.

## What is established

- The step-0 equivalence check, instrumentation ON/OFF equivalence check
  (after a real bug fix), and gradient-decomposition identity check all
  passed exactly — the diagnostic machinery is verified correct.
- St0 and S0t are both, robustly and consistently (every diagnostic step
  from step 20 onward, every retrieval metric, and the held-out test
  split), worse than both the S00 baseline and the actual Stt.
- Stt is the best of the four variants on every metric, in training and
  at test — the shared, jointly-updated encoder outperforms both
  single-sided counterfactuals and the frozen-at-init baseline.
- Query displacement exceeds candidate displacement throughout training
  (Q7) — candidates are not drifting disproportionately more than
  queries.
- Query/key gradients are measurably anti-correlated for most of
  training (Q10), a real, reproducible signal of optimization conflict
  at the parameter level, coexisting with the joint update's net benefit.
- Representation collapse (effective rank 23.9→4.5) occurs early and only
  partially recovers, timing-aligned with the onset of St0/S0t
  degradation.
- Utility-group candidate displacement is U-shaped (bottom and oracle-top
  groups drift more than the middle group), not monotonic in utility.

## What is NOT established

- WHY the coupled update is net-beneficial despite anti-correlated
  gradients and despite both single-sided counterfactuals being harmful
  — no mechanism is identified, only the outcome pattern.
- Whether this reproduces on other seeds (only seed0 was run, per spec);
  the spec explicitly forbids running seed1/2 automatically this round.
- Whether this reproduces on other horizons, datasets, or encoder types
  (Transformer/patch) — out of scope this round.
- A causal account of the early representation collapse (why it happens,
  why it only partially recovers) — only its timing relative to other
  metrics was measured, not its cause.
- Whether the utility-group displacement pattern (U-shape) generalizes or
  is specific to this run/seed.

## Interpretation-rule compliance

Per spec section 23, the following is the full extent of the claim made
here: **"In this ETTh1_720, MLP-encoder, seed0 diagnostic, freezing
either side of the shared query/candidate encoder at initialization
(query-only or candidate-only adaptation) measurably hurts full-memory
retrieval relative to the actual joint update, both during training and
at the held-out test evaluation, even though query and key
encoder-gradients are measurably anti-correlated for most of training.
This does not establish that shared-encoder training is broken, that
candidate drift is harmful in general, or that this pattern generalizes
beyond this single run."** No claim is made about Frozen-Key, Separate
Dual-Encoder, other seeds, other horizons, or other datasets being
better or worse without running them.

## Artifacts

- Audit: `research/J-shared-encoder-drift/AUDIT.md`
- Script: `scripts/train_j_shared_encoder_drift01.py`
- Unit tests: `tests/test_j_shared_encoder_drift01.py` (18 tests, all pass;
  covers spec section 17 items 1-14 plus geometry/churn/displacement math)
- Raw results: `results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/`
  (`config.json`, `probe_query_ids.json`, `step_metrics.csv`,
  `fourway_retrieval_metrics.csv`, `geometry_metrics.csv`,
  `topk_churn.csv`, `displacement_metrics.csv`,
  `utility_group_displacement.csv`, `gradient_conflict.csv`,
  `full_val_epoch_metrics.csv`, `final_test_metrics.json`,
  `checkpoint_fingerprints.json`, `baseline_reproduction.json`)
- Checkpoints: `checkpoints/track_j_shared_encoder_drift01/ETTh1_720/`
- Reference checkpoint used (architecture config only, scratch-init
  training): `checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_.../checkpoint.pth`
- Exact command:
  `python scripts/train_j_shared_encoder_drift01.py --reference_ckpt <S0_wce ETTh1 checkpoint> --cell ETTh1_720`
  (all other args at their audited defaults: `tau_t=tau_s=0.1`, `top_k=10`,
  `batch_size=32`, `lr=1e-3`, `train_epochs=10`, `patience=5`,
  `init_seed=loader_seed=0`, `n_probe=256`)
