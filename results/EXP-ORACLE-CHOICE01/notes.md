# EXP-ORACLE-CHOICE01 — notes

## Pre-training one-hot target diagnostic (`oracle_margin_diagnostic.json`)

Ran BEFORE any D1 training, per the spec's requirement — interpretation
only, does NOT change the loss in response to what it finds.

| step | near_tie_frac | rel_margin_mean | abs_margin_mean | top5_spread_mean |
|---|---:|---:|---:|---:|
| t=1 | 8.5% | 0.1361 | 0.05101 | 0.15714 |
| t=2 | 14.5% | 0.0634 | 0.01779 | 0.03759 |
| t=3 | 32.5% | 0.0303 | 0.00693 | 0.01455 |
| t=4 | 54.5% | 0.0151 | 0.00346 | 0.00705 |
| t=5 | 73.0% | 0.0080 | 0.00182 | 0.00393 |
| t=6 | 82.0% | 0.0056 | 0.00121 | 0.00257 |
| t=7 | 89.0% | 0.0035 | 0.00075 | 0.00162 |
| t=8 | 97.0% | 0.0021 | 0.00040 | 0.00095 |
| t=9 | 98.0% | 0.0014 | 0.00028 | 0.00059 |
| t=10 | 99.0% | 0.0008 | 0.00017 | 0.00038 |

**Interpretation (limitation, not acted on):** the Set Oracle's true
top-1-vs-top-2 utility margin collapses dramatically as the selected set
grows — by t=10, 99% of queries have a near-tied true top-1/top-2, and the
relative margin has shrunk to <0.1% of its t=1 value. This is a
mathematical consequence of the closed-form incremental-weighted-mean
utility: as more points enter the weighted average, one additional
candidate's marginal effect on the aggregate shrinks. **One-hot
Oracle-Choice CE is therefore an increasingly ill-posed target at later
steps** — the "true" label is often arbitrary among several near-equally
good candidates. Per the pre-registered spec, this is recorded here as an
explanation for why D1's teacher-forced top-1 accuracy stays low
throughout training (§ below) rather than triggering an automatic switch
to multi-positive/soft/KL targets.

## Training behavior

`best_epoch=10` (the LAST configured epoch) — `val_overlap@10` improved
EVERY single epoch (0.0119→0.0146) and never declined, unlike every other
loss/scorer/encoder arm tried this session (R1, R2, C1, C2, E1, E2), all of
which peaked at epoch 1 and declined thereafter. Training was NOT
early-stopped; it simply ran out of configured epochs (10, inherited
unmodified from the base checkpoint's own args) while still improving.
This alone does not confirm the loss is "better" in an absolute sense (a
longer training budget was never tried for R2 either, so this is not a
controlled comparison of training dynamics) — but it is a qualitatively
different trajectory from every other arm, worth flagging.

`wall_clock_seconds=2099.2` (~35 min, faster than R2's own 5169s largely
because — no dense_utility target changes were needed beyond what R2
already computes, and D1 has no memory-safe streaming overhead since the
encoder stays frozen — this experiment's per-epoch cost after the first
is ~209s, close to R2's own steady-state ~200s/epoch).

## Teacher-forced decision metrics (test split, `oracle_choice_diag_test.json`)

Mean across all K=10 steps: top1_acc=0.0036, top5_acc=0.0161,
top10_acc=0.0415, pred_rank_mean=380.8, step_regret=0.2956.

t1 specifically (the step with by far the least severe near-tie problem
per the margin diagnostic above): top1_acc=0.0134, top5_acc=0.0357,
top10_acc=0.1116, pred_rank_mean=67.7, step_regret=0.997 (highest of all
steps — the SCALE of `A_weighted` errors at t=1, an empty/near-empty
prefix, is itself larger in absolute terms than at later steps, so this
step_regret figure is not directly comparable across steps without
normalizing by scale).

t2: top1_acc=0.0045, pred_rank_mean=153.0, step_regret=0.381 (still one of
the largest of any step besides t1, then declines toward a floor around
0.17-0.20 for t6-t10).

**Oracle exact next-choice accuracy stays very low throughout training and
evaluation (top1_acc never exceeds ~1.3%)** — this is consistent with, and
plausibly explained by, the near-tie diagnostic above: correctly guessing
a near-arbitrary tie-break among dozens of near-equally-valid candidates
is a hard, high-entropy target, especially at later steps. Despite this,
D1's DOWNSTREAM realized-selection metrics (t1/t2 diagnostics, Stage-2,
HardAggregate — see REPORT.md) are the best of any arm this session,
suggesting the model does not need to reproduce the Oracle's exact
tie-broken choice to make substantially better SEQUENTIAL decisions than
R2's SmoothL1+pairwise surrogate.
