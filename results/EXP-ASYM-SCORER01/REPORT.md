# EXP-ASYM-SCORER01 — Report (2026-09-07)

Scope: **ETTh1 H96 only** (Weather and H720 explicitly not run, per spec).
No new training on C0 -- reuses EXP-TOPTAIL-RANK01/R2's own ETTh1 H96
checkpoint and results verbatim. C1 is the only new training run in this
experiment.

## Question

> R2 hybrid loss까지 적용했음에도 남아 있는 top-tail / continuation ranking
> failure와 Stage-2 gap이, 현재 fixed cosine geometry의 표현력 제약
> 때문인가?

`C0 = R2 + cosine` vs. `C1 = R2 + asymmetric (cos(W_q h_t, W_k e_i))`,
everything else held identical.

## 1. Implementation changes

- `models/DenseUtilityRetriever.py`: added `AsymmetricUtilityHead`, a thin
  wrapper around `layers.retrieval_metric.RetrievalMetric(kind='asymmetric',
  layer_norm=False, output='cosine')` (the project's existing, already-
  reviewed asymmetric scorer -- not reimplemented) plus the same affine
  `scale`/`bias` convention `UtilityHead` already uses.
- `scripts/train_toptail_rank01.py`: added `--scorer_mode {cosine,
  asymmetric}` (default `cosine`, i.e. EXP-TOPTAIL-RANK01's own behaviour is
  completely unchanged when this flag is omitted). When `asymmetric`: runs
  an identity-init equivalence check before training starts (aborts if
  `max_abs_score_deviation >= 1e-6`), logs `W_q`/`W_k` Frobenius-norm
  deviation from identity and condition number every epoch, and saves
  `scorer_mode` in the checkpoint.
- `scripts/eval_margutil01_stage2.py`, `scripts/eval_firstanchor_diag.py`
  (and therefore `scripts/eval_continuation_diag.py`, which imports from
  it): `load_trained_selector` now checks the checkpoint's own saved
  `scorer_mode` and instantiates `AsymmetricUtilityHead` or `UtilityHead`
  accordingly -- old checkpoints (no `scorer_mode` key) default to
  `UtilityHead`, so every prior experiment's evaluation is unaffected.
- New tests: `tests/test_exp_asym_scorer01.py` (6 tests).

## 2. Controlled-variable verification

Every item in the spec's section 6 list is either (a) literally the same
Python object/checkpoint reused across C0/C1 (frozen encoder, teacher
cache, Stage-2 checkpoint/architecture, evaluation scripts), or (b) passed
via CLI flags that were identical between the R2 training run that produced
C0 and this experiment's C1 run (`--loss_mode hybrid --lambda_rank 1.0
--top_k 10 --seed 0 --chunk_size 4096`, same `--base_ckpt`/`--teacher_cache`).
`trainable_params` logged at C1 startup: `82562` = C0's `49794`
(`SetConditioner=49664, EmptySetToken=128, UtilityHead=2`) + `32770` new
(`W_q`: 128x128=16384, `W_k`: 128x128=16384, `scale`+`bias`: 2) -- the
added parameters are exactly and only in the scorer head, confirmed by the
startup log line and by `tests/test_exp_asym_scorer01.py::test_asymmetric_adds_params_only_to_scorer_not_elsewhere`.

## 3. Identity-init equivalence check (section 4)

`scorer_init_check.json`: `max_abs_score_deviation = 0.0` (< 1e-6
threshold), computed via `layers.retrieval_metric.cosine_init_deviation`
on 256 random query/key pairs before training started. `AsymmetricUtilityHead`
at construction is bitwise-identical to `UtilityHead` (also verified by
unit test `test_identity_init_full_head_matches_cosine_utility_head_forward`).
Projection outputs are confirmed L2-normalised before the dot product
(`test_projection_output_is_l2_normalised_before_score`), so `W_q`/`W_k`
norm growth cannot inflate scores directly -- only their *direction*
relative to each other can move the ranking.

## 4. Training stability

| | C0 (R2+cosine) | C1 (R2+asymmetric) |
|---|---:|---:|
| best_epoch (by val_overlap@10) | 1 | **3** |
| best val_overlap@10 | 0.0082 | **0.0085** |
| epochs to early stop | 6 | 8 |
| `\|\|W_q-I\|\|_F` at best epoch (3) | n/a | 3.919 |
| `\|\|W_k-I\|\|_F` at best epoch (3) | n/a | 3.364 |
| `cond(W_q)` at best epoch | n/a | 12.09 |
| `cond(W_k)` at best epoch | n/a | 58.81 (growing every epoch, reached 139.6 by epoch 8) |

C1's checkpoint-selection proxy metric (`val_overlap@10`) is slightly
BETTER than C0's and peaks later (epoch 3 vs. epoch 1) -- superficially
suggesting the extra capacity helps optimisation. `W_k`'s condition number
growing monotonically and substantially faster than `W_q`'s (58.8 by epoch
3, 139.6 by epoch 8) suggests the candidate-side projection is moving
toward a more anisotropic (rank-reducing) transformation over training --
consistent with, though not proof of, an early sign of the kind of
representation narrowing this project's other diagnostics have flagged
elsewhere (EXP-SEQDIAG01's encoder-collapse finding), here in the scorer's
own projection rather than the encoder.

## 5. t=1 ranking comparison (FIRSTANCHOR-DIAG, reused unmodified)

| Metric | C0 | C1 | Delta |
|---|---:|---:|---:|
| Spearman within true top-1% | 0.213 | 0.159 | **-0.054** |
| Oracle-best predicted rank median | 99 | 180 | **+81 (worse)** |
| Top-5 containment | 4.5% | 4.5% | 0 |
| Top-10 containment | 8.0% | 8.5% | +0.5 |
| Top-50 containment | 30.5% | 26.5% | **-4.0** |
| Top-1 hit rate | 0.5% | 1.0% | +0.5 |

Mixed at the very tightest cutoffs (Top-1/Top-10 marginally better for C1)
but WORSE on the metrics that matter most for a 10-candidate greedy Top-K
(Top-50 containment, oracle rank median, top-1% Spearman).

## 6. t=2 exhaustive continuation comparison (CONTINUATION-DIAG, reused unmodified, `dense_first` anchor, same 500-query subset)

| Metric | C0 | C1 | Delta |
|---|---:|---:|---:|
| hurt_frac | 23.2% | 24.0% | +0.8 (worse) |
| hurts-and-oracle-improves frac | 23.2% | 24.0% | +0.8 (worse) |
| selected second true rank median | 1934 | 2799 | **+865 (worse)** |
| oracle second predicted rank median | 560 | 673 | **+113 (worse)** |
| Spearman (global) | 0.177 | 0.248 | **+0.071 (better)** |
| Spearman within true top-10% | 0.288 | 0.256 | **-0.032** |
| Spearman within true top-5% | 0.254 | 0.226 | -0.028 |
| Spearman within true top-1% | 0.140 | 0.137 | -0.003 (~flat) |
| continuation regret | 0.350 | 0.375 | **+0.025 (worse)** |

Full numbers in `metrics.csv`. Every metric is flat or worse for C1 except
GLOBAL Spearman (0.177→0.248, improved) -- the same global-vs-top-tail
dissociation this project's diagnostics have repeatedly found (EXP-MARGUTIL01
question 16, EXP-CONTINUATION-DIAG's central finding, EXP-TOPTAIL-RANK01's
R0 vs. R1/R2 pattern) replays here at the scorer level: more expressive
geometry fits the bulk of the ranking slightly better, but the decision-
critical extreme top tail gets WORSE, not better.

## 7. Free-running K=10 HardAggregate comparison

| | C0 | C1 |
|---|---:|---:|
| `HardAggregateMSE@10` (selector) | 0.551 | **0.690 (worse)** |
| `duplicate_rate` | 0.0 | 0.0 |
| `invalid_rate` | 0.0 | 0.0 |
| `gap_recovery` | -0.263 | **-0.488 (worse)** |

Structural invariants (no duplicates, no invalid picks, full-memory support
at every step) hold identically for both arms -- the asymmetric scorer did
not introduce any wiring problem, it is simply a worse-performing scorer
for this task.

## 8. Stage-2 comparison

| | B0 | C0 | C1 |
|---|---:|---:|---:|
| Stage-2 MSE | 0.37312 | 0.39526 | **0.41257** |
| Delta vs B0 | — | +0.02214 | **+0.03945** |
| Delta vs C0 | — | — | **+0.01731 (worse)** |

C1 is worse than C0 by +0.017 on Stage-2 Final MSE -- a regression larger
than this project's own 0.01 noise-scale reference, so this is a real,
not noise-level, degradation.

## 9. Does asymmetric support the scorer-bottleneck hypothesis?

**No.** Per section 18's decision rule, this is unambiguously **"Evidence
against scorer bottleneck"**: `t2 rank`, `continuation regret`,
`HardAggregate`, and `Stage-2` are all worse for C1, not better, and by
margins well outside plausible noise (Stage-2 +0.017 vs. C0; t=2 selected-
candidate true rank median nearly 50% worse; oracle-predicted rank median
20% worse). The training-time proxy (`val_overlap@10`) improving while
every downstream/decision-relevant metric worsens is itself informative:
it demonstrates that `val_overlap@10` is not a reliable predictor of actual
Stage-2 quality for this class of model, and checkpoint selection by that
criterion can pick a strictly worse model when the search space is larger
(more free parameters in the scorer to overfit that specific proxy).

This is **not** evidence that asymmetric geometry could never help --
identity initialisation guarantees C1 starts exactly at C0's own
performance, so C1's decline reflects what THIS training run's optimiser
did with the extra capacity (with the SAME hyperparameters/epochs/patience
tuned for the cosine arm, not retuned for asymmetric), not a proof that no
asymmetric scorer could ever help under different training conditions.

## 10. What this does NOT establish

- Whether a different learning rate, weight decay, or explicit
  regularisation on `W_q`/`W_k` (e.g. an orthogonality penalty to control
  the growing condition number) would change this outcome -- not tested,
  per the pre-registered scope (no lambda/hyperparameter sweep).
- Whether Weather H96 or H720 would show the same pattern -- not run.
- Whether Mahalanobis (the "middle rung" of the expressiveness ladder in
  `layers/retrieval_metric.py`'s own docstring) behaves differently from
  either cosine or full asymmetric -- not tested, out of scope.
- Whether the growing `cond(W_k)` is causally responsible for the
  degradation, or merely correlated with it -- this experiment observed
  but did not intervene on that quantity.

## 11. Conclusion

Fixed cosine geometry is **not** the bottleneck this experiment set out to
test for, on ETTh1 H96, under R2's hybrid loss, with this training budget
and these hyperparameters. Every decision-relevant metric (Stage-2,
HardAggregate, t=2 continuation, and most of t=1) is worse with the
asymmetric scorer than with plain cosine. The remaining gap after R2 (t=2
top-tail ranking still weak, Stage-2 still worse than B0) is not resolved
by adding scorer expressiveness under this design; per the project's own
`SetConditioner representation or frozen encoder representation` fallback
named in the spec's section 18, those become the next candidate
explanations -- but this experiment does not test either, and does not
recommend running either without further review.
