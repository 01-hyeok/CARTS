# EXP-STRONG-SCORER-DIAG01 — Report

Scope: ETTh1, H96, seed 0, top_k=10. C0 = R2 (hybrid loss, cosine scorer,
`UtilityHead`) reused verbatim from `EXP-TOPTAIL-RANK01`, not retrained. C2 =
identical R2 hybrid loss, scorer replaced with `StrongResidualPairScorer`
(cosine + zero-init nonlinear residual MLP over `[h,e,h⊙e,|h-e|]`), trained
fresh this experiment. No other change to architecture, loss, encoder, or
data. Weather / H720 / architecture sweep / encoder unfreezing were **not**
run.

---

## 1. Small-N positive control

**PASSED** — `tests/test_exp_strong_scorer_diag01.py::test_smalln_positive_control_fits_known_utility_landscape`.
`StrongResidualPairScorer`, trained with the same `pairwise_step_loss` used
on the real task, fits a synthetic known utility landscape (16 queries, 256
candidates, 400 steps) to mean oracle-best predicted rank < 20/256. This
rules out "broken scorer" or "broken training wiring" as the explanation for
any negative result below. See `smalln_summary.json`.

## 2. Implementation changes

- `models/DenseUtilityRetriever.py`: added `StrongResidualPairScorer`
  (zero-init final MLP layer ⇒ numerically identical to `UtilityHead` at
  step 0; `forward` for the full-memory bank, `forward_batched` for the
  per-row gathered pairwise-loss pool).
- `scripts/train_toptail_rank01.py`: added `--scorer_mode
  {cosine,asymmetric,strong_pair}` / `--scorer_chunk_size`; zero-init check
  at startup (abort if `max_abs_score_deviation >= 1e-6`); factored
  `mine_pairs()` out of the existing vectorized hard-negative mining logic;
  added `strong_pair_step()`, a memory-safe **streaming** per-step training
  path used only when `scorer_mode == strong_pair` and `train=True`.
- `scripts/eval_margutil01_stage2.py`, `scripts/eval_firstanchor_diag.py`,
  `scripts/eval_continuation_diag.py`: `load_trained_selector()` now reads
  `ckpt.get('scorer_mode')` and instantiates the matching head (old
  checkpoints without this key still default to `UtilityHead`, so no prior
  experiment's evaluation changed). Added `--split {train,val,test}`
  (default `test`, so all prior CLI invocations are unaffected) to enable
  the mandatory train-vs-test diagnostic below.
- **OOM fix (mid-experiment, reported and approved before the full run):**
  the original `strong_pair` training path held the full autograd graph
  across all K teacher-forced steps × channels × candidate chunks before a
  single `.backward()`. Rewritten to a streaming design: full-memory hard
  negatives are mined in a chunked `torch.no_grad()` pass (global Top-K,
  never a shortlist); the pairwise loss is computed with grad on only the
  small gathered positive/hard-negative pool and backpropped immediately;
  the dense SmoothL1 term is computed and backpropped per candidate chunk,
  freeing each chunk's graph before the next; `optimizer.step()` is called
  exactly once per batch by the caller, never inside `strong_pair_step()`.
  `SetConditioner`'s `h_t` is recomputed fresh per chunk (via an `m_fn`
  zero-arg callable, not a precomputed tensor) so each chunk's graph,
  including the graph through the trainable `EmptySetToken` at t=0, is
  independent and freeable — this was necessary to avoid a genuine "backward
  through the graph a second time" bug found on real data (not caught by the
  first unit test, which used a plain leaf tensor for `m`; a dedicated
  regression test `test_streaming_with_trainable_m_source_does_not_double_backward`
  was added using a real `EmptySetToken`). Mathematical equivalence to the
  original unchunked implementation was verified by
  `tests/test_exp_strong_scorer_streaming.py::test_streaming_gradients_match_unchunked_reference`
  (gradient equality on every `SetConditioner`/scorer parameter, `atol=1e-4`)
  **before** the fixed implementation was used for the real run. Loss
  normalization (chunk-sum / global valid count) and lambda weighting are
  unchanged from the original R2 hybrid math.
  `FULL MEMORY -> DIRECT TOP-K` semantics are unaffected: chunking is a pure
  memory optimization, every valid candidate is still scored and included in
  the global hard-negative/argmax computation.

## 3. Controlled-variable verification

- Zero-init check at C2 training startup: `max_abs_score_deviation =
  0.000e+00` (exact match, C2 ≡ C0 at step 0).
- Loss mode: `hybrid` (pairwise + 1.0·SmoothL1) — identical to R2/C0.
- Checkpoint selection criterion: unchanged, `val_overlap@10` (the only
  criterion ever used to pick `best_epoch`; top-tail rank diagnostics below
  are computed post-hoc for interpretation only, never used for checkpoint
  selection).
- Dataset/horizon/seed/top_k/encoder/`SetConditioner`/`EmptySetToken`: all
  identical to C0, reused unmodified (see `working_tree.diff` — the diff
  touches only the two eval-side loader functions, the training script's
  scorer-mode dispatch, and `DenseUtilityRetriever.py`; no `SetConditioner`
  or encoder code is touched).

## 4. Trainable parameter audit (C2)

- `trainable_params = 378243` total: `SetConditioner = 49664`,
  `EmptySetToken = 128`, `StrongResidualPairScorer = 328451`.
- The scorer head accounts for ~87% of C2's trainable parameters versus C0's
  `UtilityHead` (2 scalars) — a large, deliberate capacity increase, per the
  experiment's purpose (ceiling probe on scorer capacity).

## 5. Training / fit degree

- `best_epoch = 1`, `best_val_overlap_at_k = 0.0098`, early-stopped at epoch
  6.
- Epoch-by-epoch `val_overlap@10`: `0.0098 → 0.0047 → 0.0073 → 0.0059 →
  0.0057 → 0.0041` — **monotonic decline after epoch 1**. This matches every
  other loss/scorer arm run in this whole session (R1, R2, C1 all also
  peaked at `best_epoch=1` and declined) — checkpoint selection correctly
  keeps the epoch-1 weights, which are very close to the C0 initialization
  the scorer started from.
- `wall_clock_seconds = 8936.2`, `peak_gpu_memory_mib = 504` — confirms the
  OOM fix: the run that previously could not even complete a sanity epoch
  now trains to convergence at 504 MiB peak.

## 6. Validation / test generalization gap — the key finding

`best_epoch=1` means C2's weights are barely displaced from the C0 (cosine)
initialization by the time the best checkpoint is selected, since the loss
starts degrading val overlap immediately after. Given that, the following is
the most notable result of this experiment and does not fit any of the three
pre-registered outcomes cleanly:

**C2 is measurably *worse* on the train split than on the test split**, on
every t1/t2 top-tail diagnostic — the opposite of the usual overfitting
direction:

| metric | test | train |
|---|---|---|
| t1 Spearman (true top-1%) | 0.2251 | 0.1250 |
| t1 oracle-best predicted rank (median) | 97 | 654 |
| t1 Top-50 containment | 0.320 | 0.120 |
| t2 Spearman (true top-1%) | 0.1818 | 0.0576 |
| t2 oracle predicted rank (median) | 442.0 | 2266.5 |
| t2 selected true rank (median) | 3859.5 | 5914.5 |
| t2 hurt_frac | 0.214 | 0.324 |
| t2 continuation_regret | 0.3657 | 0.2937 |

(Full data: `train_vs_val_test.json`.) The most consistent reading is that
because the checkpoint is essentially still near the C0/cosine solution
(early-stopped at epoch 1), it has not had the chance to specialize to the
train distribution at all — so what's being measured here is closer to two
noisy evaluations of the same near-C0 model on different query subsets, not
a train-fits/test-fails generalization failure in the usual sense. This is
flagged as an open finding rather than forced into "Outcome B" — see §11.

## 7. t=1 diagnostics (first-anchor)

Test split, C0 vs C2:

| metric | C0 | C2 | Δ (C2−C0) |
|---|---|---|---|
| Spearman (true top-1%) | 0.2128 | 0.2251 | +0.0123 |
| oracle-best predicted rank (median) | 99 | 97 | −2 |
| Top-50 containment | 0.305 | 0.320 | +0.015 |

All three t1 metrics move in the favorable direction for C2 on test, by
small margins. C2's train-split t1 numbers (Spearman 0.125, rank 654,
containment 0.12) are substantially worse than either C0-test or C2-test
(no C0-train comparison exists — C0 is reused, not retrained, so no
train-split evaluation was run for it; this is a C2-only internal
comparison).

## 8. t=2 continuation diagnostics

Test split, C0 vs C2:

| metric | C0 | C2 | Δ (C2−C0) |
|---|---|---|---|
| hurt_frac | 0.232 | 0.214 | −0.018 (better) |
| selected true rank (median) | 1934.0 | 3859.5 | +1925.5 (worse) |
| oracle predicted rank (median) | 560.0 | 442.0 | −118 (better) |
| Spearman (true top-1%) | 0.1404 | 0.1818 | +0.0414 (better) |
| continuation_regret | 0.3498 | 0.3657 | +0.0159 (worse) |

Mixed: C2 improves oracle-side ranking metrics (oracle predicted rank,
Spearman, hurt_frac) but is worse on the metric closest to actual selector
behavior — `selected true rank` and `continuation_regret` both degrade. This
mirrors the Stage-2/HardAggregate mixed picture below: better local ranking
signal, worse realized selection outcome.

## 9. HardAggregateMSE@10 (unweighted, free-running)

C0 = 0.5514, C2 = 0.6224 (Δ = +0.0710, worse). C2 does not close any of the
gap to B0 on this free-running aggregate metric; it is further from B0 than
C0 was.

## 10. Stage-2 evaluation

| metric | C0 | C2 | B0 (reference) |
|---|---|---|---|
| Stage-2 MSE | 0.39526 | 0.39708 | 0.37312 |
| Δ vs B0 | +0.02214 | +0.02396 | — |
| gap_recovery | −0.2626 | −0.3392 | — |

C2 is slightly worse than C0 on Stage-2 MSE and on `gap_recovery` (more
negative = further from the set-oracle-recoverable gap). For reference, C1
(EXP-ASYM-SCORER01, asymmetric scorer) was worse still: Stage-2 = 0.41257,
gap_recovery = −0.488, HardAgg = 0.690. So the capacity ordering on
Stage-2/HardAggregate is **C0 (cosine) > C2 (strong pair) > C1
(asymmetric)** — increasing scorer capacity beyond plain cosine has not
helped the realized forecasting outcome in either of the two capacity
increases tried this session, even though C2 (the larger capacity increase)
shows small, real improvements on the t1/t2 *ranking-only* diagnostics that
C1 did not show.

## 11. Which pre-registered outcome (A/B/C) is closest?

None fits cleanly, and per this session's established norm the result is
reported as such rather than forced into a bucket:

- **Not Outcome A** (capacity bottleneck confirmed — stronger scorer clearly
  closes the gap): C2 does not beat C0 on Stage-2 MSE, gap_recovery, or
  HardAggregateMSE — the metrics closest to the actual research question
  (does this fix the forecasting failure) are flat-to-worse.
- **Not a clean Outcome B** (train fits well but generalization fails) in
  the classic sense: C2's train-split t1/t2 numbers are *worse* than its
  test-split numbers, not better — there's no evidence of it fitting the
  train distribution at all. This is consistent with `best_epoch=1`
  (near-C0 initialization, immediately early-stopped) rather than with a
  scorer that learned to overfit training data and failed to generalize.
- **Not Outcome C** (even a strong scorer can't fit/use supervision at
  all): the small-N positive control (§1) proves the scorer/loss/training
  loop *can* fit a known utility landscape, and C2 does show small, real,
  consistent improvements on t1 (test) and several t2 (test) ranking
  metrics relative to C0 — some supervision signal is being used, just not
  enough to survive the free-running selection process or move Stage-2 MSE.

**Closest characterization**: added scorer capacity produces small,
genuine local top-tail ranking improvements (t1/t2 test-split Spearman and
several rank metrics) but these do not translate into a better realized
selector — Stage-2 MSE, gap_recovery, and HardAggregateMSE are flat-to-worse,
and `continuation_regret`/`selected true rank` (the metrics closest to
actual free-running behavior) are worse. Combined with `best_epoch=1` and
the immediate post-epoch-1 val-overlap decline seen in every scorer/loss arm
tried this session (R1, R2, C1, C2), the more parsimonious explanation is
that the bottleneck is not scorer expressiveness at the per-candidate
scoring level, but something in how the *sequential* teacher-forced
training signal (dense marginal utility target, weighted-set greedy oracle
supervision, or the train/inference distribution mismatch inherent to
teacher forcing) fails to produce a checkpoint that keeps improving past
epoch 1, regardless of how much scorer capacity is given to it.

## 12. Continue or stop / reconsider?

Per the user's explicit instruction, **no next experiment has been run.**
This report and `research/REVIEW_FOR_CHATGPT.md` are ready for independent
review; the choice of next direction (e.g., investigating the
`best_epoch=1`-then-decline pattern itself, rather than further scorer
capacity increases) is left to that review and to the user.

---

### Primary result table (all arms, test split, ETTh1 H96)

| arm | Stage-2 MSE | Δ vs B0 | gap_recovery | HardAgg MSE@10 | t1 Spearman top1% | t1 oracle rank (median) | t2 Spearman top1% | t2 continuation_regret |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.37312 | — | — | — | — | — | — | — |
| C0 (R2, cosine) | 0.39526 | +0.02214 | −0.2626 | 0.5514 | 0.2128 | 99 | 0.1404 | 0.3498 |
| C1 (asymmetric) | 0.41257 | +0.03945 | −0.4880 | 0.6900 | 0.1590 | 180 | — | — |
| C2 (strong pair) | 0.39708 | +0.02396 | −0.3392 | 0.6224 | 0.2251 | 97 | 0.1818 | 0.3657 |

Source files: `stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`train_vs_val_test.json`, `metrics.csv`, `train_summary.json`,
`checkpoint_fingerprints.txt`, `working_tree.diff`, `smalln_summary.json`,
`command.txt`, `config.json`.
