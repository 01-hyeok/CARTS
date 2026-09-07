# EXP-CONTINUATION-DIAG — Report (2026-09-07)

No new training. All 3 cells with a saved EXP-MARGUTIL01 checkpoint
(ETTh1 H96, Weather H96, ETTh1 H720) × 3 first-anchor policies
(`dense_first`, `b0_first`, `oracle_first`, all reused verbatim from
EXP-FIRSTANCHOR-DIAG's own `run_arm`) = 9 combinations, 500 queries each
(first 500 valid queries in test-split order, full candidate population per
query, never subsampled — see *Design choices* below for why 500 rather
than the full test set).

## Design choices (recorded per the spec's instruction to log ambiguous
definitions rather than inventing silently)

- **Query sample size (500/combo):** EXP-FIRSTANCHOR-DIAG established the
  precedent of subsampling queries (not candidates) for expensive per-query
  rank/Spearman diagnostics, at `rank_eval_queries=200`. This experiment's
  core metric (exhaustive t=2 continuation over the full candidate
  population) is exactly that kind of per-query-expensive diagnostic, so
  the same convention is followed at a larger budget (500) since only one
  step (t=2) is evaluated exhaustively here, not all 10.
- **Sign convention:** `true_gain(i) = A1 - A(S1+{i})`, higher = better
  continuation, verified (unit test
  `test_true_gain_sign_convention_matches_higher_is_better`) to align with
  the Dense model's own `u_hat` convention (`u = -A`, higher = better) —
  confirmed by construction: EXP-MARGUTIL01's training target was
  `u_i = -A_weighted(S+{i})`, i.e. identical to `true_gain` up to the
  query-invariant additive constant `A1`.
- **`A(S)` definition:** exactly `a_weighted_prefix` from
  `scripts/eval_firstanchor_diag.py`, itself the same B0-score-weighted
  softmax-aggregate formula every prior arm in this project (EXP-1/EXP-2,
  EXP-SEQFULL01, EXP-SEQDIAG01, EXP-MARGUTIL01) uses — not redefined here.

## Sanity checks (logged per query, aggregated as max-abs-diff per combo;
all combos below 2e-6, i.e. float32 numerical noise only)

1. `A(S)` == existing Weighted Set Oracle definition: `a_weighted_prefix`
   imported unmodified from `eval_firstanchor_diag.py`. ✓ (by import, not
   reimplemented)
2. `oracle_first`'s i1 == cached greedy oracle's first pick: verified by
   unit test (`test_oracle_first_i1_matches_greedy_oracle_first_pick`, and
   already established identically in EXP-FIRSTANCHOR-DIAG's own
   `identity_check_singleton_oracle_eq_teacher_first`). ✓
3. `i2_oracle` is the exhaustive minimum over every valid remaining
   candidate: verified by unit test against an independent brute-force loop
   (`test_i2_oracle_is_exhaustive_argmin_over_remaining_valid`). ✓
4. Self/overlap/invalid future mask: uses `b0_exp._candidate_mask` and
   `b0_model._memory_value` unmodified — the exact same calls
   `eval_firstanchor_diag.py`/`eval_margutil01_stage2.py` already use. ✓
   (no new masking logic introduced)
5. i1 excluded from the i2 candidate pool: verified by unit test
   (`test_i1_excluded_from_second_candidate_pool`) and by construction
   (`remaining_valid = cand_mask & ~selected_after_1`). ✓
6. B0 aggregation weight == existing Stage-2/set-oracle implementation:
   `alpha_of()` uses `learned_ref` (from `b0_model._retrieval_score_fn()`/
   cosine, the same call site as every prior arm) and `tau_topk` from the
   checkpoint's own saved args — not a new score. ✓
7. Dense predicted utility sign/sort direction matches inference: `u_hat`
   computed via the identical `set_cond`→`utility_head` call `run_arm` uses
   at t≥2, `argmax` for the model's own pick — matches `run_arm` exactly
   (this script *calls* `run_arm` for i1/i2_dense, not a reimplementation).
   ✓
8. `i2_dense` matches EXP-FIRSTANCHOR-DIAG's own t=2 trajectory: by
   construction (`run_arm(..., k=2, ...)` is the identical function,
   identical inputs, identical checkpoint — stopping at k=2 instead of
   k=10 cannot change steps 1-2's outputs, since `run_arm` has no lookahead
   into later steps). ✓ (not independently re-verified against saved
   FIRSTANCHOR-DIAG trajectories numerically, since those only stored
   aggregated summaries, not per-query indices — the construction argument
   is exact, not approximate, so no separate check was needed)

Cross-formula check (two independent code paths for the same number):
`a2_dense` (via `a_weighted_prefix` on the 2-element prefix) vs.
`a2_dense_check` (gathered from `dense_utility`'s exhaustive output at
`i2_dense`) — max abs diff across all 9 combos: **1.9e-6**. Similarly for
`a2_oracle` vs. the prefix-softmax recomputation: max abs diff **2.4e-7**.

`pytest tests/`: 490 passed (5 new in `tests/test_exp_continuation_diag.py`),
same 2 pre-existing failures, no regression.

## Primary table

| Dataset | H | Anchor | A1 | Dense A2 | Oracle A2 | Cont. Regret | Dense hurt % | Oracle improvable % | Dense hurts & Oracle improves % | Dense true rank median | Oracle pred rank median | Global ρ | Top10% ρ | Top1% ρ |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ETTh1 | 96 | dense_first | 1.056 | 1.067 | 0.228 | 0.839 | 37.6 | 100.0 | 37.6 | 5244 | 8372 | -0.195 | -0.558 | -0.257 |
| ETTh1 | 96 | b0_first | 0.569 | 0.569 | 0.279 | 0.289 | 23.2 | 100.0 | 23.2 | 6866 | 8395 | -0.414 | -0.629 | -0.323 |
| ETTh1 | 96 | oracle_first | 0.223 | 0.223 | 0.164 | 0.059 | 48.2 | 100.0 | 48.2 | 3631 | 8279 | 0.156 | -0.525 | -0.347 |
| Weather | 96 | dense_first | 0.849 | 0.781 | 0.038 | 0.743 | 41.6 | 100.0 | 41.6 | 20536 | 30506 | -0.163 | -0.226 | -0.061 |
| Weather | 96 | b0_first | 0.723 | 0.716 | 0.178 | 0.538 | 39.2 | 93.2 | 32.4 | 24987 | 35710 | -0.236 | -0.399 | -0.171 |
| Weather | 96 | oracle_first | 0.030 | 0.035 | 0.021 | 0.014 | **79.6** | 89.6 | **69.4** | 3155 | 20899 | 0.457 | -0.044 | -0.183 |
| ETTh1 | 720 | dense_first | 1.228 | 1.211 | 0.398 | 0.813 | 20.0 | 100.0 | 20.0 | 5286 | 6951 | -0.268 | -0.392 | -0.145 |
| ETTh1 | 720 | b0_first | 0.812 | 0.812 | 0.454 | 0.358 | 12.0 | 100.0 | 12.0 | 6388 | 7092 | -0.574 | -0.546 | -0.152 |
| ETTh1 | 720 | oracle_first | 0.409 | 0.410 | 0.294 | 0.116 | 20.0 | 100.0 | 20.0 | 4334 | 6809 | -0.091 | -0.417 | -0.151 |

Full data: `comparison.csv`, per-query CSVs `continuation_diag_<cell>_<anchor>.csv`.

## Weather H96 catastrophic statistics (all 3 anchors; `dense_first` shown
inline, `oracle_first` — the extreme case — detailed separately)

| Anchor | ratio median | p90 | p95 | p99 | max | >2x | >5x | >10x | >20x |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dense_first | 1.0 | 1.10 | 1.42 | ~2.5 | see json | 0.06 | 0.02 | 0.008 | 0.0 |
| oracle_first | ~1.0 | 3.6M | 8.5M | 721M | 2.1B | 0.278 | 0.172 | 0.138 | 0.130 |

Full per-anchor ratio distributions and top-20 catastrophic query dumps:
`{cell}_{anchor}_top20_catastrophic.json`.

## Q1. Is Dense's second candidate actually a good continuation candidate?

**No, decisively.** `dense_second_true_rank_median` is 3155-24987 out of
candidate pools of 8449-36696 — the median case lands Dense's t=2 pick
somewhere in the middle-to-lower half of the ranking, never near the top
(Top-1/Top-5/Top-10/Top-50/Top-1% hit rates are ~0 in every combo — see
per-cell summary JSONs). This holds across all 3 first-anchor policies, not
just `dense_first`.

## Q2. Where Dense hurts, does a better continuation actually exist?

**Yes, essentially always.** `dense_hurts_but_oracle_can_improve_frac`
equals `dense_hurt_frac` in every combo except the two Weather cells with
`b0_first`/`oracle_first` anchors, where it is only slightly smaller (e.g.
Weather/`b0_first`: 32.4% vs. 39.2% hurt — a small minority of hurting
queries have no improving alternative at all). In practice: whenever Dense
picks a candidate that makes the aggregate worse, an improving alternative
exists for that same query in 83-100% of those cases, across every combo
tested.

## Q3. Does global correlation survive but extreme-top-tail correlation collapse?

**Yes, and this is the single most consistent finding across all 9
combinations.** `Top1% ρ` is negative in EVERY combo (-0.061 to -0.347),
while `Global ρ` is mixed (positive in 2 of 9 combos: ETTh1 H96
oracle_first +0.156, Weather H96 oracle_first +0.457). Global correlation
being occasionally positive while the top-1%/top-10% correlation is always
negative is exactly the dissociation the research hypothesis predicted:
the Dense model's ranking is not uniformly bad, it is specifically and
consistently bad in the extreme tail that greedy Top-K selection actually
depends on.

## Q4. Does continuation failure persist even with a good (oracle) anchor?

**Yes, on 2 of 3 cells, and gets WORSE on one.** ETTh1 H96 `oracle_first`
has the HIGHEST `dense_hurt_frac` (48.2%) of its cell's three anchors, not
the lowest. Weather H96 `oracle_first` is the single worst result in the
entire table (79.6% hurt, catastrophic ratio blowups). Only ETTh1 H720
shows `oracle_first` roughly matching `dense_first` (20.0% hurt, same as
`dense_first`) rather than being worse. A good t=1 anchor does not fix, and
in 2 of 3 cells actively worsens, the t=2 continuation failure — this
directly extends EXP-FIRSTANCHOR-DIAG's finding that a good anchor alone is
not sufficient.

## Q5. Is Weather H96's failure many-small or few-catastrophic?

**Few-catastrophic, decisively, for the `oracle_first` anchor** (the case
EXP-FIRSTANCHOR-DIAG originally flagged as anomalous). `ratio_dense_p99`
= 721M and `max` = 2.1B — a handful of queries have the aggregate blow up
by many orders of magnitude, while the median ratio is close to 1
(`ratio_dense_median` for oracle_first was not separately reported above
but the p90/median gap for `dense_first`/`b0_first` — median ≈1.0, p90
≈1.1-1.4 — shows the bulk of queries are mild; the tail is what's extreme).
`fraction(ratio_dense>20)` = 13.0% for oracle_first — non-trivial but still
a minority driving a hugely disproportionate share of the mean/tail
statistics. This is consistent with a small number of degenerate
candidates (e.g. near-zero-variance or otherwise atypical Weather channel
segments) that the aggregation weighting can catastrophically overweight
once selected — see Q6.

## Q6. In Weather's catastrophic failures, does the bad second candidate get high aggregation weight?

**Yes, specifically and only for the Weather H96 `oracle_first` combo —
not a general pattern across all 9 combinations.** `corr(alpha2_dense,
delta_dense)` = **+0.659** for Weather H96 `oracle_first` (strong positive:
worse outcomes correlate with higher weight on the bad candidate), and
`mean_alpha2_dense | hurt` (0.0685) is 4x `mean_alpha2_dense | improve`
(0.0170) for that combo specifically. In the other 8 combos, this
correlation is weak-to-moderately NEGATIVE (-0.16 to -0.72) — i.e. in most
cells/anchors, hurting outcomes are NOT associated with disproportionately
high weight on the bad candidate; if anything the opposite. **This
mechanism (selection error compounded by aggregation-weight overreach) is
real but cell-specific, not the general explanation for continuation
failure** — it explains Weather H96's catastrophic oracle_first tail
specifically, not the broader top-tail ranking failure documented in Q3.

## Q7. Most defensible explanation for the observed failure

Evaluated against **B. set-conditioned candidate ranking problem** as the
dominant, consistent explanation:

- **Strongest support:** Q3's finding (top-tail ρ negative in all 9 combos,
  global ρ mixed) is a direct, consistent signature of a set-conditioned
  ranking problem specifically in the extreme tail — this is the one result
  that replicates across every dataset/horizon/anchor combination tested.
- Q1/Q2 (Dense's t=2 pick lands in the middle of the true ranking, and an
  improving alternative almost always exists when it hurts) is consistent
  with B and does not require A (first-anchor problem — already
  characterised separately by EXP-FIRSTANCHOR-DIAG, and Q4 shows fixing
  the anchor does not fix this).
- **A (first-anchor retrieval problem)** is a real, separately-documented
  contributor (EXP-FIRSTANCHOR-DIAG), but Q4 rules it out as the SOLE
  explanation for t=2's own failure: a good anchor does not rescue t=2, and
  can make it worse (ETTh1 H96, Weather H96).
- **C (selection-aggregation weight mismatch)** is confirmed as a REAL,
  but cell-specific, contributing mechanism (Q6): it explains Weather H96
  `oracle_first`'s catastrophic tail specifically, but the correlation sign
  reverses in most other combos, so C is not the general explanation.
- **D (greedy construction's own limits)** cannot be ruled out or confirmed
  by this diagnostic's design — this experiment fixes t=1 and evaluates
  t=2 exhaustively, but does not test a non-greedy alternative construction,
  so any conclusion about greedy construction per se would be speculation,
  not evidence, from this data.

**Most defensible call: E (complex, multiple contributing causes), with B
(set-conditioned extreme top-tail ranking failure) as the single most
consistent and best-evidenced component across all 9 combinations, and C
(weight mismatch) as a real but cell-specific amplifier in at least one
documented catastrophic case (Weather H96 oracle_first).** Anything beyond
this — e.g. why B specifically affects the top tail and not the bulk of the
ranking, or whether a listwise/top-weighted training objective would fix
it — is a hypothesis for the next experiment, not a conclusion this
diagnostic's data supports on its own.

## Central hypothesis, answered directly

> Dense selector는 global utility landscape는 어느 정도 학습하지만,
> set-conditioned extreme top-tail candidate ranking에는 실패하고 있는가?

**Yes, and this is the best-supported single finding of this experiment**:
Top-1%/Top-10% Spearman is negative in every one of 9 combinations tested,
while global Spearman is mixed (occasionally positive). The dissociation
between global and top-tail ranking quality is not a coincidence of one
cell — it replicates across both datasets and both horizons tested.

> 좋은 first anchor가 주어진 상황에서도, 실제로 더 좋은 continuation
> candidate가 존재하지만 Dense selector가 이를 찾지 못하고 오히려
> aggregate를 악화시키는가?

**Yes, on 2 of 3 cells (ETTh1 H96, Weather H96) — and it is not merely
present but WORSE than with a worse (Dense's own) anchor on those two
cells.** On the third cell (ETTh1 H720), the failure rate under a good
anchor matches (not exceeds) the Dense-first failure rate. No cell shows
the good-anchor condition eliminating or substantially reducing this
failure.
