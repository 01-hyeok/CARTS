# REVIEW_FOR_CHATGPT.md

Handoff for independent review. Covers the 2026-09 diagnostic campaign
(`EXP-C01`), 2026-08-29 → 2026-09-02. Self-contained: no log files needed.

Provenance is marked throughout: **[repo]** = read or recomputed from artifacts
in the repository by the implementation engineer; **[user]** = supplied by the
researcher and not independently reproduced.

---

# Research Question

Why does improving Stage-1 retrieval (Recall@10) not improve Stage-2 forecasting?

Setup: Stage-1 retrieves Top-K historical candidates by comparing query past to
candidate past; candidates are past–future pairs, and Stage-2 consumes the
retrieved futures.

```
student score            s_i = similarity(f(x_q), f(x_i))
oracle candidate quality d_i = MSE(y_q, y_i)
Recall@10 ground truth   S_ind = Top-10 lowest MSE(y_q, y_i)
```

The project's founding assumption was: *if we find historical candidates whose
futures resemble the query future, Stage-2 forecasting improves.* This campaign
tested that assumption.

# Why This Experiment Was Run

Stage-1 Recall@10 turned out to be substantially improvable — roughly 2–4× via
asymmetric / pair-MLP scorers, with the largest gains at long horizons. The
initial hypothesis was that cosine's symmetric-comparison constraint was the
bottleneck. The campaign asked whether that recall gain actually transfers.

# Experiment Performed

A staged diagnostic campaign, not a single comparison. In order:

1. Stage-1 score/loss sweep (recall improvability)
2. Stage-2 transfer check on Weather (post-wiring-fix)
3. Retrieval-gain analysis under residual fusion
4. Correlation of Stage-2 MSE against Recall@10 vs aggregate-future-quality
5. Rank-loss arms → representation-collapse diagnosis
6. Frozen-encoder + rank-scorer isolation
7. Global KL anchor with gradient-conflict analysis
8. Swap analysis of which candidates the rank scorer inserted/removed
9. `I = A + V` decomposition
10. Set-level Oracle experiment (Cosine vs Individual Oracle vs Greedy Set Oracle)
11. Good+Diverse Oracle control
12. **Phase 0** — per-horizon temperature calibration (complete)
13. **Phase A** — pool × K sensitivity oracle (complete)
14. **Phase C** — direct Oracle imitation gate (complete)
15. Greedy Set Oracle restart-stability check

# Baseline

- **Recall reference:** random Top-10 overlap in a 100-candidate pool = **0.10**;
  uniform cross-entropy = ln(100) ≈ **4.605**. Both pinned before running.
- **Stage-1 baseline arm:** KL + Cosine (avg R@10 = 0.0460).
- **Selection baseline:** Cosine Top-10 within the fixed pool.
- **Forecast baseline:** no-retrieval base forecaster; residual fusion
  `Y = Y_base + λ·Y_ret`.
- **Seed noise in this repository:** ≈0.01 MSE. Smaller differences are not
  interpretable.

# Key Configuration

ETTh1 (primary), Weather (Stage-2 verification). `features=M`,
`seq_len = pred_len`, horizons 96/192/336/720, `top_k=10`, `relation_top_n=3`,
`d_model=128 / n_heads=4 / e_layers=2 / d_layers=1 / d_ff=256 / label_len=0`.
Frozen-encoder probes: healthy WCE-trained Stage-1 encoder, fixed per-query pool
P100, 3 epochs. Candidate memory bank at H96: `(7, 1, 8449, 128)`, valid pool
8449. Per-probe **seed is not recorded in the logs — UNKNOWN** (CLI default 0,
unconfirmed).

# Key Numerical Results

### 1. Recall@10 is improvable [user]

| Arm | H96 | H192 | H336 | H720 | Avg |
|---|---|---|---|---|---|
| KL + Cosine (baseline) | 0.0578 | 0.0556 | 0.0489 | 0.0216 | 0.0460 |
| WCE + Asymmetric | 0.0544 | 0.0926 | 0.1221 | 0.0942 | 0.0908 |
| WCE + MLP | 0.0664 | 0.0953 | **0.1311** | 0.0939 | **0.0967** |

### 2. Recall and Stage-2 MSE move in opposite directions [user]

Weather H96, post-wiring-fix — **the only valid downstream evidence available**:

| Arm | R@10 | Stage-2 final MSE |
|---|---|---|
| Cosine + KL | 0.0583 | 0.1925 |
| Asymmetric + KL | 0.0044 | **0.1794** |
| Δ | **−92.5%** | **−6.8% (better)** |

### 3. Stage-2 does use retrieval [user]

Retrieval gain: ETTh1 H96 ≈ 40.05%, H720 ≈ 6.18%; Weather H96 ≈ 7.75%,
H720 ≈ −1.39%. "Stage-2 ignores retrieval" is rejected.

### 4. Aggregate quality tracks Stage-2; candidate identity does not [user]

Spearman vs Stage-2 MSE:

| Metric | ETTh1 | Weather |
|---|---|---|
| Recall@10 | +0.032 | −0.350 |
| HardAggregateMSE@10 | **+0.810** | **+0.935** |

`HardAggregateMSE@10 = MSE(mean_i y_i, y_q)`;
`IndividualFutureMSE@10 = (1/K)·Σ_i MSE(y_i, y_q)`.

### 5. Rank-only encoder training collapses the representation [user]

Effective rank 16.38 (step 0) → 3.23 (5) → 1.90 (10) → 1.07 (epoch 1); pairwise
cosine ↑, sv1 fraction → ≈0.99. Proposed causal order:
**collapse → cosine saturation → gradient weakening** (not the reverse).

### 6. Removing collapse does not fix retrieval [user]

Frozen encoder, scorer-only training:

| | Frozen Cosine | Frozen + Asym Rank |
|---|---|---|
| PairAcc100 | 0.53185 | **0.54605** |
| LargeGapPairAcc | 0.56134 | **0.58531** |
| MissedBetter | 76.57 | **71.65** |
| Recall@10 | **0.05775** | 0.02046 |
| Spearman | **0.44973** | 0.30047 |
| RetrievedMSE@10 | **0.68637** | 1.01100 |

Local ordering improves; global Top-K retrieval degrades sharply.

### 7. Global KL anchor preserves ordering, not utility [user]

Retention rises with β (Top10 0.142 → 0.344 → 0.716; Top100 0.270 → 0.564 →
0.833). But `cos(g_rank, g_global) < 0`, conflict grows with β, and at β=1 only
≈**3.7%** of anchor gradient lands on Top-10 candidates — the rest on rank 101+.

### 8. The rank scorer inserted *better* candidates and still lost [user]

Removed-candidate future MSE 0.73979 → added 0.59782 (better). Top-10 individual
MSE improved. Candidate variance fell: **−74%** (rank β=0), −65% (anchor β=0.1),
−42% (β=1).

### 9. Exact decomposition [repo] — verified

Under uniform Top-K aggregation, `I = A + V` where `V` is mean candidate spread
about their own mean. Verified numerically: the `residual` column of every row of
`pool_k_sweep_pred*.csv` is ≤ **3e-9**. Lowering `I` while lowering `V` more
*raises* `A`.

### 10. Set Oracle vs Individual Oracle [repo] — recomputed from raw CSVs

Per-query means over 855–896 query×channel units, same fixed pool:

| | I_ind | I_set | A_ind | A_set | gain on A |
|---|---|---|---|---|---|
| H96 | 0.2604 | 0.3417 | 0.1303 | **0.0965** | **25.9%** |
| H192 | 0.3720 | 0.4632 | 0.2071 | **0.1680** | **18.9%** |
| H336 | 0.4977 | 0.5832 | 0.3079 | **0.2701** | **12.3%** |
| H720 | 1.4645 | 1.5310 | 1.1945 | **1.1463** | **4.0%** |

Set Oracle is **worse individually and better in aggregate at every horizon**.
Joint condition `A_set < A_ind AND I_set ≥ I_ind` holds for **100.0%** of query
units at all four horizons in this recomputation. Not a tail phenomenon.

### 11. Generic diversity fails [user]

Good+Diverse Oracle vs Individual Oracle aggregate: **−6.8% / −1.8% / −6.8% /
−3.8%** (H96/192/336/720) — worse everywhere. At H96 its variance (0.245785)
essentially equals Set Oracle's (0.245243) while aggregate differs greatly
(0.139201 vs 0.096719). **Variance is not the operative variable.**

### 12. Both bottlenecks are large [repo]+[user]

Phase A, pool=100 / K=10 / H96 [repo]:

| Arm | I | A | V |
|---|---|---|---|
| cosine | 0.594556 | 0.318727 | 0.275829 |
| individual | 0.259714 | 0.126704 | 0.133010 |
| set | 0.348854 | **0.089579** | 0.259275 |

[user] Full-memory Individual/Set Oracle Top-10 overlap = **0.265** (≈73.5% of
selected candidates differ). Candidate-selection gap ≈213%; set-composition gap
≈102%.

### 13. Phase 0 — temperature calibration [repo] — complete

Target band `N_eff ≈ 30–60`, `Mass@10 ≈ 0.5–0.8`:

| Horizon | chosen τ | N_eff | Mass@10 |
|---|---|---|---|
| H96 | 0.015 | 52.6 | 0.6332 |
| H192 | 0.015 | 56.3 | 0.6068 |
| H336 | **0.01** | 35.9 | 0.7129 |
| H720 | **0.02** | 49.9 | 0.6139 |

A single τ gives very different support per horizon (τ=0.015 → N_eff 52.6 / 56.3
/ 93.6 / 24.3), so any un-calibrated soft-objective comparison is confounded.

### 14. Phase C — direct Oracle imitation fails [repo] — verified exactly

ETTh1 VAL, epoch 3, frozen encoder + asymmetric scorer + fixed P100:

| | H96 individual | H96 set | H720 individual | H720 set |
|---|---|---|---|---|
| TeacherSetRecall@10 | 0.1363 | 0.1110 | 0.1279 | 0.0847 |
| imitation loss | 4.589 | 4.606 | 4.608 | 4.621 |
| uniform aggregate MSE@10 | 0.7430 | 0.7304 | 1.7593 | 1.6394 |
| individual MSE@10 | 0.9954 | 1.1005 | 2.1804 | 2.1500 |

Against random 0.10 and uniform CE 4.605, **every arm is at chance**. Critically,
the **Individual** target fails as badly as the Set target.

### 15. Target-noise explanation rejected [repo]

Greedy restart overlap 0.878 (H96) / 0.902 (H720); restart relative aggregate gap
≈1.6% / ≈0.5%.

# Additional Diagnostics

- **Set-target vs Individual-target imitation [user] — weak signal only.** The
  Set-target arm has lower TeacherSetRecall but better aggregate MSE
  (H96 0.7304 vs 0.7430; H720 1.6394 vs 1.7593). **Not a controlled comparison
  against a proper baseline** — offered only as further evidence that exact
  membership matching ≠ aggregate utility. Please do not upgrade this to a
  conclusion.
- Set-composition gain shrinks monotonically with horizon (25.9% → 4.0%),
  matching the pre-campaign observation that H720 behaves differently.

# Implementation Changes

Uncommitted working-tree changes (~2,600 lines) plus 8 new test files add:
boundary hard-pair rank loss + mining; rank-failure diagnostics (score-level
gradients, persistent pair sets); collapse/score geometry probes; frozen-encoder
scorer path; global anchor KL with retention reporting; set-level retrieval loss
(`soft_set_mse`, `hard_aggregate_metrics`); set-utility oracle selectors
(`select_individual_oracle`, `select_good_diverse`, `select_greedy_set`,
`greedy_set_stability`); Stage-2 retrieval-off counterfactual; Stage-2 selection
and redundancy reporting.

**A wiring bug was found and fixed mid-campaign** — see *Possible Issues* below.

# Sanity Checks

- `pytest tests/`: **396 passed, 2 failed**. Both failures reproduce in a clean
  worktree at HEAD `c306def`, so they predate this work and are the accepted
  baseline, not a regression.
- `I = A + V` identity verified numerically (residual ≤ 3e-9, all rows).
- Random and uniform reference points pinned before Phase C.
- Greedy Set Oracle restart stability measured before using it as a target.
- Phase 0 run before any soft-objective comparison, precisely to remove τ as a
  confound.

# Claude's Technical Observations

Implementation-level only.

1. **Recomputation matched the logs on the load-bearing results.** Phase C
   imitation figures and the Phase 0 τ selections reproduce exactly.
2. **Six user-supplied figures did not reproduce.** Conclusions are unaffected,
   but the numbers should be corrected before any writeup:

   | Quantity | Reported | Recomputed / logged |
   |---|---|---|
   | H336 set-oracle gain | 11.8% | **12.3%** |
   | H720 set-oracle gain | 3.6% | **4.0%** |
   | Joint-condition fraction | 99.9 / 100 / 97.8 / 95.5% | **100.0%** at all four |
   | H336 N_eff at τ=0.015 | ≈32 | **93.6** (≈35.9 occurs at τ=0.01) |
   | H336 N_eff / Mass@10 at τ=0.1 | ≈1632 / ≈0.065 | **2444.9 / 0.0202** |
   | H96 cosine I / A / V | 0.6612 / 0.3573 / 0.3038 | **0.594556 / 0.318727 / 0.275829** |

   The last row suggests the §10 "Cosine" row and the Phase A cosine arm come
   from different pool constructions; worth resolving before publication.
3. **Per-probe seeds are not in the logs.** Reproducibility of this campaign is
   therefore partial.
4. **Only Weather has valid post-fix downstream evidence** — a single dataset at
   a single horizon carrying claim 2.
5. Phases B, D, E, F of the stated 7-phase plan are unidentified in the artifacts.

# Possible Issues / Confounds To Check

1. **Wiring bug (most important).** The configured scorer was not propagated into
   Stage-2's actual Top-K selection path; some runs selected via the cosine path
   regardless of the trained scorer. Weather was re-verified post-fix. **The
   pre-fix ETTh1 Stage-2 sweep is invalidated and needs a full rerun** over
   {KL, WCE} × {Cosine, Asymmetric, MLP} × 4 horizons. Please judge whether any
   claim here still leans on pre-fix ETTh1 numbers.
2. **Single-dataset evidence.** Claim 2 rests on Weather H96 alone. Is one
   dataset at one horizon adequate to assert Recall–MSE decoupling?
3. **Checkpoint selection.** Stage-1 writes several side checkpoints
   (`best_recall10`, `best_ndcg10`, `best_retrieved_mse10`,
   `best_hard_aggregate_mse10`). Selecting on a recall criterion while evaluating
   an aggregate criterion would bias comparisons.
4. **Optimization vs capacity in Phase C.** Only 3 epochs, LR decayed to 2.5e-4.
   Could this be under-training rather than unlearnability?
5. **Oracle definition and pool construction.** The Individual Oracle is defined
   within P100, which is itself cosine-derived — the Oracle is conditioned on the
   very retriever under test.
6. **Seed / randomness.** Per-probe seeds unrecorded; seed noise ≈0.01 MSE.
7. **Metric space.** Verify MSE is compared in a consistent normalized space
   across `I`, `A`, and Stage-2 final MSE.
8. **Train/eval mismatch.** Phase C reports validation; the memorization probe
   will report train. Ensure the pool and target construction are identical.
9. **Greedy Set Oracle is a heuristic, not an optimum.** Its 88–90% restart
   overlap shows stability, not optimality — the true set optimum may be better
   still, making the reported gaps lower bounds.
10. **Parameter count.** Asymmetric vs pair-MLP scorers differ in capacity; the
    B1 sweep does not control for it.

# Questions for ChatGPT

1. What does this campaign actually demonstrate?
2. What does it **not** demonstrate?
3. Is "Recall@10's ground truth is structurally misaligned with Stage-2's
   set-level objective" supported, or does the evidence only support the weaker
   "Recall correlates poorly with Stage-2"?
4. What alternative explanations remain — particularly for the Phase C failure
   (under-training? pool construction? frozen representation? genuine
   past-only unpredictability)?
5. Does the literature already contain this? Specifically: set-level /
   submodular retrieval selection, diversity vs complementarity in RAG or
   ensemble selection, and retrieval trained end-to-end against a downstream loss
   rather than a relevance label. Is `I = A + V` a known decomposition in this
   context?
6. Is "candidate-wise retrieval objectives are misaligned with set-level
   downstream utility in retrieval-augmented forecasting" novel enough to pursue
   as the paper's main contribution?
7. Which next experiment has the highest scientific value — A/B memorization,
   cross-dataset feasibility, Track C E2E, or the ETTh1 wiring-fixed rerun?
8. Is the next experiment necessary, or is the current evidence already
   sufficient for the claim being made?
9. Does this campaign strengthen the main contribution, or is it an ablation /
   negative result that reframes the paper?

Additional: the project's stated contribution (see `RESEARCH_CONTEXT.md`) was
leakage-free cross-channel retrieval with a future-aware retriever. **If Q1 turns
out true — future-derived retrieval supervision does not generalize from
past-only input — does the "future-aware retriever" contribution survive?**

# Candidate Next Experiments

Technically feasible; the choice is the reviewer's and the user's, not Claude's.

1. **A/B — small-N memorization.** 32–64 fixed train queries, fixed P100, frozen
   WCE encoder; targets {Individual, Set} × scorers {asymmetric,
   `PairwiseScorer(pair4)`}. Separates wiring / capacity / generalization.
   Outcomes pre-specified as A/B/C/D.
2. **Cross-dataset feasibility.** Same Individual-Oracle imitation on ETTh1 /
   ETTm1 / Weather at H96. Separates dataset-specific from formulation-level
   difficulty.
3. **Track C — E2E forecasting-aligned retrieval.** Soft retrieval
   `p_i = Softmax(s_i/τ_H)`, `L = MSE(Y_final, Y_q)`, with control arms E0 / E1 /
   E1-soft / E2 sharing base checkpoint, Stage-2 init, seed and split
   (`corr(base_mse, final_mse) ≈ +0.958` makes this control mandatory).
   Case 4 — imitation fails but E2E succeeds — would be the highest-value outcome.
4. **ETTh1 Stage-2 wiring-fixed rerun.** Blocking for any ETTh1 downstream claim.

Full specifications with decision criteria: `research/CURRENT_EXPERIMENT.md`.

---

# Do Not Conclude

The researcher explicitly flagged these as unsupported. Please treat any of them
appearing in a writeup as an error:

| Claim | Why unsupported |
|---|---|
| "More diversity is always better" | Good+Diverse Oracle lost at every horizon |
| "A pointwise scorer cannot represent the Set Oracle" | Individual-Oracle imitation failed equally |
| "ETTh1 itself is the problem" | Cross-dataset feasibility not yet run |
| "E2E is the answer" | Track C not yet run |
| "The existing ETTh1 Stage-2 sweep shows asymmetric/MLP downstream effect" | Wiring bug; rerun required |
| "Recall does not matter" | Cosine Top-10 is far worse than Individual Oracle; the candidate-quality gap is real |

Most accurate current framing:

> Retrieving good candidates and constructing a good Top-K set are two distinct
> problems; and more fundamentally, it is not yet established that future-derived
> retrieval supervision generalizes from past-only information at all.

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.
