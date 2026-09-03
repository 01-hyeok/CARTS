# REVIEW_FOR_CHATGPT.md

Handoff for independent review. Self-contained: no log files needed.

**Status: two completed experiments (EXP-1, EXP-2) + one experiment IN PROGRESS
(EXP-3, Stage-1 only, Stage-2 not yet run).** EXP-3 is included because its
interim Stage-1 signal is already informative and the researcher wants a read
before the remaining ~6+ hours of compute finishes. Do not treat EXP-3's
numbers as final — explicitly flagged below.

Provenance: **[repo]** = read/recomputed from artifacts by the implementation
engineer; **[user]** = supplied by the researcher, not independently
reproduced.

---

# Background — established before this document

From the 2026-09 diagnostic campaign (`EXP-C01`, see `EXPERIMENT_LOG.md` for
full detail): Stage-1 Recall@10 is improvable several-fold, but Recall@10 does
not track Stage-2 forecast quality; `HardAggregateMSE@10` (the error of the
*aggregate* the retrieved Top-K forms) tracks it far better; and a Set Oracle
that deliberately trades individual candidate quality for a better *combination*
beats an Individual Oracle on aggregate error at every horizon tested. That
motivated the two experiments below.

---

# EXP-1 / EXP-2 — Individual Oracle vs Weighted-Set Oracle (completed)

## Research Question

Holding the Stage-2 model, base forecaster, gate, and candidate support fixed,
does replacing "the ten individually best candidates" (Individual Oracle) with
"the ten candidates that jointly minimise Stage-2's own weighted aggregate"
(Weighted-Set Oracle) lower the retrieval aggregate error?

## Method

Causal intervention, no training. Two selection rules over the same **full
memory bank** (every valid candidate, not a shortlist):

```
R1  Individual Oracle:   d_i = MSE(target_future_i, Y_q); take the 10 smallest
R2-W Weighted-Set Oracle: alpha_i = softmax(score_i / tau_topk)
                          greedily build the 10-set that minimises
                          MSE(sum_i alpha_i * target_future_i, Y_q)
                          (softmax re-normalised over the whole set at every
                          greedy step -- adding a candidate dilutes every
                          weight already assigned, not just adds a term)
```

Primary metric: **A_weighted** = `MSE(sum_i alpha_i * V_i, Y_q)` in the same
space Stage-2 actually consumes (normalized absolute, target channel, TEST
split). This is *not* Stage-2's end-to-end final MSE — it is the retrieval
aggregate's own error, upstream of the base-forecaster fusion.

Config: `relation_top_n=1` (self-retrieval, not cross-channel), `top_k=10`,
`tau_topk=0.1`, `fusion_mode=residual`, `gate_mode=scalar`
(`Y_final = Y_base + lambda*Y_ret`, lambda from a learned scalar gate — **not**
the older `raft_concat` (concat+Linear, no gate) architecture used in the
pre-campaign baselines in `RESULTS_SUMMARY.md`; the two are not
architecturally comparable). Identical git commit, identical Stage-2
checkpoint per horizon (state_dict SHA256 re-verified unchanged across arms).

## Results — [repo] recomputed from raw CSVs

### ETTh1 (FULL memory, TEST, K=10) — all 4 horizons complete

| H | Individual Oracle A_weighted | Weighted-Set Oracle A_weighted | Improvement |
|---|---:|---:|---:|
| 96 | 0.1442 | 0.1005 | 30.29% |
| 192 | 0.1958 | 0.1447 | 26.12% |
| 336 | 0.2401 | 0.1861 | 22.50% |
| 720 | 0.3116 | 0.2565 | 17.70% |

Individual candidate quality (`I`, not shown) is *worse* for the Set Oracle at
every horizon — it is deliberately trading individual quality for a better
joint aggregate, and it wins on the metric Stage-2 actually consumes at every
horizon.

### Weather (FULL memory, TEST, K=10) — 2 of 4 horizons complete

| H | Individual Oracle A_weighted | Weighted-Set Oracle A_weighted | Improvement |
|---|---:|---:|---:|
| 96 | 0.0292 | 0.0166 | 43.07% |
| 192 | N/A | N/A | run started, interrupted before completion (freed for other work) |
| 336 | N/A | N/A | never run |
| 720 | 0.1001 | 0.0568 | 43.24% |

Same direction as ETTh1, larger magnitude. H192/H336 are a genuine gap, not a
result of any kind — no number exists to report.

## Sanity checks passed

`I = A_u + V_u` and the weighted analogue verified to residual < 1e-6 on every
row; per-query CSV means match the summary CSV exactly (diff = 0); support
confirmed FULL (not a cosine-shortlisted P100) and split confirmed TEST via
runtime log assertions; same Stage-2 checkpoint used by every arm within a
horizon (SHA256-verified).

---

# EXP-3 — Learning a Stage-1 loss aligned with the aggregate (IN PROGRESS)

## Research Question

EXP-1/EXP-2 showed an *oracle* aggregate-aware selection beats an oracle
individual selection. Can a **differentiable, trainable** Stage-1 loss that
targets the same aggregate quantity actually be learned, and does it improve
the real (hard, Top-K, deployed) retrieval — not just an oracle upper bound?

```
p_i = softmax(s_i / tau_set)              over the WHOLE valid memory bank
Y_ret_soft = sum_i p_i * Y_i
L_set = MSE(Y_ret_soft, Y_q)
L_total = wce_weight * L_WCE + lambda_set * L_set
```

This reuses the pre-existing `soft_set_mse()` / `hard_aggregate_metrics()`
implementation and Stage-1 loss mode `wce_soft_set_mse`, unit-tested before
this run (existing 7 tests: manual weighted-aggregate match, complementary
candidates, gradient reaches every valid score, invalid candidates get zero
gradient, one-sided support hinge, effective-support-vs-temperature). One new
flag, `--stage1_wce_weight` (default 1.0), was added to reach `L = L_set` alone
(S1) without writing a new loss.

## Arms (score function and everything else held fixed; only the loss changes)

| Arm | wce_weight | set_mse_weight |
|---|---:|---:|
| S0 (WCE baseline) | 1.0 | 0.0 |
| S1 (SetMSE only) | 0.0 | 1.0 |
| S2 | 1.0 | 10.0 |
| S3 | 1.0 | 30.0 |
| S4 | 1.0 | 50.0 |

`tau_set`: ETTh1 uses the pre-existing Phase-0 calibrated values (0.015 @ H96,
0.02 @ H720). Weather had no such calibration on record for its actual
`tau_calibration_diag` method — the method itself had been lost before being
committed and was reconstructed and acceptance-tested against the four
surviving ETTh1 ground-truth logs (matched to 4-5 significant figures at every
sweep point before being trusted) — then run fresh for Weather. Weather's
result was a genuine two-sided conflict, not a tuning failure: at H96 the
target `N_eff` band (30-60) is **structurally unreachable** — N_eff floors at
≈2150 even as tau→0 — consistent with the severe representation collapse
already on record for the Weather encoder (`effective_rank_mean` ≈ 6.6); at
H720 `N_eff` is reachable but `Mass@10` is then pinned at ≈0.92 throughout the
whole reachable range, far outside its own (0.5, 0.8) target. Both were
resolved by the pre-registered tie-break rule (closest `Mass@10` to 0.5) rather
than forcing a value, **before** any Stage-1/Stage-2 result was seen. Chosen:
0.0025 (H96), 0.00105 (H720).

Checkpoint selection: **`hard_aggregate_mse10` on validation** for every arm
(not each arm's own training objective, and never the test split).

## Status: Stage-1 14/20 runs complete; Stage-2 0/20 (not started)

Completed: ETTh1 H96 (5/5), ETTh1 H720 (5/5), Weather H96 (4/5, S4 running).
Remaining: Weather H96 S4, then Weather H720 all 5 arms, then all 20 Stage-2
downstream runs. **The question this experiment exists to answer —
does a Stage-1 gain reach Stage-2's final forecast — cannot be answered yet.**
What follows is Stage-1-only and may not survive contact with Stage-2.

## Interim Stage-1 results — [repo]

### ETTh1 H96

| Arm | Recall@10 | HardAgg@10 | SoftSetMSE | N_eff |
|---|---:|---:|---:|---:|
| S0 WCE | 0.0570 | **0.4073** | 0.2190 | 94.6 |
| S1 SetMSE only | 0.0455 | 0.4226 | 0.1925 | 585.4 |
| S2 (lambda=10) | 0.0516 | 0.4147 | 0.2008 | 201.1 |
| S3 (lambda=30) | 0.0479 | 0.4220 | 0.1980 | 254.1 |
| S4 (lambda=50) | 0.0460 | 0.4258 | 0.1961 | 334.2 |

### ETTh1 H720

| Arm | Recall@10 | HardAgg@10 | SoftSetMSE | N_eff |
|---|---:|---:|---:|---:|
| S0 WCE | 0.0198 | **0.5732** | 0.2590 | 68.5 |
| S1 SetMSE only | 0.0197 | 0.6176 | 0.2652 | 345.9 |
| S2 | 0.0193 | 0.6017 | 0.2684 | 127.9 |
| S3 | 0.0210 | 0.5864 | 0.2568 | 207.4 |
| S4 | 0.0228 | 0.5868 | 0.2503 | 263.3 |

**Every set-loss arm is worse than WCE alone on `HardAggregateMSE@10` at both
ETTh1 horizons**, and `N_eff` (mean_q exp(entropy_q) of the softmax over the
full bank) inflates 3-9x over the WCE baseline as `lambda` grows.

### Weather H96 (4 of 5 arms; S4 still training)

| Arm | Recall@10 | HardAgg@10 | SoftSetMSE | N_eff |
|---|---:|---:|---:|---:|
| S0 WCE | 0.0579 | 0.2348 | 0.2650 | 1498.7 |
| S1 SetMSE only | 0.0548 | **0.2058** | 0.1371 | 8870.6 |
| S2 | 0.0450 | 0.2065 | 0.1407 | 4556.4 |
| S3 | 0.0534 | **0.1988** | 0.1385 | 5286.5 |
| S4 | — | training | — | — |

Here `HardAggregateMSE@10` *improves* with the set loss — the opposite
direction from ETTh1 — but `N_eff` for S1 is 8870 out of ≈36,700 valid
candidates, roughly 6x the (already very high, 1499) WCE baseline. A
representation-collapse diagnostic captured mid-run on the Weather S4 arm
showed `effective_rank_mean` falling to 2.75 and mean pairwise cosine rising to
0.91 — markedly worse than the WCE baseline's own already-poor 6.6.

## Claude's technical observations (implementation-level only, no research conclusion)

1. **ETTh1 reads as Case 3 from the pre-registered decision rules**: the soft
   objective improves (`SoftSetMSE` falls with `lambda`) but the hard,
   deployed Top-10 aggregate does not — a soft-relaxation-vs-hard-selection
   mismatch, not a representation problem (ETTh1's `N_eff` values, while
   inflated, stay in the tens-to-low-hundreds against ≈8,200-8,450 valid
   candidates).
2. **Weather's apparent improvement is confounded with two failure modes at
   once**: `N_eff` in the thousands out of ~36,700 (Case C, "excessive
   diffusion" — closer to averaging most of memory than to retrieving) and a
   further representation-rank drop as `lambda` increases (Case E,
   representation collapse). An aggregate-error improvement produced by
   averaging over thousands of near-duplicate embeddings is not evidence that
   the set-level objective is teaching useful retrieval.
3. Neither dataset has a Stage-2 number yet. Whether ETTh1's clean Case-3
   reading and Weather's confounded improvement both wash out, both persist, or
   diverge further at the fusion stage is unknown.

## Possible issues / confounds for the reviewer to weigh independently

- Is `hard_aggregate_mse10` at validation the right checkpoint criterion when
  the training loss can trivially lower it by inflating `N_eff` (diffusing
  toward the memory mean) rather than by learning a better retrieval geometry?
  A checkpoint selected this way could be selecting for diffusion, not utility.
- Weather's tau values were chosen by a fixed, pre-registered rule under a
  structural conflict the rule was not originally built to arbitrate (neither
  target being simultaneously satisfiable). Is "closest `Mass@10` to 0.5,
  restricted to the `N_eff`-band when one exists" still the right tie-break
  when the `N_eff` band is provably unreachable, versus reachable-but-conflicting?
- The representation-collapse severity gap between ETTh1 (`N_eff` stays modest)
  and Weather (`N_eff` in the thousands) tracks the two datasets' already very
  different baseline `effective_rank` (≈17-18 for ETTh1's WCE encoder vs ≈6.6
  for Weather's) — is Weather's result telling us anything about `soft_set_mse`
  specifically, or mainly reflecting that Weather's encoder was already close
  to collapsed before this loss was added?
- No Stage-2 result exists yet for either dataset. Please treat every number
  above as Stage-1-internal and not indicative of downstream utility.

## Questions for ChatGPT

1. Given only the Stage-1 evidence so far, is Case 3 (ETTh1) severe enough to
   deprioritize completing the Weather/Stage-2 legs of this experiment, or is
   the Weather divergence (opposite direction) reason enough to see it through?
2. Is "improves the training objective by increasing `N_eff` into the
   thousands" a result worth reporting as a negative finding in its own right
   (a concrete failure mode of naive full-memory soft-aggregate losses), even
   if Stage-2 numbers never materialize?
3. Does the ETTh1 Case-3 pattern (soft objective improves, hard deployed
   Top-10 does not) suggest a specific fix — e.g. restricting the softmax to a
   pre-selected shortlist rather than the whole memory bank — that would be a
   more informative next experiment than waiting out the current sweep?
4. Should checkpoint selection be changed to jointly gate on `N_eff` staying
   near the calibrated band, so a collapsing/diffusing checkpoint cannot be
   selected merely for a favorable `hard_aggregate_mse10`?

## Candidate next experiments (not chosen here)

- Finish the current sweep (Weather H720 Stage-1 + all 20 Stage-2 runs) before
  drawing any downstream conclusion — this is already queued and running.
- If Stage-2 confirms ETTh1's Case 3, try a shortlist-restricted soft objective
  (softmax over the model's own Top-100, not the full bank) as the direct fix
  for the soft/hard mismatch.
- If Weather's `N_eff` explosion is confirmed to be doing the work, an explicit
  `N_eff` penalty (already implemented as `stage1_set_support_k` /
  `stage1_set_support_weight`, currently unused at weight 0 in this sweep)
  is a natural, already-available follow-up rather than a new experiment.

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.
