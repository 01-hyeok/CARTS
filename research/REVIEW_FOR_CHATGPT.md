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

# EXP-3 — soft_set_mse: representative Stage-2 closure (COMPLETE — VERDICT: STOP)

## Research Question

Does a differentiable, aggregate-aligned Stage-1 loss (`soft_set_mse`, full
memory bank softmax) improve real Stage-2 downstream forecasting, not just an
oracle upper bound (EXP-1/EXP-2 already showed the oracle upper bound is real)?

## Why representative closure instead of the full sweep

Stage-1 (20/20 runs) showed the soft-loss arms disagreeing in direction between
datasets (ETTh1: hard_aggregate_mse10 got worse; Weather: it improved, but with
N_eff and collapse warning signs). Rather than run all 20 Stage-2 downstream
evaluations, one representative comparison per cell was prioritized to reach a
GO/STOP decision quickly: **S0 (WCE baseline) vs the single non-baseline arm
with the best VALIDATION `hard_aggregate_mse10`** (never TEST, never each arm's
own training objective) from the already-completed Stage-1 sweep. The
remaining 12 Stage-1×Stage-2 combinations were not run and are not needed for
this decision. 2 of the 8 representative runs (ETTh1 H96 S0/S2) were already
complete from an earlier partial pass and were reused as-is, not rerun.

Stage-2 architecture, base forecaster, gate (`residual`/`scalar`),
`relation_top_n=1`, `top_k=10`, `tau_topk=0.1`, split and training protocol are
identical across every arm and cell — only the Stage-1 checkpoint differs.

## Results — [repo]

| Dataset | H | Arm | S2 Final MSE | Delta vs S0 | Rel% | Stage-1 HardAgg@10 | Recall@10 | N_eff | Eff. rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | S0 (WCE) | 0.37312 | — | — | 0.4073 | 0.0570 | 94.6 | 20.8 |
| ETTh1 | 96 | S2 (lam10) | 0.37911 | +0.00599 | +1.61% | 0.4147 | 0.0516 | 201.1 | 12.1 |
| ETTh1 | 720 | S0 (WCE) | 0.46827 | — | — | 0.5732 | 0.0198 | 68.5 | 17.6 |
| ETTh1 | 720 | S3 (lam30) | 0.50947 | +0.04120 | +8.80% | 0.5864 | 0.0210 | 207.4 | 11.4 |
| Weather | 96 | S0 (WCE) | 0.17553 | — | — | 0.2348 | 0.0579 | 1498.7 | 12.2 |
| Weather | 96 | S1 (set only) | 0.17681 | +0.00128 | +0.73% | 0.2058 | 0.0548 | 8870.6 | 5.8 |
| Weather | 720 | S0 (WCE) | 0.31869 | — | — | 0.5672 | 0.0116 | 81.5 | 20.0 |
| Weather | 720 | S3 (lam30) | 0.38574 | +0.06705 | +21.04% | 0.4650 | 0.0105 | 700.0 | 4.4 |

MSE lower is better. **Every representative arm is worse than S0 at every cell
— 0 of 4 improved.** ETTh1 H720 and Weather H720 exceed the repository's
recorded seed noise (≈0.01 MSE) by 4-7x, decisively in the worse direction; the
two H96 deltas are within/near noise but are also the wrong sign to claim
improvement regardless of magnitude.

Weather's Stage-1-internal `hard_aggregate_mse10` *did* improve (−12.3% H96,
−18.0% H720) — but `N_eff` exploded (+492%, +759% over baseline, against a
~36,700-candidate bank) and `effective_rank` collapsed (−53%, −78%) at the same
time. The Stage-1 improvement does not survive Stage-2 and reads as the
diffusion/collapse confound flagged as a risk before this run, not as genuine
retrieval learning.

## Verdict: STOP

All of the pre-registered stop conditions hold: ETTh1 fails to improve Stage-2
at both horizons; Weather's apparent gain is confounded by N_eff explosion and
representation collapse; all 4 cells move in the same (worse) direction rather
than merely disagreeing; two of four deltas exceed seed noise and the other two
are the wrong sign. **No further lambda/tau/support-penalty sweep is planned on
this direction.**

## Conclusion (stated within what the data supports)

Set Oracle analysis (EXP-1/EXP-2) showed real headroom in candidate
*combination* — a Weighted-Set Oracle beats an Individual Oracle on
A_weighted at every horizon tested, on both datasets. But training a
full-memory soft aggregation loss (`soft_set_mse`) to capture that headroom did
not transfer to Top-K retrieval or downstream forecasting: on ETTh1 the soft
objective improves while the hard, deployed Top-10 aggregate does not
(soft/hard selection mismatch); on Weather the apparent hard-aggregate gain is
inseparable from the loss diffusing probability mass across thousands of
candidates while the encoder representation collapses. Neither dataset shows a
Stage-2 improvement that survives these confounds.

## What this does NOT establish

- That no differentiable aggregate-aware Stage-1 loss could ever work — only
  that a full-memory softmax over the whole bank, at these temperatures and
  lambdas, does not.
- That the Individual Oracle → Weighted-Set Oracle gap (EXP-1/EXP-2) is closed
  or irrelevant — it remains a real, measured upper bound; this experiment
  shows one specific way of trying to reach it failing, not that the gap is
  unreachable in principle.
- Anything about cross-channel relations — this stayed self-only
  (`relation_top_n=1`) throughout, per the standing decision to keep that
  question separate.

## Questions for ChatGPT

1. Is "STOP on this specific loss formulation" the right scope for the
   conclusion, or does the ETTh1 vs Weather divergence (opposite Stage-1
   directions, same Stage-2 direction) suggest something more general about
   full-memory soft losses that should be stated more strongly?
2. Given the confirmed Individual→Set Oracle gap and this closure's negative
   result, is "retrieve-then-rerank" (coarse Top-100 by the existing retriever,
   then a separate reranker minimizing set-level utility only inside that
   shortlist) the most information-dense next experiment, or is there a
   cheaper diagnostic that should come first?
3. Does the Weather N_eff/collapse pattern (worse baseline representation
   quality **and** a much larger relative diffusion under the same loss)
   suggest the representation should be stabilized before any further
   set-level objective is tried on Weather specifically?

## Candidate next experiment considered, then independently tested (see below)

The reranker-on-a-shortlist design proposed after EXP-3 (still in
`research/CURRENT_EXPERIMENT.md`, unstarted) is one candidate direction. A
*different* direction — full-memory residual-conditioned retrieval, explicitly
avoiding any shortlist — was specified by the researcher and run to
completion as EXP-FRR01 below, independent of this candidate.

---

# EXP-FRR01 — Full-memory forecast-conditioned residual retrieval (COMPLETE — VERDICT: STOP)

## Research Question

Redefine retrieval from "similar future" to "similar historical
forecast-error/residual pattern." Five arms, each adding one mechanism on top
of the last, all full-memory (no Top-M shortlist anywhere):

```
R0   residual teacher only: WCE's Top-K target graded by residual similarity
     (S_R(q,k) = -MSE(R_q, R_k), R = Y - base_forecast) instead of raw future
     similarity, everything else (architecture, loss family) unchanged
R1   R0 + query conditioning: the query embedding gets an additive projection
     of its own base forecast (Y_q - R_q) before scoring
R2   R0 + candidate conditioning: same additive idea, on each candidate's own
     historical residual, at every embedding site (key bank, differentiable
     re-encoding, full_online re-encoding)
R12  R1 + R2 together
R3   R12 + an asymmetric dual-encoder comparison (separate W_q/W_k
     projections) in place of plain cosine
```

Primary (only) success metric: Stage-2 Final MSE. Pre-registered rule: an
arm needs to beat B0 by >= 0.01 to trigger a 3-seed confirmation run;
Stage-1-proxy improvement (recall, hard_aggregate_mse10) alone is not success.
ETTh1 H96 only, 1 seed, per the pre-registered pilot scope.

## A real architecture gap found mid-run

Before any Stage-2 run, auditing `RelationStage2.load_stage1_checkpoint`
showed it only transplanted `encoder`/`shared_cross_projection`/
`retrieval_metric` weights from a Stage-1 checkpoint — nothing loaded the new
`query_cond_proj`/`candidate_cond_proj` conditioning modules R1/R2 add. Left
as-is, Stage-2 would have kept only "an encoder shaped by conditioning during
training" and silently dropped R1/R2's actual mechanism at retrieval time —
not what the arms are defined to be. Flagged to the researcher before
proceeding; they chose to wire it properly (over two faster/partial options)
rather than accept a reduced test. Full account in
`results/EXP-FRR01/notes.md`.

## Results — [repo], ETTh1 H96, B0 reused from EXP-3 (0.37312, not rerun)

| Arm | Stage-1 HardAgg@10 | Recall@10 (orig) | Recall@10 (residual-target) | Stage-2 Final MSE | Delta vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.4073 | 0.0570 | — | 0.37312 | — |
| R0 | 0.4154 | 0.0507 | 0.0282 | 0.37397 | +0.00085 |
| R1 | 0.4703 | 0.0505 | 0.0477 | 0.37099 | -0.00213 |
| R2 | 0.4667 | 0.0861 | 0.0606 | 0.38023 | +0.00711 |
| R12 | 0.4655 | 0.0866 | 0.0669 | 0.38112 | +0.00800 |
| R3 | 0.4537 | 0.0805 | 0.0596 | 0.37953 | +0.00641 |

No arm crosses +/-0.01 -> no 3-seed confirmation triggered, no GO. R1's small
improvement (-0.00213) is below this project's own ~0.01 seed-noise reference
and is not distinguishable from noise at 1 seed. R2/R12/R3 raised Recall@10
substantially (+51%/+52%/+41% relative to B0) while Stage-1
`hard_aggregate_mse10` and Stage-2 Final MSE both got worse.

## Sanity checks passed

`pytest tests/`: 444 passed, exactly the 2 pre-existing repository failures,
both before and after every code change. 1-epoch smoke test of R3 (exercises
every new code path at once) run to completion at both stages before the real
10-epoch runs. `[legality]` assertion (candidate-side `memory_residual` row
count equals the train split's query count, i.e. it is the train-only
archive) passed for every Stage-1 arm. Retrieval-selection wiring guard
(`configured_metric == actual_selection_score_fn`) passed for R3 at Stage-2.

## Conclusion (stated within what the data supports)

Redefining the retrieval target/representation around residual similarity,
in five increasingly complex forms, does not improve Stage-2 forecasting on
ETTh1 H96 at 1 seed. The R2/R12/R3 result reproduces EXP-C01/EXP-3's
"Recall@10 does not predict Stage-2" finding through a structurally different
mechanism (embedding conditioning, not a soft aggregate loss) — evidence the
disconnect is about what Recall@10 measures, not an artifact specific to the
soft_set_mse loss family.

## What this does NOT establish

- That no residual-conditioned retrieval mechanism could ever help — only
  that this specific set of additive, frozen-Stage-1-encoder mechanisms does
  not, at 1 seed, on ETTh1 H96.
- Anything about Weather or other horizons — not run, per the pre-registered
  scope (no arm reached the 3-seed/follow-up bar that would have justified
  extending scope).
- Anything about cross-channel relations — self-only (`relation_top_n=1`)
  throughout.

## Questions for ChatGPT

1. Is a single ETTh1-H96/1-seed pilot sufficient evidence to close this whole
   direction, or does R1's small (noise-scale) improvement in the opposite
   direction from R2/R12/R3 warrant a second seed before concluding STOP,
   even without crossing the pre-registered 0.01 bar?
2. R2/R12/R3 raising Recall@10 by ~50% while making both `hard_aggregate_mse10`
   and Stage-2 worse is now a two-mechanism replication (soft losses in EXP-3,
   embedding conditioning here). Is there a more direct diagnostic that would
   explain *why* Recall@10 is this decoupled from aggregate/downstream
   quality, rather than reconfirming that it is?
3. Given EXP-1/EXP-2's confirmed Individual->Set-Oracle gap remains real and
   unreached by every mechanism tried so far (soft losses, residual
   conditioning), is the reranker-on-a-shortlist design in
   `research/CURRENT_EXPERIMENT.md` now the most promising remaining
   candidate, or does EXP-FRR01's result suggest that direction should also be
   pressure-tested against a cheaper diagnostic first?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.

