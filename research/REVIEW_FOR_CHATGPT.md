# REVIEW_FOR_CHATGPT.md

Handoff for independent review. Self-contained: no log files needed.

**Status: nine completed/scoped experiments (EXP-1/EXP-2 Oracle
Intervention, EXP-3 soft_set_mse closure, EXP-FRR01 residual-conditioned
retrieval, EXP-SEQFULL01 sequential set-conditioned retrieval, EXP-SEQDIAG01
cross-dataset frozen-encoder collapse diagnostic, EXP-MARGUTIL01 dense
marginal-utility successor, EXP-FIRSTANCHOR-DIAG causal t=1 decomposition,
EXP-CONTINUATION-DIAG exhaustive t=2 continuation diagnostic,
EXP-TOPTAIL-RANK01 loss-formulation comparison R0/R1/R2). EXP-TOPTAIL-RANK01
is IN PROGRESS: ETTh1 H96 is complete for all three arms (R0 SmoothL1, R1
pairwise, R2 hybrid); Weather H96 R2 is still training (R1 was stopped
before completion by explicit user decision, so Weather H96 will compare
R0 vs. R2 only, not R1). This section will be refreshed again once Weather
H96 R2 finishes.
EXP-MARGUTIL01/EXP-FIRSTANCHOR-DIAG/EXP-CONTINUATION-DIAG are complete for
a reduced 3-cell scope (Weather H720 cancelled by explicit user decision,
D-0013) and all three need the reviewer's interpretation rather than
carrying a mechanical pass/fail verdict — see their sections at the end of
this document. EXP-CONTINUATION-DIAG's single most consistent finding:
extreme-top-tail utility ranking correlation is negative in all 9
dataset×horizon×anchor combinations tested, while global ranking
correlation is mixed.**

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
2. *(Superseded 2026-09-06 — the researcher's standing constraint is
   full-memory direct Top-K, not a shortlist/reranker stage; a
   shortlist-based next experiment is no longer a live candidate. Left here
   only as the historical record of what was asked when this section was
   written. See EXP-FRR01/EXP-SEQFULL01/EXP-SEQDIAG01 below for the
   full-memory directions actually pursued instead.)* Given the confirmed
   Individual→Set Oracle gap and this closure's negative result, is there a
   cheaper full-memory diagnostic that should come before another full-memory
   objective/procedure variant?
3. Does the Weather N_eff/collapse pattern (worse baseline representation
   quality **and** a much larger relative diffusion under the same loss)
   suggest the representation should be stabilized before any further
   set-level objective is tried on Weather specifically?

## Research direction taken instead (see below)

The reranker-on-a-shortlist design considered after EXP-3 was **not** pursued
— the researcher's standing constraint is full-memory direct Top-K
throughout, not a shortlist/reranker stage (see `research/CURRENT_EXPERIMENT.md`).
Three full-memory directions were specified and run to completion instead:
EXP-FRR01 (residual-conditioned retrieval), EXP-SEQFULL01 (sequential
set-conditioned imitation), and EXP-SEQDIAG01 (collapse-vs-generalization
diagnostic for EXP-SEQFULL01) — all below.

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
   unreached by every full-memory mechanism tried so far (soft losses,
   residual conditioning), what full-memory diagnostic (not a shortlist/
   reranker stage — outside this project's research constraint, see
   `research/CURRENT_EXPERIMENT.md`) would most cheaply test whether that gap
   is reachable at all from past-only information?

---

# EXP-SEQFULL01 — Full-memory set-conditioned sequential retrieval (COMPLETE — VERDICT: STOP)

## Research Question

Rather than scoring candidates independently (every prior arm) or relaxing
the set objective into a differentiable soft aggregate (EXP-3), directly
imitate the Full-Memory Weighted Set Oracle's own DISCRETE GREEDY CONSTRUCTION:
a sequential selector conditions each of K=10 steps on the set chosen so far
and scores the WHOLE valid memory bank at every step (never a Top-M
shortlist), trained via teacher-forced full-memory cross-entropy against the
oracle's own step order. Does this let hard Top-K retrieval capture the
Individual→Set-Oracle headroom EXP-1/EXP-2 measured, and does that reach
Stage-2?

## Method

`SetConditioner`: `concat(q, mean(selected embeddings)) -> Linear -> GELU ->
Linear -> residual -> norm` (a learned empty-set token at t=1). Teacher:
`utils.oracle_intervention.select_greedy_weighted_set` (the exact function
EXP-1/EXP-2 validated), run offline over the FULL memory bank per query,
weighted by B0's own frozen retrieval score (fixed, never the student's own
changing score) — cached once, never recomputed during training. Candidate
encoder re-encoded live (gradient on, no `.detach()`) once per optimisation
step and reused across all K steps, mirroring the existing `full_online`
mechanism. Stage-2: the final Top-10 is injected via
`RelationStage2.set_forced_selection`, the same mechanism EXP-1/EXP-2 already
use — Stage-2's base forecaster/gate/fusion/aggregation weighting are B0's,
completely unchanged; only Top-K *membership* differs.

ETTh1 H96 only, self-only (`relation_top_n=1`), 1 seed. B0 reused as-is
(EXP-3's S0_wce checkpoints, not rerun).

## Results — [repo]

**Small-N gate: PASS** (16 queries/256 candidates/single channel/400 steps).
Val overlap@10 with the tiny-universe teacher = 0.875 (correct chance for
this metric, `|S_pred∩S_teacher|/K`, is `K/N` ≈ 0.042 for N≈240, K=10 — an
earlier write-up of this section used the wrong formula, `1/N` ≈ 0.004; see
EXPERIMENT_LOG.md's ERRATUM). Still an unambiguous pass either way (~21x the
correct chance). Candidate-side gradient norm nonzero throughout (0.081); 0
duplicate/invalid selections (structural, not merely observed).

**Full ETTh1 H96 (8449 candidates): trained cleanly; a small, real
generalisation signal, far too weak to produce a competitive aggregate.**
Candidate gradient stayed nonzero across all 10 epochs (0.044–0.091); train
loss decreased monotonically. Validation Set overlap@10 ranged 0.009–0.013
across epochs 1–10. **Corrected chance baseline: `K/N` ≈ 0.00118** (an
earlier write-up used `K²/N` ≈ 0.0118, ~10x too large — see EXPERIMENT_LOG.md's
ERRATUM). The measured range is therefore **~8–11x the correct chance**, not
"at chance" — a real but weak signal.

| Quantity (test split) | Value |
|---|---:|
| Individual Oracle MSE (diagnostic) | 0.1910 |
| B0's own A_weighted (natural Top-10) | 0.4073 |
| Full-Memory Weighted Set Oracle A_weighted | 0.1030 |
| Sequential selector's own A_weighted | 0.5304 |
| gap_recovery = (A_B0 − A_seq)/(A_B0 − A_set_oracle) | **−0.404** |
| Sequential Set Recall@10 vs. oracle (chance ≈0.00118, corrected) | 0.0135 |
| Effective rank of trained encoder | 5.43 / 128 (B0's own ≈20.8) |

**Stage-2** (fingerprint-verified: this run's own B0-unforced pass reproduces
0.373122 against the recorded 0.37312):

| Arm | Stage-2 Final MSE | Delta vs B0 |
|---|---:|---:|
| B0 (reproduced) | 0.37312 | — |
| Sequential selector (forced Top-10) | 0.40277 | **+0.02965** |

## Sanity checks passed

`pytest tests/`: 458 passed (14 new), exactly the 2 pre-existing repository
failures, no regression. A dataloader-unpacking bug crashed the first
full-run attempt on batch 1 (caught immediately by the crash, fixed, rerun
from scratch — see `results/EXP-SEQFULL01/notes.md`). Structural masking
guarantees (0 duplicate, 0 invalid) verified both by direct measurement and
by 5 of the 14 unit tests.

## Conclusion (stated within what the data supports)

The sequential, teacher-forced, full-memory imitation of the Weighted Set
Oracle's own greedy construction memorizes cleanly at small N. At full scale
it shows a small, real generalisation signal (~8–11x the corrected chance
baseline for Set Recall@10) that is nonetheless far too weak to produce a
competitive aggregate: `gap_recovery` is negative (the sequential selector's
own aggregate is worse than B0's simple retriever's, not merely short of the
Set Oracle's), and Stage-2 Final MSE is worse than B0 by +0.02965 (~3x this
project's own seed-noise reference, wrong direction).

This is **H2-consistent, not H1**: the failure is not attributable to a soft
relaxation (EXP-3's mechanism) or to insufficient candidate/query information
richness (EXP-FRR01's mechanisms) — this arm removed both potential
confounds (discrete, teacher-forced, full-memory training; live
non-detached candidate gradient) and still produced an aggregate worse than
B0's naive retrieval. The revised (post-erratum) reading is more specific
than "zero generalization": weak-but-real membership-level signal does not
survive into aggregate/downstream quality. Whether the encoder's own
representation collapse (effective rank 20.8→5.43) is a *cause* of that
weak signal, rather than only a correlate, is exactly what EXP-SEQDIAG01
(below) was designed to separate.

## What this does NOT establish

- That no sequential/set-conditioned mechanism could ever work — only that
  this specific minimal instantiation (small MLP conditioner, mean-pooled set
  summary, cosine scoring, fresh encoder) does not generalize on ETTh1 H96 at
  1 seed.
- That the Individual→Set-Oracle gap (EXP-1/EXP-2) is unreachable in
  principle — it remains a real, measured upper bound; three structurally
  different attempts to reach it (soft relaxation, embedding conditioning,
  discrete sequential imitation) have now all failed to reach Stage-2, which
  is itself the accumulating evidence worth weighing.
- Anything about Weather, other horizons, or cross-channel relations — not
  run, per the pre-registered scope.

## Questions for ChatGPT (in addition to EXP-FRR01's, above)

4. Three structurally different mechanisms (EXP-3 soft relaxation, EXP-FRR01
   embedding conditioning, EXP-SEQFULL01 discrete sequential imitation) have
   now all failed to convert the confirmed Individual→Set-Oracle headroom
   into a Stage-2 improvement. Does this pattern point toward the headroom
   being fundamentally unreachable from past-only `X_q`/`X_i` information
   (a generalization/information ceiling), or is there a specific mechanism
   class not yet tried that these three failures don't rule out?
5. EXP-SEQFULL01's negative `gap_recovery` (the learned sequential selector
   is worse than B0's own simple retriever, not just short of the oracle) is
   a stronger negative result than EXP-FRR01's (which stayed close to B0).
   Does a discrete, high-cardinality (8449-way), sequential classification
   objective have a known failure mode (e.g. an effective label space too
   large relative to ~8449 training queries) that would explain this
   specifically, independent of the set-conditioning idea itself?
6. Given this, should any further Set-Oracle-imitation work require adding
   inference-observable information to the query/candidate representations
   (as EXP-FRR01 tried, also without success) before another objective/
   procedure variant is tried on the current representation, or does this
   three-way failure suggest the representation itself needs to change
   before the objective does?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.

---

# EXP-SEQDIAG01 — Cross-dataset frozen-encoder collapse diagnostic (COMPLETE)

## Research Question

EXP-SEQFULL01 trained cleanly but its encoder's effective rank collapsed
(20.8 → 5.43 on ETTh1) alongside the generalization failure. Is the collapse
a *cause* of the failure (H1), or is the failure about the discrete
set-construction rule not transferring to held-out queries regardless of
representation quality (H2)? A Frozen-B0 control arm (B0's own encoder
weights, never updated; only the new `SetConditioner` trains) isolates this,
run on both ETTh1 and Weather H96 to also check dataset-dependence.

## Method

Identical protocol to EXP-SEQFULL01 (teacher-forced, full-memory, step-wise
cross-entropy imitation of the same Weighted Set Oracle teacher; Stage-2
injection via `set_forced_selection`, unchanged), with a `--frozen_encoder`
flag: B0's own Stage-1 encoder weights are loaded (not randomly
initialised) and excluded from the optimizer; only `SetConditioner` +
`EmptySetToken` train. Four arms: ETTh1 Trainable (reused from
EXP-SEQFULL01, not rerun), ETTh1 Frozen-B0, Weather Trainable, Weather
Frozen-B0. Seed 0, `top_k=10`, H96 only, both `relation_top_n=1` (self-only).

## Results — [repo]

`gap_recovery` = `(A_B0 − A_seq)/(A_B0 − A_set_oracle)`, same definition as
EXP-SEQFULL01; negative = worse than B0, 0 = matches B0, 1 = matches the set
oracle. Comparable across datasets; raw MSE is not.

| Arm | Dataset | b0_unforced_mse | sequential_forced_mse | gap_recovery | seq_set_recall@10 |
|---|---|---:|---:|---:|---:|
| Trainable (reused) | ETTh1 | 0.37312 | 0.40277 | −0.404 | 0.01354 |
| **Frozen-B0** | ETTh1 | 0.373122 | 0.387924 | **−0.248** | 0.01242 |
| Trainable | Weather | 0.175529 | 0.268140 | −0.816 | 0.00917 |
| **Frozen-B0** | Weather | 0.175529 | 0.217331 | **−0.608** | 0.02293 |

`duplicate_rate` / `invalid_rate` = 0.0 on all four arms.

Effective-rank diagnostic (post-hoc, `embedding_geometry`, full candidate
bank, channel 0):

| Encoder | Dataset | effective_rank (of 128) | mean pairwise cosine |
|---|---|---:|---:|
| B0 | ETTh1 | ~20.8 (EXP-3 figure, not recomputed) | UNKNOWN |
| Trainable | ETTh1 | 5.433 | 0.0122 |
| B0 | Weather | 19.576 | 0.8123 |
| Trainable | Weather | 4.297 | 0.7063 |

Weather's trained encoder loses ~78% of its effective rank (19.576→4.297);
ETTh1's loses ~74% (20.8→5.433) — consistent relative-magnitude collapse
across both datasets. Mean pairwise cosine, notably, does *not* move the
same direction on Weather (B0's own cosine there is already high, 0.812; the
trained arm's, 0.706, is *lower*) — cosine is not a reliable cross-dataset
collapse indicator here; effective rank is.

**HardAggregateMSE@10** (`= MSE(mean_i y_i, y_q)`, equal-weight aggregate,
distinct from the B0-weighted `A_weighted` above; the production eval script
only reports the latter, so this was computed post-hoc):

| Arm | Dataset | sequential | B0 (unforced) | seq / b0 |
|---|---|---:|---:|---:|
| Trainable (reused) | ETTh1 | 0.59017 | 0.40741 | 1.449x |
| **Frozen-B0** | ETTh1 | **0.51901** | 0.40741 | **1.274x** |
| Trainable | Weather | **4.30396** | 0.23421 | **18.38x** |
| **Frozen-B0** | Weather | **0.92057** | 0.23421 | **3.93x** |

Same direction as `gap_recovery` (freezing helps on both datasets) but a
much larger and markedly asymmetric magnitude: freezing cuts Weather's
unweighted aggregate error 4.7x (4.304→0.921) versus only 1.14x on ETTh1
(0.590→0.519). Trainable Weather's unweighted error (4.30) is over 18x B0's
— a far starker failure than the weighted `A_weighted` number suggested,
visible only once the B0-derived weighting is removed. No arm beats its own
B0 under this metric either.

**Structural invariants, independently re-verified beyond training-time
logging:** encoder-freeze checked by SHA256 of every `encoder.*` tensor —
Arm B's and Arm D's checkpoints are byte-identical to their source B0
checkpoints; Arm C's (trainable) differs, as expected. Full-memory checked
by code inspection of `step_logits()` — it scores the complete
`candidate_embeddings` tensor (8449/36696) at every step; no shortlist stage
exists anywhere in the training/eval scripts.

## Sanity checks passed

`pytest tests/`: 466 passed (8 new, in `tests/test_exp_seqdiag01.py`),
exactly the same 2 pre-existing repository failures, no regression.
`b0_unforced_final_mse` independently re-derived and fingerprint-matched by
every arm's own evaluation run (ETTh1: 0.373122 vs. recorded 0.37312;
Weather: 0.175529 reproduced identically by both Weather arms). A server
reboot mid-campaign killed Arm D at 2/10 Stage-1 epochs; it was discarded
(not resumed — no epoch-resume logic exists) and rerun from scratch, all 10
epochs, after reboot. Full detail: `results/EXP-SEQDIAG01/notes.md`,
`logs/exp_seqdiag01/RESUME_STATE.md`.

## Conclusion (stated within what the data supports)

Freezing the encoder moved `gap_recovery` toward zero on **both** datasets
(ETTh1: −0.404→−0.248, Δ+0.156; Weather: −0.816→−0.608, Δ+0.208) — a
consistent, same-direction, cross-dataset effect, corroborated by a second,
independent line of evidence (relative effective-rank collapse magnitude
matching within ~4 points of each other across the two datasets). This is
**partial support for H1**: representation collapse is a genuine,
consistent contributing cause of EXP-SEQFULL01's generalization failure, not
merely a correlate specific to ETTh1.

It is not **full** support. Every one of the four arms — including both
Frozen-B0 controls, where the representation is held fixed at B0's own,
uncollapsed quality — remains net `gap_recovery`-negative: even with
representation quality controlled for, the sequential selector's forced
Top-10 is still worse than B0's own unforced selection on both datasets.
**H2 (the discrete greedy set-construction rule does not transfer from
training queries to held-out ones from past-only `X_q`/`X_i` information,
independent of representation quality) remains necessary** to explain this
residual failure. The honest reading is: **collapse is a real contributing
factor, but not the whole explanation** — this is not a clean Case A (H1
fully vindicates, mechanism otherwise sound) nor a clean Case B (H1 plays no
role); it sits between them, with H1 explaining part of the gap and H2
explaining the rest.

HardAggregateMSE@10 (equal-weight, computed post-hoc) sharpens this: on
Weather, freezing cuts the sequential arm's unweighted aggregate error by
4.7x (4.304→0.921) — a far larger effect than `gap_recovery` alone showed —
while on ETTh1 the same metric moves only 1.14x (0.590→0.519). Both datasets
still agree on *direction*; the *size* of the freeze benefit is not
comparable across datasets, and Weather's Trainable arm in particular is
catastrophically bad under the unweighted metric (18x worse than B0) despite
looking only ~2x worse under the B0-weighted `A_weighted` metric — the
weighting in `A_weighted` (derived from B0's own retrieval score) appears to
substantially mask how poor the Trainable arm's raw candidate selection is
on Weather specifically.

**Clarification on what "Frozen" means here (stated explicitly per this
project's protocol):** the Frozen-B0 arm is **not** a different Stage-1/
Stage-2 architecture. Every arm in this experiment, Trainable and Frozen
alike, already separates Stage-1 (produces a hard Top-10 via
`set_forced_selection`) from Stage-2 (B0's own unchanged forecaster/gate/
fusion/aggregation; the forecasting loss is never backpropagated into
Stage-1). "Frozen" refers only to whether the Stage-1 encoder's own weights
update during Stage-1 training — an encoder-representation control *within*
the existing separated two-stage design, not a new end-to-end or
joint-training configuration. True Stage-1+Stage-2 joint end-to-end training
(Stage-2 loss backpropagated into Stage-1) was not implemented and remains
out of scope.

## What this does NOT establish

- That freezing the encoder, or any other collapse-prevention mechanism, is
  sufficient on its own to reach competitive Stage-2 performance — it is
  not, on either dataset tested.
- Anything about horizons other than H96, or about `relation_top_n>1`
  (cross-channel) configurations — not run, per the pre-registered scope.
- The exact quantitative split between "how much of the gap is H1 vs. H2" —
  only the qualitative direction (both contribute) is supported by this
  design; a cleaner decomposition would need a different experiment.
- Why Weather's `gap_recovery` is uniformly worse than ETTh1's on both
  Trainable and Frozen arms (−0.816 vs. −0.404 Trainable; −0.608 vs. −0.248
  Frozen) — the effective-rank collapse magnitude is *similar* across
  datasets, so this cross-dataset gap in absolute severity is not explained
  by collapse alone; not investigated further here.

## Questions for ChatGPT

7. Given collapse contributes but does not fully explain the failure on
   either dataset (H1 partial, H2 still necessary), what is the most
   informative next experiment to isolate H2's remaining contribution —
   i.e., to test whether the discrete greedy rule itself is learnable from
   past-only information, independent of both representation collapse and
   the specific conditioner architecture tried so far?
8. Effective rank (not mean pairwise cosine) is the metric that agrees in
   direction across ETTh1 and Weather; cosine moves the "wrong" way on
   Weather (trained arm's cosine is *lower* than B0's, despite lower
   effective rank). Is this a known dissociation between these two collapse
   metrics, and does it change how much weight effective rank alone should
   get as *the* collapse diagnostic going forward, versus needing a second
   corroborating metric on every future encoder-quality claim?
9. Weather's `gap_recovery` is substantially worse than ETTh1's at both
   Trainable and Frozen settings, despite comparable relative effective-rank
   collapse. Is this consistent with a channel-count / candidate-pool-size
   confound (Weather: 21 channels, 36696 candidates/channel vs. ETTh1: 7
   channels, 8449 candidates/channel — a ~4.3x larger, higher-cardinality
   problem for the same sequential classification objective), or does it
   suggest something dataset-specific unrelated to scale?
10. `A_weighted` (B0-weighted) and `HardAggregateMSE@10` (equal-weight) agree
    on direction but disagree sharply on magnitude for Weather's Trainable
    arm (~2x worse than B0 vs. 18x worse than B0, respectively). Given
    `A_weighted` is the metric `gap_recovery` and this project's other
    verdicts are built on, should `HardAggregateMSE@10` (or another
    unweighted view) be reported as a standard companion metric going
    forward, on the reasoning that a B0-derived weighting could obscure how
    bad a candidate set's raw composition is?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.

---

# EXP-MARGUTIL01 — Full-Memory Set-Conditioned Dense Marginal Utility (COMPLETE, 3/4 cells)

## Research Question

D-0011 closed exact one-hot next-candidate-ID imitation. This tests the
named successor: replace the one-hot CE target with a DENSE per-candidate
set-utility regression target, `u_i^(t) = -A_weighted(S*_{t-1} + {i})` for
EVERY valid remaining candidate at every teacher-forced step (not just the
argmin) — does that let a held-out query build a competitive Top-K and
reach Stage-2? Encoder frozen (B0's weights) throughout; EXP-SEQDIAG01
already answered the trainable-encoder question, not reopened here.

## Method

`u_i^(t) = -A_weighted(S*_{t-1}+{i})`, computed via the exact closed-form
incremental-weighted-mean trick `select_greedy_weighted_set` already uses
internally (`utils/dense_utility.py`), chunked over the candidate dimension
for GPU memory (never a shortlist — every valid candidate still scored).
SmoothL1 regression, minimal `SetConditioner` (reused verbatim from
EXP-SEQFULL01) + a new affine `UtilityHead` (`a*cosine+b`, the only new
learnable parameters — no new architecture). Oracle-prefix teacher-forcing
during training; free-running argmax at inference. Stage-2 injection via
`set_forced_selection`, B0's weighting/gate/fusion/base-forecaster
unchanged. 4 cells approved (ETTh1/Weather × H96/H720); **Weather H720 was
cancelled by the user (D-0013) after ~2h without completing epoch 1** —
reported here as 3 cells, no number fabricated for the 4th.

## Results — [repo]

| Cell | B0 unforced | Dense forced | Delta vs B0 | gap_recovery | Utility Spearman | Utility Pearson | HardAgg (seq/b0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | **+0.157** | -1.832 | 0.543 | 0.424 | 1.657 / 0.407 |
| Weather H96 | 0.17553 | 0.33253 | **+0.157** | -1.744 | 0.658 | 0.503 | 4.465 / 0.234 |
| ETTh1 H720 | 0.46827 | 0.49269 | **+0.024** | -1.866 | 0.383 | 0.413 | 1.961 / 0.572 |

`duplicate_rate`/`invalid_rate` = 0.0 on all 3 cells. Every cell's Stage-2
MSE is worse than B0 by more than the pre-registered 0.01 noise threshold —
**0 of 3 cells show meaningful improvement.** This is a *larger* regression
than EXP-SEQDIAG01's one-hot Frozen-B0 arm on ETTh1 (+0.0148) — the dense
target is not an improvement over the one-hot predecessor it replaced, on
any cell tested.

## Sanity checks passed

`pytest tests/`: 466→477→485 passed across this campaign and its
follow-up, same 2 pre-existing failures throughout, no regression. A
1-epoch GPU smoke run caught and fixed two bugs before the full run:
`HardAggregateMSE@10` off by exactly the horizon factor `H`, and step-wise
`regret` computed against the oracle's own (tautologically zero) pick
instead of the model's own predicted pick. Both caught by comparing against
EXP-SEQDIAG01's already-verified figures before proceeding.

## Conclusion (stated within what the data supports)

Utility Spearman (0.38-0.66) shows the dense target IS partially learnable
from a frozen representation — the prior objective's *sparsity* was not the
whole story (H1's learnability premise is not rejected). But
`gap_recovery`/`HardAggregateMSE@10`/Stage-2 show this learned utility does
not translate into a competitive free-running Top-K selection on any cell —
H2 (information/translation ceiling from learned-utility to good-selection)
is the dominant finding on the 3 cells run. The immediate follow-up
(EXP-FIRSTANCHOR-DIAG, below) decomposes why: step-wise regret concentrated
almost entirely at t=1 pointed at the first candidate choice as a candidate
explanation, tested directly.

## What this does NOT establish

- Weather H720 — no result exists; not extrapolated from the other 3 cells.
- Whether a *weighted* SmoothL1 loss (explicitly deferred per the spec, "첫
  pilot에서는 별도 utility weighting을 넣지 않는 것을 기본으로 한다") would
  change the outcome — not tested.
- Anything about `relation_top_n>1` (cross-channel) — not run.

## Questions for ChatGPT

11. Utility correlation (0.38-0.66) is clearly above zero but Stage-2 is
    uniformly worse than the one-hot predecessor it replaced. Is there a
    known failure mode where a moderately-correlated dense regression
    target produces WORSE downstream selection than a sparse but exact
    one-hot target, e.g. because regression errors compound differently
    across 10 sequential argmax steps than classification errors do?
12. Given H1's premise (learnability) held but the translation to
    selection failed, should the next experiment target the
    utility-to-selection translation step directly (e.g. a listwise/rank
    loss instead of pointwise regression) rather than another supervision
    format for the same pointwise target?

---

# EXP-FIRSTANCHOR-DIAG — causal decomposition of EXP-MARGUTIL01's t=1 choice (COMPLETE, 3/3 available cells)

## Research Question

EXP-MARGUTIL01's regret was overwhelmingly concentrated at step t=1. Is the
first candidate's failure the primary cause of the whole sequence's
downstream/Stage-2 failure (H1), or does the selector fail on its own
merits at t=2..10 even given a good t=1 anchor (H2)? No new training —
reuses each EXP-MARGUTIL01 checkpoint exactly; the only variable is which
rule picks candidate 1.

## Method

Four arms: **Dense-first** (EXP-MARGUTIL01's own free-running result,
reused), **B0-first** (t=1 = B0's own production retrieval-score argmax),
**Oracle-first** (t=1 = argmin singleton future MSE, diagnostic-only, uses
`Y_q`), **B0 baseline**. t=2..10 identical, unmodified, free-running Dense
selector in every arm — no teacher forcing or oracle/B0 injection past t=1,
verified by unit test (forcing t=1 to the value an arm would have picked
anyway reproduces the byte-identical trajectory). Ran on GPU 0 in parallel
with EXP-MARGUTIL01's (then-active) Weather H720 on GPU 1, per explicit
user instruction (D-0014, a one-time exception to this project's
single-GPU convention). Only the 3 cells with a saved EXP-MARGUTIL01
checkpoint could be run (Weather H720 was cancelled before completing one).

## Results — [repo]

Stage-2 MSE and Recovery-to-B0 (`(MSE_dense-MSE_arm)/(MSE_dense-MSE_B0)`; 0=no
recovery, 1=fully recovers to B0, >1=beats B0):

| Cell | B0 | Dense-first | B0-first (Recovery) | Oracle-first (Recovery) |
|---|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | 0.43661 (**59.5%**) | 0.21717 (**199.5%**) |
| Weather H96 | 0.17553 | 0.33253 | 0.21261 (**76.4%**) | 0.41411 (**-52.0%**) |
| ETTh1 H720 | 0.46827 | 0.49269 | 0.51995 (**-111.6%**) | 0.43635 (**230.7%**) |

**No single clean pattern.** B0-first: strong recovery on both H96 cells,
but *worsens* ETTh1 H720. Oracle-first: beats B0 outright on both ETTh1
cells, but *worsens* Weather H96.

**What IS consistent:** the stepwise A_weighted(S_t) trajectory barely
moves after t=1 on every cell/arm (e.g. ETTh1 H96 oracle-first: 0.19101 at
t=1 → 0.19665 at t=10, a 3% relative move over 9 further steps) — **t=1
dominates the final outcome on every cell.** The cross-cell disagreement is
about which t=1 rule is good, not about whether t=1 dominates.

t=1 candidate quality (full test set): B0's own singleton pick beats the
Dense model's own t=1 pick 64-70% of the time on every cell. t=1 rank
diagnostics (200-query subsample, full candidate population per query): the
oracle-best candidate's median predicted rank is 489-1083 (out of the full
memory bank); Top-1 hit rate 0% everywhere measured;
`spearman_within_teacher_top1pct` (0.04-0.14) is markedly lower than
EXP-MARGUTIL01's *global* teacher-forced Spearman (0.38-0.66) — the model's
ranking degrades sharply in exactly the top tail that t=1 selection
actually depends on.

## Sanity checks passed

`pytest tests/`: 485 passed (8 new), same 2 pre-existing failures, no
regression. `identity_check_singleton_oracle_eq_teacher_first: true` on
every cell (Oracle-first's t=1 exactly reproduces the cached greedy
oracle's own first pick, per query). `duplicate_rate`/`invalid_rate` = 0.0
everywhere.

## Conclusion (stated within what the data supports)

t=1 is confirmed as the dominant determinant of every cell's outcome — a
consistent, load-bearing finding. But **no single first-candidate rule is
uniformly sufficient**: the deployable B0-first hybrid helps substantially
on 2 of 3 cells (both H96) and hurts on the third (ETTh1 H720); the
non-deployable Oracle-first "ceiling" helps decisively on 2 of 3 cells
(both ETTh1) and hurts markedly on the third (Weather H96). This is neither
a clean "t=1 is everything, any reasonable anchor fixes it" (ruled out by
the ETTh1 H720 B0-first regression and the Weather H96 Oracle-first
regression) nor a clean "t=1 doesn't matter" (ruled out by the
trajectory-flatness finding on every cell). The honest reading: **t=1
quality is necessary everywhere, but which concrete rule delivers good t=1
quality is dataset/horizon-dependent** — not settled by this diagnostic
alone.

## What this does NOT establish

- Weather H720's FIRSTANCHOR result — no EXP-MARGUTIL01 checkpoint exists
  to diagnose.
- WHY B0-first regresses specifically on ETTh1 H720, or why Oracle-first
  regresses specifically on Weather H96 — not investigated further here
  (would need per-query case analysis of the divergent cell(s)).
- Any claim that "B0-first anchor + Dense set-conditioning" is a validated
  method — 1 of 3 cells contradicts it; this project's workflow reserves
  that call for the reviewer, not Claude Code.

## Questions for ChatGPT

13. B0-first recovers well on H96 (both datasets) but regresses on ETTh1
    H720; Oracle-first recovers well on ETTh1 (both horizons) but regresses
    on Weather H96. Is there a principled explanation for this
    cross-cutting (not simply per-dataset or per-horizon) disagreement
    pattern, or does it suggest the "good anchor" property itself is not a
    single scalar the way singleton MSE treats it?
14. Given t=1 dominates every cell's aggregate but no single t=1 rule is
    uniformly good, is the more promising next direction (a) a
    learned/adaptive first-candidate selector conditioned on
    dataset/horizon signals, (b) investigating why the Dense
    set-conditioned steps t=2..10 apparently cannot correct for a bad t=1
    (the "barely moves" trajectory finding) even when they clearly have the
    numerical capacity to move the aggregate a lot (Dense-first's own
    A_weighted values are far from t=1's, just not toward B0/oracle), or
    (c) something else this diagnostic's design cannot distinguish?
15. Is the top-tail-specific ranking failure (`spearman_within_teacher_top1pct`
    0.04-0.14 vs. global Spearman 0.38-0.66) itself informative about what
    kind of training signal to try next — e.g. does it argue for a loss
    that specifically weights the top of the ranking (listwise/NDCG-style)
    over the pointwise SmoothL1 regression EXP-MARGUTIL01 used?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.

---

# EXP-CONTINUATION-DIAG — exhaustive t=2 continuation diagnostic (COMPLETE, 3/3 available cells)

## Research Question

EXP-FIRSTANCHOR-DIAG showed fixing t=1 does not uniformly help. This asks
directly, at t=2, exhaustively over EVERY valid remaining candidate: does
the Dense selector's own t=2 pick land near the true best continuation
(`i2_oracle = argmin_i A(S1+{i})`, exhaustive, not sampled), and is any
failure a global-ranking problem or specific to the extreme top tail? No
new training — reuses `run_arm`/`a_weighted_prefix` from
EXP-FIRSTANCHOR-DIAG verbatim (via import) for i1/i2_dense/`A(S)`, and
`dense_utility` from EXP-MARGUTIL01's own shared math for the exhaustive
`A(S1+{i})` sweep.

## Method

For each of 3 first-anchor policies (`dense_first`/`b0_first`/`oracle_first`,
identical definitions to EXP-FIRSTANCHOR-DIAG) × 3 cells (ETTh1 H96,
Weather H96, ETTh1 H720 — the cells with a saved EXP-MARGUTIL01 checkpoint),
500 queries: `A1=A({i1})`; `i2_dense` via the exact `run_arm(...,k=2,...)`
call that produces EXP-FIRSTANCHOR-DIAG's own trajectory; `i2_oracle` via
exhaustive argmin over every valid remaining candidate (chunked, never
shortlisted). Reports `continuation_regret=A2_dense-A2_oracle`, the rank of
each side's pick in the other's ordering, and Spearman correlation between
Dense's predicted utility and the true continuation gain — globally and
within the true top 10%/5%/1%/top-50/top-10 tail specifically.

## Results — [repo]

| Dataset | H | Anchor | A1 | Dense A2 | Oracle A2 | Cont. Regret | Dense hurt % | Oracle improvable % | Dense hurts & Oracle improves % | Dense true rank median | Oracle pred rank median | Global ρ | Top10% ρ | Top1% ρ |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ETTh1 | 96 | dense_first | 1.056 | 1.067 | 0.228 | 0.839 | 37.6 | 100.0 | 37.6 | 5244 | 8372 | -0.195 | -0.558 | -0.257 |
| ETTh1 | 96 | b0_first | 0.569 | 0.569 | 0.279 | 0.289 | 23.2 | 100.0 | 23.2 | 6866 | 8395 | -0.414 | -0.629 | -0.323 |
| ETTh1 | 96 | oracle_first | 0.223 | 0.223 | 0.164 | 0.059 | **48.2** | 100.0 | **48.2** | 3631 | 8279 | 0.156 | -0.525 | -0.347 |
| Weather | 96 | dense_first | 0.849 | 0.781 | 0.038 | 0.743 | 41.6 | 100.0 | 41.6 | 20536 | 30506 | -0.163 | -0.226 | -0.061 |
| Weather | 96 | b0_first | 0.723 | 0.716 | 0.178 | 0.538 | 39.2 | 93.2 | 32.4 | 24987 | 35710 | -0.236 | -0.399 | -0.171 |
| Weather | 96 | oracle_first | 0.030 | 0.035 | 0.021 | 0.014 | **79.6** | 89.6 | **69.4** | 3155 | 20899 | 0.457 | -0.044 | -0.183 |
| ETTh1 | 720 | dense_first | 1.228 | 1.211 | 0.398 | 0.813 | 20.0 | 100.0 | 20.0 | 5286 | 6951 | -0.268 | -0.392 | -0.145 |
| ETTh1 | 720 | b0_first | 0.812 | 0.812 | 0.454 | 0.358 | 12.0 | 100.0 | 12.0 | 6388 | 7092 | -0.574 | -0.546 | -0.152 |
| ETTh1 | 720 | oracle_first | 0.409 | 0.410 | 0.294 | 0.116 | 20.0 | 100.0 | 20.0 | 4334 | 6809 | -0.091 | -0.417 | -0.151 |

**Top1% ρ is negative in all 9 of 9 combinations** — the single most
consistent finding of this diagnostic. Global ρ is mixed (positive in 2/9,
both `oracle_first`). Weather H96 `oracle_first` is a distinct catastrophic
case: `ratio_dense` (A2_dense/A1) reaches p99=721M, max=2.1B;
`corr(alpha2_dense,delta_dense)=+0.659` there specifically (mis-selected
candidates get disproportionate aggregation weight) — the other 8 combos
show this correlation weak-to-moderately NEGATIVE, so this is a
cell-specific amplifying mechanism, not a general explanation.

## Sanity checks passed

`pytest tests/`: 490 passed (5 new), same 2 pre-existing failures, no
regression. Two independent code paths for `A(S1+{i2})` (prefix-softmax vs.
the incremental closed form) agree to within 1.9e-6 across all 9 combos.
Full 8-item sanity checklist (candidate exclusion, exhaustive-argmin
verification against brute force, oracle-first identity, sign convention,
etc.) verified and logged: `results/EXP-CONTINUATION-DIAG/REPORT.md`.

## Conclusion (stated within what the data supports) — Q1-Q7 answered with numbers in full at `results/EXP-CONTINUATION-DIAG/REPORT.md`

**Q1 (is Dense's t=2 pick good?)** No — median true rank 3155-24987 out of
8449-36696 candidates, ~0% Top-1/5/10/50 hit rate, in every combo. **Q2
(does a better continuation exist when Dense hurts?)** Yes, in 83-100% of
hurting queries across every combo. **Q3 (global vs. top-tail
correlation)** Confirmed dissociated: top-1% ρ negative everywhere, global
ρ mixed. **Q4 (does a good anchor fix it?)** No — `oracle_first` has the
HIGHEST hurt-rate on 2/3 cells (ETTh1 H96: 48.2% vs. dense_first's 37.6%;
Weather H96: 79.6%, the worst result in the table). **Q5 (Weather H96:
many-small or few-catastrophic?)** Few-catastrophic for `oracle_first`
(p99/max ratios in the hundreds of millions/billions; median near 1). **Q6
(does the bad candidate get high aggregation weight?)** Yes, but only for
Weather H96 `oracle_first` specifically (+0.659 correlation there; weak
negative in the other 8 combos) — a real but cell-specific mechanism, not
general. **Q7 (A/B/C/D/E judgement)**: most defensible call is **E
(multiple causes)**, with **B (set-conditioned extreme top-tail ranking
failure)** as the single best-evidenced, most consistent component (every
combo), and **C (weight mismatch)** confirmed as a real cell-specific
amplifier in the one documented catastrophic case. **D (greedy
construction's own limits)** is explicitly not testable from this
diagnostic's design and is not claimed either way.

Central hypothesis — "Dense selector learns global utility structure but
fails at set-conditioned extreme top-tail ranking" — **is supported**, and
is the best-evidenced single finding of this experiment (9/9 combos agree
on sign). The secondary claim — "a good first anchor should let a genuinely
better continuation be found, but Dense misses it" — **is also supported**,
and notably the failure rate is HIGHER, not lower, under the good
(oracle) anchor on 2 of 3 cells, which was not predicted going in.

## What this does NOT establish

- Weather H720's t=2 behavior — no checkpoint exists to diagnose.
- WHY the top-tail ranking specifically fails (architecture limitation of
  the minimal `SetConditioner`+affine `UtilityHead`? training-time
  objective mismatch — pointwise SmoothL1 vs. what greedy selection needs?
  something else?) — this diagnostic locates the failure precisely but does
  not test a fix.
- Whether a listwise/rank-aware training objective would resolve it — named
  as a candidate question (14/15 above) but not implemented or tested here.
- Anything about greedy Top-K construction's own fundamental limits (Q7's
  option D) — this diagnostic's design (fix t=1, exhaustively evaluate t=2)
  cannot speak to whether a non-greedy construction would do better.

## Questions for ChatGPT

16. Top-1% ρ is negative in all 9 combos while global ρ is occasionally
    positive. Is there a known reason a pointwise regression target
    (EXP-MARGUTIL01's `SmoothL1(u_hat, u_norm)`, normalised per-query/step)
    would systematically fail to preserve extreme-tail ranking even when it
    captures bulk/global structure reasonably? Does this argue specifically
    for a listwise or top-k-focused loss (e.g. ListMLE, LambdaRank-style,
    or a margin loss restricted to the true top percentile) as the next
    experiment, over another pointwise-target variant?
17. The oracle-anchor condition making t=2 failure WORSE (not better) on 2
    of 3 cells was not predicted by the pre-experiment hypotheses. Is there
    a plausible mechanism (e.g. the model's `SetConditioner` state `m_{t-1}`
    for an unusually strong/atypical anchor falls outside the distribution
    of anchors it saw during oracle-prefix teacher-forced training, since
    training always conditioned on the ACTUAL greedy-oracle prefix, not on
    "an unusually good singleton pick that the oracle prefix wouldn't
    necessarily reach") that would explain why a better anchor produces a
    worse `m_{t-1}` state for the conditioner?
18. Given B (top-tail ranking) is well-evidenced and C (weight mismatch) is
    confirmed but cell-specific, is the recommended next step (a) a
    listwise/top-focused training objective (addresses B directly), (b) an
    aggregation-weight regularisation or cap (addresses C), (c) both
    together, or (d) a smaller, cheaper follow-up diagnostic isolating
    WHICH of B or C dominates before committing to either fix?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.

---

# EXP-TOPTAIL-RANK01 — loss-formulation comparison for the Dense selector (IN PROGRESS: ETTh1 H96 complete, Weather H96 R2 training)

## Research Question

EXP-CONTINUATION-DIAG's question 16 asked whether EXP-MARGUTIL01's
pointwise SmoothL1 regression target itself was the bottleneck for
top-tail ranking. This tests three loss formulations, EVERYTHING else held
identical (same frozen B0 encoder, `SetConditioner`, `EmptySetToken`,
`UtilityHead = a*cosine+b`, oracle-prefix teacher forcing, full-memory
candidate support, checkpoint-selection criterion, evaluation protocol —
reusing EXP-MARGUTIL01's and EXP-CONTINUATION-DIAG's own code, not
reimplemented):

- **R0** = SmoothL1 pointwise regression (EXP-MARGUTIL01's own arm, reused).
- **R1** = top-tail logistic pairwise ranking loss only (true top-1%
  positives vs. predicted-high hard negatives).
- **R2** = R1 + R0 combined (`L = L_pairwise + 1.0 * L_smoothl1`), testing
  whether pointwise supervision stabilises/improves the pairwise objective.

`trainable_params=49794` verified identical across all three arms (same
`SetConditioner`+`EmptySetToken`+`UtilityHead` classes, no capacity
change) — logged at the start of every training run.

## Method

New file `scripts/train_toptail_rank01.py`, importing (not reimplementing)
`build_experiment`/`encode`/`memory_value`/`run_sequence_dense` from
`scripts/train_margutil01.py`. Pairwise loss: fully vectorised (batched
`topk`/`scatter`/`gather`, no per-row Python loop) — an early per-row-loop
implementation was replaced after a real-data sanity run showed it was
prohibitively slow (>>300s/epoch and not converging within a reasonable
window); the vectorised rewrite reproduces the identical mathematical
definition (verified by the same 6-test suite,
`tests/test_exp_toptail_rank01.py`) at ETTh1-H96 epoch times of
~200-500s, comparable to the per-row version's own isolated sanity timing.
Evaluation reuses `scripts/eval_margutil01_stage2.py` (Stage-2, teacher-forced
utility), `scripts/eval_firstanchor_diag.py` (t=1 rank/top-tail
diagnostics), and `scripts/eval_continuation_diag.py` (exhaustive t=2
continuation) completely unmodified — only the `--sequential_checkpoint`
argument points at the new R1/R2 checkpoints.

**Weather H96 R1 was stopped mid-training by explicit user decision**
(mirroring the same kind of scope-reduction as EXP-MARGUTIL01's Weather
H720/D-0013) after 2 epochs, both showing the loss plateaued near
`ln(2)≈0.693` (the value a pairwise logistic loss takes when it cannot
distinguish positives from negatives) — no checkpoint was ever promoted
past epoch 1 quality, and Weather H96's comparison will be R0 vs. R2 only,
not R1. **R2 training on GPU 1 ran literally in parallel with R1's Weather
H96 training** (same physical GPU, two processes, per explicit user
instruction) for part of this campaign — both trained correctly and
independently under this sharing (verified: no cross-contamination
possible, separate model instances/optimizers/processes), only wall-clock
time was affected (epochs took ~5-7x longer while sharing than isolated).

## Results — ETTh1 H96 (complete, 3/3 arms) — [repo]

### Stage-2

| Arm | Stage-2 MSE | Delta vs B0 | gap_recovery | HardAgg (seq) | Global Spearman (teacher-forced) |
|---|---:|---:|---:|---:|---:|
| B0 | 0.37312 | — | — | 0.407 | — |
| R0 (SmoothL1) | 0.52991 | +0.157 | -1.832 | 1.657 | +0.543 |
| R1 (pairwise) | 0.39978 | +0.027 | -0.303 | 0.600 | -0.162 |
| R2 (hybrid) | **0.39526** | **+0.022** | **-0.263** | **0.551** | -0.103 |

### t=1 (FIRSTANCHOR-DIAG diagnostics, reused)

| Arm | Spearman within true top-1% | Oracle rank median | Top-50 containment | Top-1 hit |
|---|---:|---:|---:|---:|
| R0 | -0.558 | 489 | 11.5% | 0% |
| R1 | +0.163 | 387 | 20.0% | 0% |
| R2 | **+0.213** | **99** | **30.5%** | **0.5%** |

### t=2 (CONTINUATION-DIAG exhaustive, `dense_first` anchor, reused)

| Arm | hurt_frac | true rank median | oracle predicted rank median | Spearman top-1% | Spearman top-10% | regret |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 37.6% | 5244 | 8372 | -0.257 | -0.558 | 0.839 |
| R1 | 24.6% | 2305 | **502.5** | **+0.155** | **+0.305** | 0.479 |
| R2 | **23.2%** | **1934** | 560 | 0.140 | 0.288 | **0.350** |

**R0→R1→R2 is a monotonic improvement on Stage-2, `gap_recovery`,
HardAggregate, and every t=1 metric — a clean, consistent staircase.** t=2
is the one place R2 is NOT uniformly better than R1: R2 wins on
`hurt_frac`/true-rank/regret but R1 has a very slightly better
oracle-predicted-rank and top-tail Spearman at t=2 specifically (differences
of a few percentage points/ranks, not large). Every arm's global
teacher-forced Spearman is negative or near-zero for R1/R2 while positive
for R0 — reproducing EXP-TOPTAIL-RANK01's central pattern (pointwise
regression fits the global distribution better, pairwise-family losses fit
the decision-relevant top tail better) at three points along a
SmoothL1-weight spectrum, not just two.

## Training stability (R1 vs. R2, ETTh1 H96)

Both R1 and R2 selected `best_epoch=1` by the (unchanged) `val_overlap@10`
criterion and both show `val_overlap@10` declining after epoch 1 — this
specific symptom is NOT resolved by adding SmoothL1. However, the
CHARACTER of the decline differs: R1's `pairwise_loss` value converges
essentially exactly to `ln(2)=0.693` by epoch 6 (the value indicating the
model can no longer separate positive from negative pairs at all); R2's
stays in the `0.697-0.703` range throughout 6 epochs (never fully
collapsing), and `UtilityHead`'s scale parameter `a` declines gradually
(0.921→0.799 over 6 epochs) rather than collapsing toward 0. The
positive-negative `margin` also improves slightly over R2's own epochs
(-0.0183→-0.0082). **R2 measurably softens, but does not eliminate, the
degradation pattern seen in R1** — best checkpoint is still epoch 1 for
both.

## Sanity checks passed

`pytest tests/`: 496 passed (6 new in `tests/test_exp_toptail_rank01.py`,
covering: known-ordering recovery via gradient descent, positives-from-true-
top-tail-only, hard-negatives-from-predicted-high-excluding-true-positives,
invalid/selected-candidate exclusion with exact zero gradient outside the
valid region, loss ordering (already-correct pairs score lower than
already-wrong pairs), and `UtilityHead` param-count parity), same 2
pre-existing failures, no regression.

## What this does NOT establish yet

- Weather H96's R2 comparison — training in progress, not complete.
- Whether `lambda_rank` values other than 1.0 would change the R1-vs-R2
  tradeoff at t=2 specifically (not swept, per the pre-registered scope).
- Anything about H720, or about an asymmetric/non-cosine scorer (explicitly
  out of scope for this experiment, deferred as a separate follow-up
  question, not run here).

## Questions for ChatGPT

19. R0→R1→R2 is a clean monotonic improvement on Stage-2/`gap_recovery`/t=1,
    but R2 is NOT uniformly better than R1 at t=2 (R1 has marginally better
    oracle-predicted-rank and top-tail Spearman there specifically, while
    R2 wins on hurt-fraction/true-rank/regret). Is there a principled reason
    pointwise auxiliary supervision would help the FIRST selection step more
    than the SECOND (set-conditioned) one, or is this within noise at
    n=500 queries and not a real dissociation?
20. Both R1 and R2 still select `best_epoch=1` by `val_overlap@10` and
    decline afterward, though R2's decline is visibly softer (score scale
    `a` shrinks gradually instead of collapsing, margin improves slightly
    instead of flatlining at `ln(2)`). Is `val_overlap@10` (unordered
    final-Top-10-set overlap with the oracle) simply a noisy/uninformative
    early-stopping criterion for a ranking-loss-trained model, and would a
    different checkpoint-selection metric (e.g. one closer to the t=1/t=2
    top-tail diagnostics themselves) likely change which epoch gets
    selected and the resulting Stage-2 numbers?
21. Given R2 is the best arm on ETTh1 H96 across nearly every metric, is a
    `lambda_rank` sweep (explicitly not run in this pilot) or a longer
    training budget (patience/epochs, also not swept) likely to matter
    more for closing the remaining gap to B0 (R2's Stage-2 MSE is still
    +0.022 worse than B0), versus the t=2-specific set-conditioned ranking
    limitation EXP-CONTINUATION-DIAG originally identified?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`. Weather
H96 results will be appended once R2 training completes.

