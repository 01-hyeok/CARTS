# TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01

**Status: COMPLETE (ETTh1_720, seed0). Verdict: NO-GO per pre-registered
numeric thresholds**, despite K2 achieving the best Uniform Aggregate
MSE@10 AND best Recall@10 of all four arms compared (J0, J1, K1, K2).
K2's individual retMSE@10 degrades 11.16% relative to J1 -- exceeding the
spec's explicit 10% NO-GO ceiling (criterion 6) -- so per the
pre-registered decision rule this is reported as NO-GO, not spun as a
success despite the genuinely positive aggregate-quality result.

## 0. Combined interpretation with TRACK-J3 (spec section 29)

TRACK-J3 (companion diagnostic, run in parallel, CPU-only, see
`research/J-shared-encoder-drift/TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01.md`)
concluded **GO** on the error-complementarity hypothesis: J1's aggregate
advantage over J0 is explained by cross-candidate-error-interaction
reduction (`R_cross=1.071>1`), not diversity for its own sake. Per the
pre-registered combined-interpretation rule: **"J3 supports
complementarity, K2 fails" -> "problem diagnosis correct, but the current
multi-slot surrogate did not solve it."** That is the applicable outcome
here. The set-level misalignment TRACK-J3 identified is real; this
specific multi-slot surrogate recovers a genuine aggregate-quality gain
but at an individual-quality cost outside the pre-registered acceptable
range.

## 1. Final comparison table (test split)

| Arm | Key gradient | Slots | retMSE@10 | D | Cross C | AggMSE@10 (hard) | AggMSE@10 (soft) | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| J0 Joint Shared *(reused)* | Yes | 1 | 0.953599 | 0.095360 | 0.546371 | 0.641731 | -- | 0.020257 |
| J1 StopGrad-Key *(reused)* | No | 1 | 1.008351 | 0.100835 | 0.463512 | 0.564347 | -- | 0.021987 |
| K1 Multi-Slot Relevance | No | 10 | 1.139629 | 0.113963 | 0.459787 | 0.573750 | -- (no agg loss) | 0.020017 |
| **K2 Multi-Slot + Agg** | No | 10 | 1.120915 | 0.112091 | 0.434852 | **0.546943** | 0.504919 | **0.027461** |

`D+C=Agg` holds exactly for K1/K2's hard-selected Top-10 as well (same
identity from TRACK-J3, reused; verified in unit test item 22, never
failed in the actual eval).

## 2. Answers to the pre-registered questions

**Q1** (does J3 explain J1's aggregate gain via cancellation?): Yes --
see TRACK-J3, GO verdict, `R_cross=1.071`.

**Q2** (does multi-slot structure ALONE improve aggregate, no aggregate
loss)?: **No.** K1's aggregate MSE (0.5738) is slightly WORSE than J1's
(0.5643), and K1's retMSE (1.1396) is also worse than J1's (1.0084).
Adding 10 independently-perturbed relevance-only slots, anchored to the
same collective teacher distribution with an overlap penalty, does not
by itself recover J1's aggregate advantage -- if anything it costs a
little on both metrics relative to J1.

**Q3** (does adding the differentiable aggregate objective help over
K1)?: **Yes, clearly.** K2 beats K1 on every metric: retMSE (1.1209 vs
1.1396), Agg (0.5469 vs 0.5738), Recall@10 (0.0275 vs 0.0200). The
aggregate objective is doing real work, not just the multi-slot
architecture.

**Q4** (does K2 improve the J0/J1 Pareto frontier)?: **No, by the
pre-registered numeric criteria.** K2's aggregate MSE (0.5469) IS better
than both J0 (0.6417) and J1 (0.5643) -- the best of all four arms. But
K2's retMSE (1.1209) is worse than BOTH J0 (0.9536) and J1 (1.0084), and
the degradation vs J1 specifically (11.16%) exceeds the spec's 10%
NO-GO ceiling. K2 sits outside the target Pareto region (J0's individual
quality + J1's aggregate quality) -- it improves aggregate further at a
degree of individual-quality cost the spec pre-registered as
unacceptable.

**Q5** (did real slot specialization occur)?: **Yes.** Mean Top-1
selected-candidate identity overlap across the 45 slot pairs is low for
both arms (K1: 3.55%, K2: 1.45% -- K2's slots are even MORE
differentiated than K1's), confirming the 10 slots are not collapsing to
picking the same candidate. This rules out NO-GO criterion 3
("10 slots collapse to the same candidate").

**Q6** (did the soft-objective improvement transfer to hard Top-10)?:
**Partially.** K2's soft aggregate MSE (0.5049, the actual training-time
relaxation) is meaningfully better than its own hard aggregate MSE
(0.5469) -- a gap of 0.042. The relaxation does NOT fully transfer:
there is a real, measurable soft-hard gap, though the hard result still
clearly improves over K1 and J1. This is not the "soft improves, hard
doesn't move at all" NO-GO failure mode (criterion 2) -- hard DID improve
-- but the gap shows real headroom the current relaxation is leaving on
the table.

**Q7** (is this computationally practical, no Set-Oracle-level cost)?:
**Yes.** No `[B,S,N,H]` tensor was ever allocated (confirmed by the
`soft_aggregate_loss` implementation's per-slot Top-32 gather design and
a dedicated unit test); peak VRAM was 1.46-1.47 GB for both K1 and K2 --
essentially the same footprint as the single-slot J1 baseline, and both
full 10-epoch (early-stopped at 6) runs completed in under 3 minutes each
(K1: 109.9s, K2: 158.6s). This is indexable (see Section 4).

**Q8** (does this support "learn complementary predictive retrieval
roles instead of imitating an oracle set")?: **Partial support, not
confirmed at production quality.** The direction shows real promise --
K2 achieves the best aggregate MSE AND best Recall@10 of any arm tested,
using no Set Oracle, no combinatorial search, and no future information
at inference time -- but the specific surrogate tested here trades away
more individual-candidate quality than the pre-registered threshold
allows. **Verdict: NO-GO** on this specific pilot's numeric criteria;
**INCONCLUSIVE-but-promising** on the broader research direction.

## 3. Soft-hard relaxation gap detail

| | Hard AggMSE@10 | Soft AggMSE@10 | Gap |
|---|---:|---:|---:|
| K2 | 0.546943 | 0.504919 | 0.042024 |

The gap (soft better than hard by ~7.7% relative) indicates the
`L_soft=32` future-blind subset + tempered softmax relaxation is a
reasonably faithful but not perfect proxy for the actual greedy-unique
hard selection used at inference -- consistent with expectations for any
differentiable relaxation of a hard top-k operation.

## 4. Indexability

`indexable = YES`. Inference requires only: (1) a one-time candidate
embedding cache `K = E(X_memory)` (identical for all 10 slots), (2) a
single query embedding `z_q = E(X_q)`, (3) 10 cheap linear projections
`q_m = Normalize(W_m z_q)`, (4) 10 independent nearest-neighbor lookups
against the SAME cached index. No candidate-specific pairwise scorer, no
per-candidate MLP, no combinatorial search.

## 5. What is established

- The multi-slot architecture alone (K1, no aggregate loss) does not
  recover J1's aggregate advantage -- it costs slightly on both
  individual and aggregate quality relative to J1.
- Adding the differentiable, future-blind-selection aggregate objective
  (K2) produces a clear, consistent improvement over K1 on every metric
  measured, and achieves the best aggregate MSE and Recall@10 of all
  four arms compared in this session's J-track family.
- Real slot specialization occurs (low cross-slot Top-1 overlap), not
  collapse.
- The approach remains fully indexable at inference time and
  computationally practical (no `[B,S,N,H]` allocation, VRAM/wall-clock
  comparable to the single-slot baseline).
- K2's individual-quality cost (11.16% retMSE degradation vs J1)
  exceeds the pre-registered 10% NO-GO ceiling -- this is a real,
  measured trade-off, not an artifact.

## 6. What is NOT established

- Whether a different `lambda_agg`/`beta`/`L_soft` setting could reach
  the target Pareto region (Condition A/B) -- no coefficient sweep was
  run this pilot, per the spec's explicit prohibition.
- Whether the soft-hard gap (0.042) could be reduced by a tighter
  relaxation (larger `L_soft`, different temperature) -- not explored.
- Whether this reproduces on other seeds, horizons, or datasets --
  explicitly out of scope this round.
- Root cause of WHY K2's individual quality degrades this much relative
  to J1 -- only the outcome, not the mechanism, was measured.

## 7. Interpretation-rule compliance

Per spec section 17/29, this NO-GO verdict is scoped to "ETTh1 H720
seed0, this specific pilot's fixed hyperparameters (`lambda_agg=1.0`,
`beta=0.05`, `L_soft=32`)." It does not claim the multi-slot direction is
dead -- the K1-vs-K2 comparison and the aggregate/Recall@10 results
provide real, if partial, support for the broader idea. No coefficient
sweep, additional seed, Weather, or Stage-2 was run, per spec.

## 8. Unit tests

16/16 unit tests pass (`tests/test_k_multislot_predictive_retrieval01.py`),
covering: candidate count/leakage, key-detach zero-gradient (J1-style),
candidate embeddings still move after an optimizer step, slot score
shape `[B,S,N]`, slot-head determinism and near-identity init,
future-blind Top-32 selection, no `[B,S,N,H]` allocation, hard-inference
uniqueness and determinism, K1/K2 structural difference, collective
`p_bar` sums to 1, KL-from-probability correctness, brute-force-matched
soft aggregate computation, checkpoint reload exactness, and the
`Agg=D+C` identity on hard multi-slot selections.

## Artifacts

- Audit: `research/K-multislot-predictive-retrieval/AUDIT.md`
- Report: `research/K-multislot-predictive-retrieval/TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01.md`
- Script: `scripts/train_k_multislot_predictive_retrieval01.py`
- Unit tests: `tests/test_k_multislot_predictive_retrieval01.py`
- Parallel-execution driver: `scripts/run_j3_k_parallel01.sh`
- Raw results: `results/TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01/ETTh1_720/`
  (`K1_multislot_relevance/`, `K2_multislot_aggregate/`,
  `comparison.json`, `exact_commands.txt`, `working_tree.diff`)
- Checkpoints: `checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/`
