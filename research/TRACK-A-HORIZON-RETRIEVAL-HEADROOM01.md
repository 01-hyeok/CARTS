# TRACK-A-HORIZON-RETRIEVAL-HEADROOM01

**Status: ETTh1_720 and Weather_720 complete (full train/val/test). Solar_720
NOT run — see section 15.**

## 1. Research Question

> Does one retrieved Top-K set suffice for the entire forecasting horizon, or
> would horizon-block-specific Top-K sets retrieve meaningfully better
> evidence?

This is an **Oracle upper-bound diagnostic only**. No retriever is trained,
no model weights are loaded, and no result here should be read as an
achievable inference-time number — see section 6.

## 2. Motivation

Every retrieval-augmented forecasting experiment in this project so far
(TRACK-A-FACTORIAL-E2E01, TRACK-A-TF-ORACLE-LEARNABILITY01) retrieves **one**
Top-K candidate set per query and reuses it across the full prediction
horizon. If the historical evidence that best explains the next 96 steps
differs from what best explains steps 337-720, a single retrieved set is
structurally capped, independent of how well any retriever or router is
trained.

## 3. Existing Track-A Assumption

`Exp_Stage1_Relation`/`RelationStage1`/`train_factorial_e2e01.py`'s entire
Top-K selection (Oracle or learned) picks one candidate set per query,
scored against the FULL `pred_len`-length future. That full-horizon
future MSE is exactly what `individual_utility`/`individual_utility_memsafe`
compute today; this experiment reuses that function unmodified, only
slicing its inputs by horizon block, so the Oracle definitions are provably
consistent with everything else in this project (see section 8).

## 4. Oracle Definitions

For query `q`, channel `c`, candidate `i`, reconstructed candidate future
`Y_{i->q,c}` (via `memory_value()`, section 8):

- Global: `D_G(q,c,i) = MSE(Y_{i->q,c,1:720}, Y_{q,c,1:720})`
- Block:  `D_Bk(q,c,i) = MSE(Y_{i->q,c,block_k}, Y_{q,c,block_k})`

Lower is better; ranking is ascending distance (`stable_topk_indices`,
ties broken by candidate index).

## 5. Horizon Partition

Fixed, per spec section 3, Python slice semantics:

| Block | Range | Length |
|---|---|---|
| B1 | `[0:96]`   | 96  |
| B2 | `[96:336]` | 240 |
| B3 | `[336:720]`| 384 |

Verified exactly by `tests/test_horizon_retrieval_headroom01.py::test_t1_block_boundaries_exact`.

## 6. Fairness / Support Control

Global and every Block Oracle share, by construction (same `batch_x`/
`batch_start_idx` inside one loop iteration): the same query, the same
`exp._candidate_mask(batch_start_idx)` call, the same candidate bank, the
same channel loop, the same K=10. `tests/test_horizon_retrieval_headroom01.py`
T4/T5 assert mask identity and that Oracle Top-K never contains an invalid
candidate, on real ETTh1_720 data.

Both Global and Block Oracles see the query's **true future** — this is
symmetric (an upper-bound-vs-upper-bound comparison), so it is a fair
diagnostic even though it is never a deployable retriever.

## 7. Implementation Audit

Code read line-level before implementation, per spec section 1:

- `scripts/train_margutil01.py::memory_value(args, batch_x, memory_y, memory_x_last, c)`
  returns `(memory_value_c, query_offset)`. When
  `args.relation_value_space == 'delta_last'`, `memory_value_c` is candidate
  future MINUS the candidate's own last-history value, and the caller
  reconstructs the query-level absolute future as
  `memory_value_c + query_offset.view(-1,1,1)` where `query_offset =
  batch_x[:, -1, c]` (the QUERY's own last value, not the candidate's).
  This project-wide convention is reused unmodified; this experiment never
  invents its own future-reconstruction logic.
- `scripts.train_factorial_e2e01.individual_utility_memsafe(memory_c,
  offset_c, query_future, chunk_size)` computes
  `u_i = -MSE(memory_c[i] + offset_c, query_future)`, chunked over the
  candidate axis, mean over the LAST axis of whatever tensors are passed in.
  **This experiment's only implementation trick**: calling this SAME,
  unmodified function with `memory_c[:, lo:hi]` and `query_future[:, lo:hi]`
  computes the exact block-restricted Oracle utility, with zero new MSE
  math and the existing memsafe/chunked behavior for free (no `[B,N,720]`
  tensor is ever materialized, at any dataset scale).
- `Exp_Stage1_Relation._candidate_mask(batch_start_idx)` is a pure function
  of `batch_start_idx`; called once per batch and reused by Global and all
  three Block Oracles.
- `stable_topk_indices` (`models/RelationStage1.py`) is reused unmodified
  for every Top-K selection in this experiment.
- `free_running_aggregate_future_mse`/Stage-2 host weighting is **NOT**
  used anywhere in this diagnostic (host-independent primary metric, spec
  section 9) — confirmed by code read, `diag_horizon_retrieval_headroom01.py`
  never imports `HostScorer`.

`--reference_ckpt` supplies dataset/model config args only (`d_model`,
`enc_in`, `data`, `root_path`, candidate-mask settings); its weights are
never loaded, matching the `build_experiment` "args-only" pattern used
throughout this project.

## 8. Unit Tests

`tests/test_horizon_retrieval_headroom01.py`, 11/11 passing:

| Test | Verifies |
|---|---|
| T1 | Block boundaries exactly `[0:96]/[96:336]/[336:720]`, lengths 96/240/384 |
| T2 | Global utility == direct reference MSE (synthetic) |
| T3 | Block utility == direct time-sliced reference MSE (synthetic) |
| T4/T5 | Candidate mask identical across calls; Oracle Top-K never selects an invalid candidate (real ETTh1_720 data) |
| T6 | `stable_topk_indices` tie-breaking is deterministic by candidate index |
| T7 | Uniform-aggregate reconstruction matches a manual toy example |
| T8 | Block-length concatenation sums to exactly 720 |
| T9 | Synthetic short/mid/long-best candidates are selected by the correct block Oracle |
| T10 | Synthetic uniform-relevance case: Global Top-K == every Block Top-K |
| T11 | Budget-matched (`TopU`) masked-mean reconstruction matches manual per-row computation |
| T12 | `memory_value()`'s delta-last reconstruction matches production semantics on real data |

Full repo regression suite: 1038 passed / 2 pre-existing failures (unrelated
to this experiment's code), confirmed before this round.

## 9. Results — Main Summary Table

Primary metric: uniform `(1/K)`-weighted aggregate MSE, host-independent.
Test split.

| Dataset | Global Top10 | Global Top30 | Global Budget-Matched | Block Oracle | Block Gain vs Global10 | B1<->B3 Overlap |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 H720 | 0.29685 | 0.29712 | 0.29608 | **0.22050** | **25.72%** | 2.78% |
| Weather H720 | 0.09876 | 0.09279 | 0.09280 | **0.05983** | **39.42%** | 0.67% |
| Solar H720 | -- | -- | -- | -- | -- | -- |

(n_test_queries: ETTh1_720 = 2161, Weather_720 = 9820.)

## 10. Top-K Overlap Analysis

| Dataset | G<->B1 | G<->B2 | G<->B3 | B1<->B2 | B1<->B3 | B2<->B3 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1_720 | 8.16% | 15.59% | 31.75% | 4.92% | 2.78% | 3.91% |
| Weather_720 | 1.93% | 6.95% | 18.81% | 0.91% | 0.67% | 1.09% |

The Global Top-10 overlaps most with B3 (the block that dominates the
720-length MSE by sheer length: 384/720 = 53%) and least with B1 -- exactly
what a length-weighted-average selection criterion predicts. B1<->B3
overlap is the near-total-disjointness case the research question asked
about: 2.78% (ETTh1) and 0.67% (Weather) -- essentially two different
candidate sets.

**Rank displacement** confirms this is not a Top-K boundary artifact:

| Dataset | rank_B3(B1's Top10) | rank_B1(B3's Top10) | rank_Global(B1's Top10) | rank_Global(B3's Top10) |
|---|---:|---:|---:|---:|
| ETTh1_720 | 1934 | 1582 | 1320 | 186 |
| Weather_720 | 11795 | 9616 | 9661 | 1775 |

B1's best-10 candidates sit at mean rank ~1900 (ETTh1) / ~11800 (Weather)
under the B3 criterion, out of thousands of valid candidates -- nowhere near
"almost Top-K", i.e. genuinely different evidence, not near-ties across a
boundary. B3's Top-10 is comparatively closer to the Global criterion's own
ranking (rank ~186/~1775) than B1's is, again consistent with B3's larger
share of the full-horizon MSE.

## 11. Budget-Control Analysis

Two controls isolate whether the Block Oracle's gain is just "more unique
candidates":

- **Global Top-30** (3x the single-set budget, still one set for the whole
  horizon): 0.29712 (ETTh1) / 0.09279 (Weather) -- barely different from
  Global Top-10, and still far worse than Block Oracle.
- **Global budget-matched** (`TopU`, exactly `|S_B1 u S_B2 u S_B3|` unique
  candidates per query, mean ~28.9 (ETTh1) / ~29.7 (Weather) -- i.e. almost
  the full 30-candidate budget): 0.29608 (ETTh1) / 0.09280 (Weather) --
  again essentially unchanged from Global Top-10.

Paired bootstrap (2000 resamples, query-level, seed=0):
`Global-budget-matched minus Block` = 0.0756 [0.0743, 0.0769] (ETTh1),
0.0330 [0.0326, 0.0334] (Weather) -- both **strictly positive with a tight
CI far from zero**. Giving the Global criterion the SAME candidate budget
the Block Oracle used does not close the gap. The improvement is
attributable to horizon-specific selection, not candidate count (Case A
condition, not Case B).

## 12. Query-Level Benefit Distribution

| Dataset | improved (delta>0) | >1% | >5% | >10% |
|---|---:|---:|---:|---:|
| ETTh1_720 | 100.0% | 100.0% | 100.0% | 99.8% |
| Weather_720 | 100.0% | 100.0% | 100.0% | 100.0% |

Median delta: 0.0722 (ETTh1, p25=0.0557/p75=0.0938/p90=0.1176), 0.0320
(Weather, p25=0.0256/p75=0.0460/p90=0.0683). This is not a benefit
concentrated in a small subset of queries -- essentially every test query
benefits from horizon-specific retrieval by a large margin (>10% relative
for 99.8-100% of queries).

## 13. Block-Wise Regret

`Regret_Bk = MSE(Aggregate(Global Top-10)_Bk, Y_Bk) - MSE(Aggregate(Block-k Top-10)_Bk, Y_Bk)`:

| Dataset | B1 regret | B2 regret | B3 regret |
|---|---:|---:|---:|
| ETTh1_720 | 0.1572 | 0.0874 | 0.0492 |
| Weather_720 | 0.0596 | 0.0487 | 0.0276 |

Regret is largest in B1 (short horizon) and smallest in B3 (long horizon)
in BOTH datasets, in that consistent order -- the Global (length-weighted)
retrieval set is closest to optimal for the block it structurally resembles
most (B3, the largest and most MSE-dominant block) and worst for the block
it resembles least (B1).

## 14. Interpretation

Both completed cells land unambiguously in **Case A**:

- Block Oracle << Global Oracle (25.7% / 39.4% relative improvement)
- Block Oracle < Global budget-matched (bootstrap CI strictly positive)
- B1<->B3 overlap very low (2.8% / 0.67%)

> One retrieved set does not suffice for the full H=720 horizon; short- and
> long-range evidence are substantially different, and the difference is not
> explained by candidate-count budget alone.

## 15. Limitations

- **Solar H720 was NOT run.** Its canonical S0_wce Stage-1 reference
  checkpoint at `seq720_pred720` does not exist -- the same GPU1-capacity
  OOM already reported for `TRACK-A-TF-ORACLE-LEARNABILITY01`'s Solar_720
  host build (`_apply_full_memory_gradient` on 137 channels x pred_len=720
  needs ~69GB alone, confirmed OOM in full isolation on an 80GB card).
  Since this diagnostic never loads checkpoint WEIGHTS (only `args`), a
  workaround exists (reuse Solar_96's saved args with `pred_len`/`seq_len`
  overridden to 720, exactly as `build_experiment`'s override mechanism is
  already used for every other cell) -- but this was not run without
  explicit confirmation, per this project's no-silent-substitution rule.
  Two-dataset replication (ETTh1, Weather) is reported as-is.
- This is an Oracle upper-bound diagnostic. It says nothing about whether a
  trainable horizon-specific router could recover this headroom -- that is
  exactly the deferred next-round question (section 16).
- Uniform aggregation was used as the primary metric by design (host-
  independent, spec section 9); a host-weighted secondary version was not
  computed in this round (optional per spec section 18, not required once
  the primary result was already unambiguous).

## 16. Decision for Next Experiment

Per spec section 33/34, this round's headroom is large, consistent across
two datasets, robust to the budget-matched control, and benefits nearly
every query -- strong empirical motivation exists for a follow-up
`TRACK-A-HORIZON-RETRIEVAL-EXPERT01` (Shared Future-Aligned Retriever +
Horizon-Specific Residual Retrieval Heads), per spec section 34. That model
is explicitly NOT implemented in this round.

## Final Questions

1. **Short/mid/long horizon에서 Oracle Top-K가 실제로 다른가?**
   **YES.** B1<->B3 overlap 2.78% (ETTh1) / 0.67% (Weather); rank
   displacement of B1's Top-10 under the B3 criterion averages ~1900 /
   ~11800.
2. **특히 short vs long evidence가 충분히 다른가?**
   **YES.** Same evidence as above -- near-total disjointness, not a
   Top-K boundary tie artifact (T9/rank-displacement both confirm).
3. **Block-specific Oracle가 Global Oracle보다 전체 H720 MSE를 유의미하게 낮추는가?**
   **YES.** 25.72% (ETTh1) / 39.42% (Weather) relative improvement, 100%
   of test queries improved, bootstrap CI strictly excludes zero.
4. **그 개선은 대부분 단순히 candidate budget이 증가해서 발생한 것인가?**
   **NO.** Global Top-30 (0.2971 / 0.0928) is barely different from Global
   Top-10 (0.2969 / 0.0988) and still far worse than Block Oracle.
5. **budget-matched global retrieval보다도 Block retrieval이 좋은가?**
   **YES.** Global budget-matched (0.2961 / 0.0928) vs Block Oracle
   (0.2205 / 0.0598); bootstrap mean difference 0.0756 [0.0743, 0.0769]
   (ETTh1) and 0.0330 [0.0326, 0.0334] (Weather), both strictly positive.
6. **어떤 block에서 global retrieval regret가 가장 큰가?**
   **B1 (short horizon)** in both datasets: 0.1572 (ETTh1) / 0.0596
   (Weather), vs B3's 0.0492 / 0.0276.
7. **query 대부분이 Block retrieval의 이득을 얻는가, 일부 query에만 집중되는가?**
   **거의 전부.** 100% of test queries have `delta>0`; 99.8-100% see
   >10% relative gain. Not a niche-query effect.
8. **ETTh1과 Weather에서 결과가 반복되는가?**
   **YES**, same direction and same qualitative pattern (B1 regret >
   B2 regret > B3 regret; Global closest to B3; budget-matched control does
   not close the gap) -- Weather's absolute headroom is even larger
   (39.4% vs 25.7%).
9. **Solar까지 검증할 가치가 있는가?**
   **YES, in principle** -- both completed cells show large, consistent,
   robust headroom, so a third dataset would strengthen the claim. It was
   not run this round because Solar_720's canonical reference checkpoint
   does not exist yet (section 15); a low-risk workaround (Solar_96's args,
   `pred_len` overridden) is available pending explicit approval.
10. **최종적으로 `Shared + Horizon-Specific Retrieval Experts`를 구현할 충분한 empirical motivation이 있는가?**
    **YES**, on the two datasets tested. This is a Case A result across the
    board (large Global-vs-Block gap, budget-matched control does not close
    it, low overlap, near-universal per-query benefit) -- the strongest and
    cleanest headroom result of the three Track-A diagnostics run this
    session (Router-Oracle-Headroom01's headroom was 7.8-62.4% depending on
    cell; this experiment's was 25.7-39.4% with a tighter, more uniform
    story and a budget-matched control that Router-Oracle-Headroom01 did
    not need). Recommend proceeding to
    `TRACK-A-HORIZON-RETRIEVAL-EXPERT01` after Solar_720 either replicates
    the pattern or is explicitly deferred.
