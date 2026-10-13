# FINAL RESEARCH REPORT -- CARTS Multi-Query / Expert / Router Program

**Scope**: synthesizes four sequential tracks, each gated on the previous one's measured results (not assumed):
1. `TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01` (Phase 1)
2. `TRACK-AGGREGATE-ROUTER-FROZEN01` (Phase 2)
3. `TRACK-JOINT-AGGREGATE-ROUTED-MQ01` (Phase 3)
4. `TRACK-JOINT-ROUTER-GENERALIZATION01` (Phase 4)

**Stage-2 Mode throughout: Original CARTS Trainable Global Lambda** (`Y_final = B + lambda*(R-B)`, `lambda=sigmoid(a)`, gradient-trained global scalar, VAL-selected, TEST-evaluated once). This is the PRIMARY metric in every phase; validation-only beta-grid fusion appears only as a CONTROL/diagnostic.

---

## 1. How the research chained together

**Phase 1** corrected an earlier diagnostic to measure head utility at the *real* per-(query, channel) granularity the Hard trainer actually supervises on, confirmed bit-exact (modulo GPU float tie-breaking) against the real trainer. It found: (a) a 9.2pp gap between per-channel and global oracle gain, deciding the router should be **CHANNEL_AWARE**; (b) Individual-criterion utility (the Hard trainer's own training signal) and Aggregate-criterion utility (what actually gets deployed) are measurably **misaligned** -- a head that looks best by one criterion is often not the best by the other.

**Phase 2**, given CHANNEL_AWARE and the misalignment finding, asked whether a small router trained ONLY on frozen, past-only backbone features could recover some of that misalignment's cost -- i.e., select Aggregate-good heads without ever seeing the future. It found yes, with the Hard backbone, R1 (Aggregate Winner CE) target, Soft routing (`BEST_BACKBONE=Hard, BEST_ROUTER_TARGET=R1, BEST_ROUTING_MODE=Soft`), reproducible at the Aggregate-MSE level and partially at Stage-2 (6/8 cells).

**Phase 3**, given Phase 2's picks, asked whether jointly training the whole backbone (encoder + slot heads) with the router -- replacing the Individual Hard-argmin loss entirely with a pure Aggregate-KL routing loss, per Phase 1's misalignment finding -- does better than the cheaper frozen-router alternative. At seed=0 only, it found a real but non-uniform win (GO=TRUE), driven disproportionately by two Weather cells, with an explicitly-flagged, traced anomaly: router entropy collapse at H720 that helps Weather_720 (best result in the whole program, at the time) but hurts ETTh1_720.

**Phase 4** asked the question a single seed cannot answer: does any of this reproduce? Running the entire Phase 1->3 P100 chain fresh at seed=1 and seed=2 (seed=0 reusing Phase 1-3's own numbers, with one real gap found and filled with fresh evidence-matched provenance), it found that **the answer is no, not where it matters**. Phase 3's headline Weather_720 result reverses sign at seed=2. Only the *internal* comparison (Joint beats the Hard backbone's own naive mixture) reproduces; the comparison against the cheapest baseline in the whole program (V5 Mean-Mixture, no specialization, no routing) does not -- V5 MM has the best 8-cell-average Stage-2 MSE at every one of the 3 seeds tested.

## 2. What is the final recommended method?

**None of the specialization-plus-routing machinery (Hard backbone, Frozen Router, or Joint Router) is recommended as a deployable forecasting method.** The 3-seed Phase 4 evidence is unambiguous on this point: V5 Mean-Mixture -- simply averaging 5 query views with no per-head specialization and no learned routing at all -- has the best average actual-forecast (Stage-2) MSE at every seed tested. If a single practical recommendation must be made from this research program, it is: **use V5 Mean-Mixture; do not add Hard specialization or a router unless targeting the one cell (Weather, H=336) where Joint Router's win is reproducible.**

This is not the conclusion the research program initially expected to reach (Phase 2 and 3 were both GO), and it is reported here exactly as found, per this program's own standing instruction never to dress up a disappointing result.

## 3. What CAN legitimately be claimed as a contribution

- **A validated methodological finding** (Phase 1, bit-exact tested): Individual-criterion and Aggregate-criterion utility are measurably misaligned for multi-candidate retrieval heads. This finding does not depend on seed and is not contradicted by anything in Phase 2-4.
- **A validated mechanism finding** (Phase 2, reproduced internally across all 3 Phase-4 seeds): a frozen, past-only router CAN recover part of a specialized backbone's own internal Aggregate-oracle gap, outperforming that backbone's own naive mixture in 7/8 cells at every seed. This is real and reproducible **as an internal comparison**.
- **A validated diagnostic finding** (Phase 4, Q10): where Aggregate-level routing gains fail to reach the final forecast, the more likely bottleneck is the single global-scalar Stage-2 lambda fusion, not the routing/retrieval signal itself -- the fusion mechanism does not adapt to how much the retrieval output actually improved.
- **A validated negative/cautionary finding** (Phase 4, the program's central result): translating any of the above into an actual forecasting improvement over a simple, cheap Mean-Mixture-of-views baseline is **not reliable** across seeds, and a single-seed headline result (Weather_720 Joint Router, "best result in the whole program") can reverse sign at a different seed. This is itself a publishable finding -- a cautionary result about evaluating multi-head specialization/routing methods on a single seed.

## 4. What CANNOT legitimately be claimed

- That Hard-backbone specialization is a net-positive forecasting method relative to V5 Mean-Mixture. (Robustly contradicted, Phase 4 Table 6.)
- That the Joint Router improves over the much cheaper Frozen Router in a way that justifies its cost. (2 of 3 Phase-4 seeds show no Stage-2 improvement in either dataset.)
- That any of this holds, or fails to hold, under Full-Candidate support. (Not tested -- explicitly infeasible within this program's resources, ~660 GPU-hours for the full 3-seed x 8-cell grid given the ~27.5h/cell empirically measured cost. This is a genuine open question, not a finding either way.)
- Any claim of statistical significance from the 3-seed comparisons. Per the program's own standing instruction, n=3 is explicitly treated as too small for bootstrap/CI-style significance claims; all seed-robustness statements in this report are descriptive (raw values, win/loss counts), not statistical tests.

## 5. Anomalies traced during this program (not smoothed over at the time, and not contradicted later)

- **Phase 1**: 4/224 winner-computation mismatches between a naive `argmin` and the real trainer's `sort`-based tie-break -- traced to GPU floating-point non-determinism on exact ties, not a logic bug; fixed by matching the real trainer's own tie-break operation.
- **Phase 2**: Weather_720 is pathological for every router variant on both backbones -- consistent with Phase 1's finding that Weather_720 has the most extreme train/validation alpha-domain shift and with an even earlier track's finding that Hard's specialization gain was its *worst*, not best, forecasting result there. A single, internally-consistent story across three independent tracks, not a fluke of any one implementation.
- **Phase 3**: router entropy collapse at H720 under the no-auxiliary-loss variant, with opposite forecasting outcomes on ETTh1 (hurts) vs Weather (helps) -- traced through data/metric/implementation/protocol/model-structure and attributed to Weather_720's much larger pre-existing oracle gap giving a hard-committing router more real headroom to exploit.
- **Phase 4**: the Weather_720 "best result in the program" from Phase 3 does not survive to seed=2 (sign reversal). This is the program's most important late-stage correction: a result that passed every check available at the time (including explicit anomaly-tracing) turned out to be a single-seed artifact, discoverable only by the seed ablation Phase 4 was designed to run.

## 6. Standing rules compliance (brief summary)

Every GPU-touching run across all 4 phases executed on physical GPU1 only (verified via `nvidia-smi` before each launch; two incidental brief GPU0 touches by trivial already-completed test invocations were caught, confirmed clean, and transparently reported at the time). All training used fixed epoch counts with no early-stop-induced epoch skipping and every-epoch checkpointing; checkpoint selection was always post-hoc on validation metrics. Resource/timing logs (`resource_metrics.json`) were recorded per run. Stage-2 PRIMARY was the Original CARTS Trainable Global Lambda in every phase; validation-only beta-grid fusion never served as a primary result. `research/EXPERIMENT_LOG.md` was appended-only throughout, with every phase's purpose, implementation, validation, results, anomalies, and GO/STOP reasoning recorded before moving to the next phase.

## 7. Open questions for future work

1. **Full-Candidate validation** of Phase 4's findings (infeasible here; would need either a much smaller candidate-pool budget or substantially more GPU-time budget than this program had).
2. **A Stage-2 fusion mechanism that adapts to retrieval quality** (e.g. a per-sample or per-head-confidence lambda, rather than one global scalar) -- directly motivated by Phase 4's Q10 finding that the current fusion mechanism is the more likely bottleneck, not the routing signal.
3. **A seed-robust evaluation protocol as a standing practice** for any future router/specialization experiment in this codebase -- Phase 4's central lesson is that single-seed GO decisions in this problem family are not reliable; 3-seed (or more) evaluation should be the default going forward, not an afterthought.
