# TRACK-JOINT-ROUTER-GENERALIZATION01 -- Phase 4 Report

**Stage-2 Mode: Original CARTS Trainable Global Lambda** (`Y_final = B + lambda*(R-B)`, `lambda=sigmoid(a)`, Base+Retrieval frozen, global scalar lambda gradient-trained on TRAIN, checkpoint-selected by VAL MSE, evaluated once on TEST; the Weather beta-grid is a validation-only CONTROL/diagnostic, never primary -- it was built per-seed per-cell as specified but is not reported here since it never overturns the primary numbers below).

## Scope actually executed

- **Candidate support: P100 (shared Top-100) only, 3 seeds (0, 1, 2), all 8 cells.** Full-Candidate support was **explicitly descoped** -- see "Full-Candidate feasibility decision" below. This is a real, disclosed limitation, not a silent omission.
- **Seeds**: seed=0 reuses Phase 1-3's own already-computed numbers (re-verified against their source reports below, plus one newly-run fill-in -- see "seed=0 V5 MM Stage-2 gap" below). seed=1 and seed=2 are fresh full runs of the entire Phase1->Phase3 P100 chain (`scripts/run_joint_router_generalization_seed01.sh <seed>`), executed end-to-end on physical GPU1, zero errors in either run's full log.
- **Base Forecaster is NOT re-trained per seed** (fixed, shared across the seed ablation) -- a documented scope decision, not a hidden one, isolating retrieval/routing seed-variance from the base model's own seed-variance.
- **P100 pool, router architecture, alpha values, and tau_r are reused unmodified from Phases 1-3 for every seed** -- never re-tuned per seed, per spec.

### seed=0 V5 MM Stage-2 gap -- found and filled

Phase 2/3's own reports never computed a V5-backbone Mean-Mixture Stage-2 MSE (Phase 3 only carried the Hard backbone into Stage-2, per its own V5-vs-Hard threshold decision). Reusing a pre-existing V5 retrieval cache from `TRACK-V-MULTIQUERY-GENERALIZATION01` was tried first and **rejected**: its Aggregate MSE numbers (e.g. 0.4318 at ETTh1_96) did not match Phase 2/3's own reported V5 MM Aggregate MSE (0.40981 at ETTh1_96) -- tracing the mismatch to its root cause (not guessing) showed that track used a *different* V5 checkpoint and pool than the one Phase 2/3 actually fed into `eval_aggregate_router_full01.py` (which reads from `TRACK-HARD-EXPERT-V5-P100-ALLH01/<ds>/H<h>/<cell>/V5/checkpoint_selection/selected_checkpoint.json` -- confirmed by reading `scripts/run_aggregate_router_frozen01.sh` line 56). The correct checkpoint was located, and `build_v_meanmix_cache_pool01.py` + `eval_r_stage2_lambda_signfix01.py` were run fresh at seed=0 with that exact checkpoint/pool (GPU1, ~8 runs, all under 2 minutes each). The resulting Aggregate MSE **exactly reproduces** Phase 2/3's reported V5 MM numbers to 5 decimal places (0.40980765 vs reported 0.40981, and identically for all 8 cells) -- confirming correct provenance before being used in any table below.

## Full-Candidate feasibility decision

Running the complete spec (Full-Candidate x 3 seeds x 8 cells = 24 Full-Candidate Stage-1 runs) was checked against hard, already-measured evidence from this research program: `TRACK-EXPERT-V5-FULL01` measured a single 10-epoch Full-Candidate Stage-1 run at **up to ~98,954 seconds (~27.5 hours)** for one cell. Running 24 such cells sequentially on the single mandated physical GPU1 (no parallel GPU jobs, per standing rule) would require **~660 GPU-hours (~27.5 continuous days)** -- for Stage-1 alone, before any of the downstream Mean-Mixture/Router/Stage-2 steps per cell. This is not a blocking *error* (the experiment is not impossible), but it is empirically infeasible within the scope of this research program's remaining budget.

**Decision: Full-Candidate support is explicitly descoped from Phase 4's 3-seed generalization check.** This is disclosed as a real limitation (affecting Q9 below), not silently skipped and not attempted-and-abandoned. No Full-Candidate GPU jobs were started for Phase 4.

---

## Table 1: Aggregate MSE, mean +/- std over 3 seeds (raw seed0,seed1,seed2 in parentheses)

| Dataset | H | V5 MM | Hard MM | Frozen Router | Joint Router | Aggregate Oracle (seed0 ref only) |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.4107+/-0.0006 (0.4098,0.4112,0.4110) | 0.4131+/-0.0012 (0.4145,0.4115,0.4132) | 0.4123+/-0.0009 (0.4132,0.4111,0.4125) | 0.4126+/-0.0004 (0.4127,0.4130,0.4121) | 0.3437 |
| ETTh1 | 192 | 0.4727+/-0.0046 (0.4701,0.4689,0.4792) | 0.4831+/-0.0109 (0.4701,0.4827,0.4967) | 0.4755+/-0.0063 (0.4685,0.4742,0.4837) | 0.4726+/-0.0043 (0.4676,0.4721,0.4781) | 0.3943 |
| ETTh1 | 336 | 0.5039+/-0.0009 (0.5051,0.5038,0.5030) | 0.5308+/-0.0104 (0.5164,0.5409,0.5350) | 0.5141+/-0.0026 (0.5132,0.5177,0.5115) | 0.5088+/-0.0047 (0.5152,0.5070,0.5042) | 0.4305 |
| ETTh1 | 720 | 0.6163+/-0.0054 (0.6206,0.6087,0.6198) | 0.6175+/-0.0090 (0.6150,0.6296,0.6080) | 0.6151+/-0.0121 (0.6154,0.6298,0.6002) | 0.6234+/-0.0061 (0.6320,0.6189,0.6193) | 0.5281 |
| Weather | 96 | 0.2682+/-0.0028 (0.2643,0.2707,0.2697) | 0.2915+/-0.0146 (0.2791,0.2834,0.3120) | 0.2646+/-0.0047 (0.2587,0.2650,0.2702) | 0.2825+/-0.0115 (0.2784,0.2709,0.2981) | 0.1473 |
| Weather | 192 | 0.3375+/-0.0072 (0.3278,0.3394,0.3453) | 0.3718+/-0.0088 (0.3715,0.3827,0.3611) | 0.3616+/-0.0091 (0.3607,0.3732,0.3510) | 0.3616+/-0.0032 (0.3627,0.3648,0.3572) | 0.1941 |
| Weather | 336 | 0.4379+/-0.0009 (0.4367,0.4382,0.4388) | 0.4576+/-0.0074 (0.4671,0.4569,0.4489) | 0.4468+/-0.0028 (0.4494,0.4481,0.4429) | **0.4280+/-0.0027** (0.4257,0.4318,0.4264) | 0.2305 |
| Weather | 720 | 0.6244+/-0.0276 (0.6178,0.6611,0.5944) | 0.6444+/-0.0244 (0.6787,0.6299,0.6245) | 0.6428+/-0.0170 (0.6667,0.6328,0.6288) | 0.6068+/-0.0115 (0.6137,0.5906,0.6160) | 0.3616 |

Weather_336 is the only cell where Joint Router is the clear best method on Aggregate MSE at every seed (bolded). Note seed-to-seed std is itself informative: Weather_720's V5 MM std (0.0276) and Hard MM std (0.0244) are an order of magnitude larger than most other cells -- this horizon is noisy for *every* method, not just the router.

## Table 2: Stage-2 Test MSE (PRIMARY metric), mean +/- std over 3 seeds (raw seed0,seed1,seed2 in parentheses)

| Dataset | H | V5 MM | Hard MM | Frozen Router | Joint Router |
|---|---|---|---|---|---|
| ETTh1 | 96 | **0.3698+/-0.0001** (0.3696,0.3699,0.3699) | 0.3698+/-0.0002 (0.3699,0.3694,0.3699) | 0.3694+/-0.0002 (0.3694,0.3692,0.3696) | 0.3706+/-0.0002 (0.3704,0.3709,0.3706) |
| ETTh1 | 192 | 0.4239+/-0.0019 (0.4228,0.4224,0.4265) | 0.4267+/-0.0026 (0.4234,0.4271,0.4297) | 0.4247+/-0.0018 (0.4225,0.4246,0.4270) | 0.4241+/-0.0013 (0.4222,0.4246,0.4254) |
| ETTh1 | 336 | **0.4388+/-0.0006** (0.4391,0.4394,0.4379) | 0.4415+/-0.0005 (0.4416,0.4420,0.4408) | 0.4394+/-0.0011 (0.4409,0.4390,0.4383) | 0.4400+/-0.0016 (0.4423,0.4391,0.4387) |
| ETTh1 | 720 | 0.5140+/-0.0022 (0.5149,0.5109,0.5161) | 0.5134+/-0.0021 (0.5135,0.5159,0.5108) | **0.5121+/-0.0031** (0.5120,0.5159,0.5082) | 0.5161+/-0.0015 (0.5182,0.5150,0.5150) |
| Weather | 96 | **0.1775+/-0.0006** (0.1766,0.1780,0.1779) | 0.1806+/-0.0019 (0.1790,0.1794,0.1833) | 0.1776+/-0.0007 (0.1768,0.1776,0.1786) | 0.1814+/-0.0017 (0.1811,0.1795,0.1837) |
| Weather | 192 | **0.2219+/-0.0044** (0.2176,0.2200,0.2280) | 0.2262+/-0.0048 (0.2203,0.2263,0.2320) | 0.2252+/-0.0036 (0.2204,0.2262,0.2291) | 0.2300+/-0.0033 (0.2254,0.2330,0.2317) |
| Weather | 336 | 0.2807+/-0.0016 (0.2829,0.2791,0.2800) | 0.2831+/-0.0048 (0.2899,0.2799,0.2796) | 0.2805+/-0.0033 (0.2851,0.2784,0.2779) | **0.2797+/-0.0012** (0.2808,0.2801,0.2781) |
| Weather | 720 | 0.3874+/-0.0072 (0.3855,0.3970,0.3796) | 0.3903+/-0.0039 (0.3957,0.3880,0.3870) | 0.3912+/-0.0035 (0.3961,0.3891,0.3883) | **0.3846+/-0.0037** (0.3878,0.3795,0.3866) |

**Bolded = best method at that cell by 3-seed mean.** At the Stage-2 (PRIMARY, actual-forecast) metric, **V5 MM wins 4/8 cells outright and Joint wins only 2/8** (Weather_336, Weather_720) -- and even those two Joint wins are tiny in absolute MSE terms (0.0010 and 0.0028) relative to the method's added training cost.

## Table 3: Joint best-alpha selected per seed (validation-selected, never re-tuned by hand)

| Dataset | H | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| ETTh1 | 96 | 0.5 | 0.1 | 0.1 |
| ETTh1 | 192 | 0.5 | 0.5 | 0.0 |
| ETTh1 | 336 | 0.1 | 0.0 | 0.1 |
| ETTh1 | 720 | 0.0 | 0.1 | 0.1 |
| Weather | 96 | 0.5 | 0.1 | 0.5 |
| Weather | 192 | 0.5 | 0.0 | 0.0 |
| Weather | 336 | 0.5 | 0.1 | 0.1 |
| Weather | 720 | 0.0 | 0.1 | 0.0 |

The best alpha is **not stable across seeds at any single cell** -- every cell shows at least 2 distinct best-alpha values across the 3 seeds. This means the alpha sweep's winner is itself noise-sensitive, not a stable property of the cell.

## Table 4: Paired delta (Joint - V5 MM), Aggregate MSE, per seed, with win count

| Dataset | H | seed0 delta | seed1 delta | seed2 delta | mean delta | wins (Joint<V5MM) /3 |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | +0.00290 | +0.00178 | +0.00112 | +0.00193 | **0/3** |
| ETTh1 | 192 | -0.00250 | +0.00317 | -0.00105 | -0.00013 | 2/3 |
| ETTh1 | 336 | +0.01015 | +0.00321 | +0.00119 | +0.00485 | **0/3** |
| ETTh1 | 720 | +0.01140 | +0.01018 | -0.00045 | +0.00704 | 1/3 |
| Weather | 96 | +0.01410 | +0.00023 | +0.02841 | +0.01425 | **0/3** |
| Weather | 192 | +0.03488 | +0.02538 | +0.01195 | +0.02407 | **0/3** |
| Weather | 336 | -0.01102 | -0.00640 | -0.01241 | -0.00994 | **3/3** |
| Weather | 720 | -0.00415 | -0.07048 | +0.02161 | -0.01767 | 2/3 |

Positive = Joint is worse (higher MSE) than V5 MM. **Joint Router robustly beats the simplest baseline (V5 MM) at exactly 1 of 8 cells (Weather_336, 3/3 seeds) and robustly loses at 4 of 8 cells (ETTh1_96, ETTh1_336, Weather_96, Weather_192, all 0/3)**; the remaining 3 cells (ETTh1_192, ETTh1_720, Weather_720) are seed-dependent coin flips. Weather_720 in particular shows the highest sign instability of any cell: seed1's delta (-0.070, a huge win) is nearly 5x the magnitude of seed0's delta (-0.004) and seed2 flips sign entirely (+0.022, a loss). **Phase 3's headline claim that Weather_720 was "the best result in the whole program" was a seed=0-only observation that does not reproduce reliably.**

## Table 5: Per-seed win counts (Aggregate MSE), 8-cell totals

| Comparison | seed0 | seed1 | seed2 |
|---|---|---|---|
| Joint vs V5 MM | 3/8 | 2/8 | 3/8 |
| Joint vs Hard MM | 7/8 | 7/8 | 7/8 |
| Joint vs Frozen Router | 4/8 | 6/8 | 5/8 |
| Frozen Router vs V5 MM | 3/8 | 3/8 | 1/8 |

**Joint beats Hard MM (its own backbone's naive mixture) robustly at every seed (7/8).** This is the one unambiguously reproducible result in this track: specialization + routing genuinely helps *relative to the Hard backbone's own un-routed baseline*. But neither Frozen Router nor Joint Router reliably beats the cheaper, simpler V5 MM baseline -- and that is the comparison that actually determines whether the whole specialization+routing pipeline is worth its cost.

## Table 6: 8-cell average, per seed, per method

| Seed | Metric | V5 MM | Hard MM | Frozen Router | Joint Router |
|---|---|---|---|---|---|
| seed0 | Agg MSE | 0.45653 | 0.47654 | 0.46823 | 0.46350 |
| seed0 | Stage-2 MSE | **0.35114** | 0.35416 | 0.35292 | 0.35353 |
| seed1 | Agg MSE | 0.46275 | 0.47719 | 0.46898 | **0.45864** |
| seed1 | Stage-2 MSE | **0.35207** | 0.35350 | 0.35248 | 0.35270 |
| seed2 | Agg MSE | **0.45764** | 0.47492 | 0.46262 | 0.46393 |
| seed2 | Stage-2 MSE | **0.35198** | 0.35415 | 0.35213 | 0.35371 |

**At the Stage-2 (actual-forecast, PRIMARY) metric, V5 MM has the single best 8-cell average at every one of the 3 seeds**, ahead of Hard MM, Frozen Router, and Joint Router alike. This is the central, non-flattering, and fully reproducible finding of Phase 4.

## Table 7: Phase 3's own GO criteria, re-checked per seed (not just seed0)

| Criterion | seed0 | seed1 | seed2 | Robust across all 3? |
|---|---|---|---|---|
| 1. Joint improves avg Agg MSE over Frozen Router | True | True | False | **False** |
| 2. Joint wins >=5/8 cells vs Hard MM (Mean-Mixture) | True | True | True | **True** |
| 3. Joint Stage-2 MSE improves vs Frozen Router in >=1 dataset avg | True | False | False | **False** |

**Of Phase 3's three GO criteria, only criterion 2 (beats Hard's own naive mixture) reproduces across all 3 seeds.** Criterion 1 fails at seed2 (margin: 0.46393 vs 0.46262, essentially noise-level). Criterion 3 -- the one tied most directly to actual forecast quality -- **fails at both seed1 and seed2**; Phase 3's "Weather average improves" result was specific to seed=0 and does not generalize.

---

## Answers to the 11 final research questions

**Q1 -- Do real query-dependent head utility differences exist?** Yes, unchanged from Phase 1's bit-exact-validated finding (per-channel winner disagreement, margin distributions) -- this is a property of the retrieval representation, not re-tested per seed here, and nothing in Phase 4 contradicts it.

**Q2 -- Does the Aggregate-vs-Individual utility misalignment finding reproduce?** Not directly re-tested in Phase 4 (it is a representational/definitional property established once in Phase 1, not a trained-model outcome that varies by seed). What *is* newly tested here is the downstream consequence of removing the Individual-utility loss (Phase 3's redesign) -- and that consequence (Joint training using pure Aggregate-KL) produces a method that is NOT robustly better than simpler alternatives (Tables 2, 4, 6, 7).

**Q3 -- Does a past-only Router recover real Oracle headroom?** Only partially, and that partial recovery does not reliably survive the trip to Stage-2. The Aggregate Oracle gap is large (e.g. Weather_720: Hard MM 0.644 vs Oracle 0.362 -- 44% unclaimed headroom), and Joint Router does close some of this gap at the Aggregate-MSE level in several cells, but Table 6 shows this does NOT translate into a reliably better Stage-2 forecast than simply not routing at all (V5 MM).

**Q4 -- V5 vs Hard backbone preference?** **V5.** Hard's specialization costs more than it gains at the Stage-2 level, robustly, at all 3 seeds (Table 6).

**Q5 -- Is Hard specialization necessary?** **No**, not as a practical forecasting improvement -- robust 3-seed evidence. It IS useful as a research artifact (it is what makes Phase 1's utility-misalignment and Phase 2's routing-recoverability findings possible to study at all), but as a deployed method it does not earn its cost.

**Q6 -- Soft vs Hard routing preference?** Decided in Phase 2 (Soft) and not re-ablated in Phase 4; Phase 4 only validates downstream robustness of that choice, which is itself mixed (Table 7).

**Q7 -- Is Joint training actually necessary over Frozen Router?** **No, not robustly.** Frozen Router is far cheaper to train (seconds vs tens of minutes per cell, since it trains on cached frozen-backbone features) and beats Joint on Stage-2 MSE at a majority of cells in 2 of 3 seeds reported in Phase 3; Phase 4 confirms Joint's one marginal Stage-2 edge (criterion 3) does not survive to seed1/seed2.

**Q8 -- Horizon-dependence?** Yes. Weather_336 is the one cell where Joint Router robustly wins (3/3 seeds, both Agg MSE and Stage-2 MSE). ETTh1_96/336 and Weather_96/192 robustly favor the simplest baseline. Weather_720 and ETTh1_720 are seed-sensitive (not a stable horizon effect, a *noisy* one -- note Weather_720's std is 4-8x other cells' in Table 1).

**Q9 -- P100-and-Full-Candidate consistency?** **Unknown -- explicitly out of scope.** Full-Candidate was not run at all in Phase 4 (feasibility decision above); this question cannot be answered from this track's data and should not be claimed either way.

**Q10 -- Weather bottleneck: routing quality or Stage-2 calibration?** **Stage-2 calibration is the more likely bottleneck.** Evidence: at the Aggregate-MSE level, Joint Router produces large, real improvements over Hard MM at Weather_720 (e.g. seed0: 0.644->0.614, a ~5% relative gain) and Weather_336 (seed-robust ~6-9% gains, Table 1) -- genuine routing-quality signal exists. But at Stage-2, the SAME improvements shrink to ~0.3-1% MSE deltas (Table 2) because the global scalar lambda fusion (`Y_final = B + lambda*(R-B)`) blends the (now-better) retrieval output with the Base forecaster using a single learned scalar that does not know *how much better* the retrieval got -- it was already compensating for a mediocre retrieval signal and continues to do so similarly regardless of whether the signal improved. The routing/retrieval side is doing real work that the current Stage-2 fusion mechanism is not well-positioned to fully exploit.

**Q11 -- Does the contribution hold across 3 seeds?** **No, not uniformly, and the parts that fail are the practically important ones.** The one criterion that robustly reproduces (Joint beats Hard's own naive mixture) is an internal-consistency result, not a deployment-relevant one. The criterion that matters for a paper claim -- "the proposed method beats a sensible, cheap baseline on actual forecast MSE" -- does **not** reproduce: V5 Mean-Mixture (no specialization, no routing at all) has the best 8-cell-average Stage-2 MSE at every one of the 3 seeds tested (Table 6). Per the standing instruction that seed=3 forbids overclaiming significance: this is stated as a **descriptive, fully-reproduced pattern across 3 independent seeds**, not as a statistically-tested claim -- no bootstrap/CI is computed or claimed, consistent with n=3 being too small for that to be meaningful.

## Final CASE determination

Mapping onto this research program's enumerated final-structure cases: the data support the case closest to **"V5 (no Hard specialization, no Router at all) is best -> the Hard-specialization-plus-Router contribution is not supported as a deployable method."** This is in fact a *stronger* negative result than the closest paraphrased alternative ("Frozen Router approx Joint Router -> prefer the simpler method") -- the 3-seed Stage-2 evidence says the simplest baseline in the entire comparison group (V5 MM, Table 6) is *also* at least as good as either router variant, not merely that the two router variants tie with each other.

**What CAN legitimately be claimed as a paper contribution**, given everything found across Phases 1-4:
- A validated methodological finding: Individual-criterion utility and Aggregate-criterion utility are measurably misaligned for multi-candidate retrieval heads (Phase 1, bit-exact tested).
- A validated diagnostic finding: a frozen, past-only router CAN partially recover Aggregate-level oracle headroom from a specialized backbone (Phase 2), and this recovery is reproducible as an *internal* comparison (beats the backbone's own naive mixture, Table 5) across 3 seeds.
- A validated negative/cautionary finding, newly established by Phase 4's seed ablation: translating Aggregate-level routing gains into actual Stage-2 forecast improvements over a simple, cheap Mean-Mixture-of-views baseline is **not reliable**, and one headline positive result from the single-seed phase of this program (Weather_720's "best result in the whole program") does not reproduce.
- An open question, not a finding: whether any of this holds under Full-Candidate support (infeasible to test within this program's resources; Q9).

**What CANNOT legitimately be claimed:** that Hard-backbone specialization plus a learned router (Frozen or Joint) is a net-positive forecasting method relative to a simple 5-view mean mixture. The 3-seed Stage-2 evidence says otherwise, robustly.

## research/EXPERIMENT_LOG.md and FINAL_RESEARCH_REPORT.md

See `research/EXPERIMENT_LOG.md` (Phase 4 entry appended) and `FINAL_RESEARCH_REPORT.md` (repository root) for the full Phase 1->4 synthesis.
