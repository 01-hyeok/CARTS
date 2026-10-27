# TRACK-V5-HEAD-OBJECTIVE-AUDIT01 -- Final Report

> **PARTIALLY SUPERSEDED (2026-10-27)**: Table B / Table F's cross-head
> error-cosine numbers (PHASE D) were computed with a normalization bug
> (reported cosine = true cosine / H) and are WRONG -- heads are actually
> strongly redundant (cosine 0.65-0.95), not near-orthogonal. L2's verdict
> is revised from NOT SUPPORTED to PARTIALLY SUPPORTED. Every other number
> in this report (Tables A/C/D/E, L1/L3 verdicts, the overall MODIFY
> conclusion) is confirmed UNCHANGED by byte-level re-verification. See
> `results/TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01/FINAL_REPORT.md`.

**Diagnostic-only track. No new model, no new Loss, no Encoder/Scorer change, no Stage2 retrain.** Canonical sign-fixed V5 checkpoints, Shared-P100, seed0, GPU1 only (strictly sequential), all 4 cells (ETTh1 H96/H720, Weather H96/H720). Git commit at close: `f769be3e7b2598d03b439f876ccd7a824cf09311` (code authored this track; see `git status`/`git log` at commit time for the exact state this report describes).

## Provenance (every table below)

| Item | Value |
|---|---|
| Checkpoints | `checkpoints/track_v_sharedtop100_signfix01/{ETTh1,Weather}/H{96,720}/seed0/pool_top100/.../V5/checkpoint_epoch{10,9,4,6}.pth` |
| Candidate support | Shared-P100 (`candidate_pool_mode=coarse_topk`, size=100, metric=`delta_last_cosine`), `results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/pool_top100/shared_candidate_pool/{train,val,test}.pt` |
| seed | 0 (no new seed run) |
| Query/channel counts | ETTh1_96 test: 2785 queries x 7 ch = 19495 rows; ETTh1_720: 2161 x 7 = 15127; Weather_96: 10444 x 21 = 219324 (val 108675, test 219324 -- see note); Weather_720: 9820 x 21 = 206220 |
| Scripts (new, this track) | `scripts/diag_v5_head_objective01.py` (Phase B/C/D/E/G data), `scripts/diag_v5_head_gradient01.py` (Phase F), `scripts/analyze_v5_head_objective01.py` (CPU aggregation), `tests/test_v5_head_objective_audit01.py` (12/12 pass) |
| Raw artifacts | `results/TRACK-V5-HEAD-OBJECTIVE-AUDIT01/phase_b/<cell>/phase_b_verify.json`, `phase_cdeg/<cell>/full_pass_{val,test}.parquet` (per-query-channel raw rows), `phase_f/<cell>/phase_f_gradient.json`, `phase_analysis/<cell>/analysis.json` (aggregated) |
| Repo regression | 1679 passed, 2 pre-existing failures (unchanged, unrelated: `test_stage1_new_losses.py`, `test_stage2_oracle_topk.py`), no new regression |

## Table A -- Head-level retrieval quality (see PHASE_C, full numbers)

Summary: Mean-Mixture is the best AggMSE arm on ETTh1_96 only; a single head beats Mean-Mixture's AggMSE on the other 3 cells (ETTh1_720 by 0.8%, Weather_96 by 20.1%, Weather_720 by 9.5%). Mean-Mixture is the best Recall@10/NDCG@10 arm in EVERY cell. Best-individual-head and best-aggregate-head coincide on ETTh1 (both cells) and disagree on Weather (both cells).

## Table B -- Head diversity and complementarity (see PHASE_D, full numbers)

Cross-head error cosine is small everywhere (0.001-0.009), far smaller than TRACK-J3's candidate-level error cosine (0.50-0.62) -- the 5 heads' own aggregate predictions are nearly orthogonal, not strongly redundant or strongly cancelling. Negative-cosine (true cancellation) fraction is larger on Weather (5.7-14.9%) than ETTh1 (0.0-3.3%). Overlap and error-cosine correlate positively within every pair/cell (query/channel-level Spearman, never pooled-as-independent-candidates) -- low overlap associates with lower cosine, but never with strong negative cosine on average.

## Table C -- Head contribution (see PHASE_E, full numbers)

| Cell | All-5 Mean-Mixture | Best Single Head (val-sel.) | Best Fixed Subset (val-sel.) | Per-Query Oracle (31 subsets, upper bound) | Gap % |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.40981 | 0.42085 | 0.41079 | 0.32441 | 20.84 |
| ETTh1_720 | 0.62057 | 0.61826 | 0.61826 | 0.56499 | 8.96 |
| Weather_96 | 0.26433 | 0.22004 | 0.27689 (worse than All-5) | 0.12156 | 54.01 |
| Weather_720 | 0.61780 | 0.63972 (worse than All-5) | 0.64389 (worse than both) | 0.37866 | 38.71 |

## Table D -- Gradient analysis (see PHASE_F, full numbers)

| Cell | Encoder grad norm L0 (Mean-Mixture-first) | Encoder grad norm L_head (per-head avg) | Gradient cosine |
|---|---:|---:|---:|
| ETTh1_96 | 1.727 | 2.195 | 0.853 |
| ETTh1_720 | 5.478 | 5.641 | 0.990 |
| Weather_96 | 1.050 | 1.388 | 0.711 |
| Weather_720 | 5.392 | 6.302 | 0.910 |

## Table E -- Soft/Hard alignment (see PHASE_G, full numbers)

| Cell | Soft AggMSE | Hard AggMSE | Prediction Gap | Query-wise correlation |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.39912 | 0.40981 | soft better 2.6% | 0.969 |
| ETTh1_720 | 0.57958 | 0.62057 | soft better 6.6% | 0.971 |
| Weather_96 | 0.22417 | 0.26433 | soft better 15.2% | 0.589 |
| Weather_720 | 0.45393 | 0.61780 | soft better 26.5% | 0.485 |

## Table F -- Overlap with existing research (duplication check)

| This track's piece | Closest existing experiment | Same? | Difference | New training needed? |
|---|---|---|---|---|
| PHASE C head quality | `TRACK-HARD-EXPERT-V5-P100-ALLH01` per-head JSONs | Same checkpoint, same P100, same per-head decomposition machinery | This track's "Mean-Mixture" row uses TRUE `mean_mixture_topk_selection`; that track's own `rr` field is a DIFFERENT round-robin rule (PHASE B finding) | No -- V5 per-head rows transcribed; Mean-Mixture row freshly (but cheaply) recomputed |
| PHASE D head complementarity | TRACK-J3 (candidate-level complementarity) | Same math (dot/cosine/D-C), different object (heads' own `R_m` vs candidates within one set) | New object of study | No |
| PHASE E subset oracle | `TRACK-HARD-EXPERT-V5-P100-ALLH01`'s single-head-only oracle | Single-head oracle already existed; 31-subset oracle did not | Strictly larger enumeration | No |
| PHASE F gradient | Soft/Hard Expert RE-TRAINING (full new training runs) | Different mechanism entirely -- this is a one-step frozen-checkpoint diagnostic, never a training run | New measurement angle on an old loss-design question | No |
| PHASE G soft/hard gap | K2's sequential-greedy soft/hard gap (`EXP-SET-LOSS01`) | Same qualitative phenomenon (soft-hard gap exists, non-trivial) | Different mechanism (Mean-Mixture single-shot pooling vs sequential greedy choice) | No |
| L2 Complementarity Loss feasibility | TRACK-M (relevance-constrained multislot) | M's own result (Stage1 STRONG_GO, Stage2 NO-GO for an aggregate-aligned method) is the single most load-bearing prior-art fact against assuming any Stage1 win here would transfer | Different architecture (5 shared-encoder heads vs 10 relevance-slots), same cautionary lesson | N/A (caution, not duplication) |

## Final questions (spec section 14)

**1. 기존 V5의 Head들은 실제로 서로 다른 후보를 검색하는가?** Yes, pairwise Top-10 overlap is well below 1.0 everywhere (e.g. ETTh1_96: 0.237-0.509; Weather_96: 0.148-0.353) -- heads are not collapsed, consistent with `TRACK-HARD-EXPERT-V5-P100-ALLH01`'s own finding that plain V5 already shows non-trivial diversity with zero specialization pressure.

**2. Head별 RetMSE와 AggMSE는 얼마나 다른가?** On ETTh1 the best-RetMSE head is also the best-AggMSE head (both cells). On Weather they diverge in both cells (Weather_96: best RetMSE=H4, best AggMSE=H3; Weather_720: best RetMSE=H3, best AggMSE=H2) -- individual relevance and aggregate quality are NOT interchangeable criteria, specifically on the dataset where this project has repeatedly found the retrieval signal weakest.

**3. Head 간 오류 보완성이 실제로 존재하는가?** Weakly. Cross-head error cosine is small (0.001-0.009) everywhere -- heads are close to orthogonal, not strongly complementary (true cancellation, cosine<0, occurs in only 0-15% of query-channel-pair rows). There is SOME complementarity (more on Weather than ETTh1), but it is modest, not dramatic.

**4. 그 보완성이 최종 Mean-Mixture Top-10에 반영되는가?** Partially and inconsistently. Mean-Mixture's own within-set `C` is the dominant term of its AggMSE in every cell (70-87% of AggMSE), but Mean-Mixture AggMSE beats the best single head in only 1 of 4 cells (ETTh1_96) -- the modest cross-head complementarity found in Q3 does not reliably translate into Mean-Mixture being the best available choice.

**5. Head를 일부 제거하거나 조합하면 실제 예측 성능이 개선되는가?** Yes, with a large PER-QUERY oracle headroom (9-54%), but NO deployable fixed strategy tested here reliably captures it -- the val-selected Best Fixed Subset is WORSE than plain Mean-Mixture on 2 of 4 cells (Weather_96, Weather_720), a genuine overfitting risk from comparing 31 correlated candidates against a finite validation set.

**6. 기존 KL은 Head별 학습 신호를 충분히 제공하는가?** No head is starved (gradient norm spread never exceeds ~1.9x within a cell), but the per-head teacher-matching pressure is measurably DILUTED relative to an unweighted per-head-KL alternative (PHASE F: `L_head` gradients are uniformly 1.04-1.55x larger per head; full-gradient cosine with `L0` ranges 0.71-0.99, i.e. not identical).

**7. Head별 KL을 추가할 가치가 있는가?** PARTIALLY SUPPORTED by the gradient evidence alone (Q6), but TRACK-K's K1 arm (multi-slot structure + implicit per-head pressure, NO aggregate loss) is a close, discouraging precedent: it did not beat the single-slot J1 baseline. Per-head KL without any aggregate-awareness has no established causal path to better AggMSE in this codebase.

**8. Aggregate Loss는 기존 실험과 무엇이 다른가?** V5's current loss never supervises the aggregate prediction at all (pure KL on the pooled probability distribution). PHASE G1 shows a natural, already-computable relaxation (`R_soft`, the full-pool probability-weighted expectation) has a real, substantial, dataset-dependent gap to the actual hard Top-10 output (2.6-6.6% on ETTh1, 15.2-26.5% on Weather) and a correlation with the hard target that COLLAPSES on Weather (0.49-0.59 vs 0.97 on ETTh1) -- exactly where an aggregate-aligned loss is most needed, its natural relaxation is least faithful.

**9. Complementarity Loss는 Aggregate Loss에 비해 어떤 새로운 학습 신호를 제공하는가?** Very little that is not already inside the aggregate MSE's own `C` term (PHASE D: cross-head cosine is already small without any explicit pressure; mathematically, `AggMSE = D + C`, and an aggregate-aligned loss already carries gradient through `C`). TRACK-M's own finding (an aggregate-aligned objective alone exceeded K2's complementarity-specific gain, with no separate complementarity term) is direct precedent that this redundancy is not just a theoretical concern.

**10. 세 Loss 중 어떤 것을 다음 실험에서 먼저 검증해야 하는가, 명확한 우선순위와 근거:**

**Priority 1 -- L3 (Aggregate-aligned Loss), ETTh1 cells first, Weather cells as a SEPARATE, harder test.** Rationale: (a) the soft/hard gap and oracle-subset gap are both real and substantial, meaning there IS headroom an aggregate-aligned loss could in principle capture; (b) on ETTh1, the natural relaxation (`R_soft`) is highly faithful to the hard target (r=0.97) -- a strong prior that training against it is likely to transfer reasonably to the actual hard inference behavior, unlike Weather where the relaxation itself is unreliable (r=0.49-0.59) and should be treated as a SEPARATE, higher-risk test, not assumed to work the same way; (c) TRACK-K2's own precedent shows this family of loss CAN move AggMSE and Recall@10 favorably at Stage1, with the caveat (TRACK-M's Stage2 NO-GO) that Stage2 transfer must be checked explicitly and not assumed.

**Priority 2 -- L1 (Head-KL addition), as a secondary/ablation arm alongside L3, not alone.** The gradient evidence (PHASE F) shows it is a genuinely different signal, but TRACK-K1's precedent (structure + implicit per-head pressure alone, no aggregate loss, did not beat baseline) argues against testing it in isolation; it is a reasonable ADDITION to test jointly with L3 (does `lambda_H` help on top of an aggregate loss?), not a standalone priority-1 candidate.

**Not recommended as a separate term -- L2's Complementarity term.** Both the mathematical redundancy argument (PHASE G3-C) and TRACK-M's own empirical precedent argue against adding it; if L3 alone already reduces `C` sufficiently (as TRACK-M found for its own aggregate-aligned loss), no separate complementarity term is needed. This track's Final Verdict addresses L2 as "L3 alone, not L3+explicit-complementarity" for exactly this reason.

## Final Verdict (section 13)

| Loss | Differentiation from existing work | Training objective validity | Head Gradient meaning | Hard Top-10 alignment | Error-Complementarity basis | Implementation difficulty | Instability risk | Stage2 improvement basis |
|---|---|---|---|---|---|---|---|---|
| **L1** (Mean-Mixture KL + Head KL) | Real (unweighted avg, distinct from Soft/Hard Expert) | Valid but historically unrewarded alone (TRACK-K1) | Stronger, less-diluted per-head pressure (PHASE F) | N/A (does not touch selection) | N/A | Low (reuses `kl_per_head`) | Low | **No established basis** -- no prior track showed per-head KL alone helps AggMSE |
| **L3** (Mean-Mixture KL + Aggregate-aligned) | Real; first Mean-Mixture-specific aggregate loss in this codebase (closest precedent K2 is a different mechanism) | Valid, directly targets the actual deployed quantity | N/A | **Good on ETTh1 (r=0.97), poor on Weather (r=0.49-0.59)** -- dataset-dependent risk flagged explicitly | N/A | Moderate (needs a differentiable relaxation decision -- `R_soft` is the natural, already-available candidate) | Moderate (TRACK-K2's own soft-hard gap persisted even after training against the soft version) | **Real Stage1 basis (K2 precedent); Stage2 transfer is UNPROVEN (TRACK-M's own aggregate-aligned loss failed Stage2 despite a Stage1 STRONG_GO)** |
| **L2** (+ explicit Complementarity term) | Marginal -- largely redundant with L3's own `C`-term gradient (PHASE G3-C) | Valid in isolation but likely duplicative given L3 | N/A | Same risk profile as L3, plus an added degenerate-collapse risk the spec itself flags (naive error-dot minimization can inflate error magnitude) | **Weak basis for a SEPARATE term** -- cross-head cosine already near-zero without one (PHASE D) | Higher (extra loss term, extra coefficient to tune, extra degeneracy risk) | Higher (explicit anti-correlation objectives are a known degeneracy risk, per spec's own warning) | No basis beyond what L3 alone would already provide, per TRACK-M precedent |

**GO / MODIFY / STOP: MODIFY.**

Do not proceed directly to implementing L1+L2+L3 as specified. Instead: **test L3 alone first** (Mean-Mixture KL + a differentiable aggregate-MSE term using the readily-available `R_soft` full-pool expectation, or a Top-M soft-subset relaxation matching K2's own precedent), on ETTh1 H96/H720 first (where PHASE G showed the relaxation is faithful), with an EXPLICIT Stage2 evaluation gate (per TRACK-M's own cautionary precedent -- never approve based on Stage1 AggMSE alone) before any Weather cell or any L1/L2 addition is attempted. This track's own evidence does not support L2's complementarity term as a separate addition under any ordering, and only PARTIALLY supports L1 as a secondary ablation once L3 is validated at Stage2, not as a priority-1 candidate on its own.

## Explicit NOT VERIFIED / out-of-scope items

- No actual training run under L1/L2/L3 was performed (diagnostic-only track, per spec). All feasibility judgments are based on gradient/relaxation-fidelity diagnostics at the frozen checkpoint, not trained outcomes.
- Stage2 impact of any candidate loss is NOT VERIFIED -- this track explicitly reused the existing Stage2 reference numbers only for PHASE B reproduction, never retrained Stage2.
- Secondary cells (ETTh1_96, Weather_96) were run to the SAME depth as the primary cells (ETTh1_720, Weather_720) in this track, not merely spot-checked -- all 4 cells appear in every table.
- Single seed (seed0) throughout; no claim of cross-seed stability is made for any numeric finding in this report.
