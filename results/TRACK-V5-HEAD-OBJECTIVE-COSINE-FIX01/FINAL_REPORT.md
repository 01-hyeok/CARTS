# TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01 -- Final Report

Bug-fix-and-revalidation track. No new model, no new Loss, no Encoder/Scorer change, no Stage2 retrain. Fixes a real normalization bug in `TRACK-V5-HEAD-OBJECTIVE-AUDIT01`'s Head Error Cosine computation and re-derives every downstream interpretation that depended on it, using the EXISTING canonical V5 checkpoints and Shared-P100 pool (seed0, all 4 cells, GPU1 only).

## Summary of the bug

`scripts/diag_v5_head_objective01.py` computed `cos_ = (raw_dot/H) / (norm_m*norm_m)` instead of `raw_dot / (norm_m*norm_m)` -- it reused the already-`/H`-scaled Error Dot tensor as the Error Cosine numerator. Confirmed by a dedicated regression test (`test_test10_regression_old_buggy_formula_scaled_by_horizon`) and by direct row-level comparison of the real data: **the reported cosine was exactly `1/H` of the true cosine in every single row, every cell** (ratio measured = 96.0 for H96 cells, 720.0 for H720 cells, to 6 significant figures).

## Corrected numbers

| Cell | Old (buggy) Error Cosine | Corrected Error Cosine | Ratio |
|---|---:|---:|---:|
| ETTh1_96 | 0.0088 | **0.844** | 96.0 |
| ETTh1_720 | 0.00132 | **0.952** | 720.0 |
| Weather_96 | 0.0068 | **0.650** | 96.0 |
| Weather_720 | 0.00112 | **0.809** | 720.0 |

## What changed, what did not

**Changed**: Error Cosine itself, and every narrative conclusion drawn directly from its (wrong) magnitude -- the claim that "the 5 heads' errors are nearly orthogonal" is retracted; the corrected picture is that heads are **strongly redundant** in error space (cosine 0.65-0.95, medians even higher).

**Did not change (verified by byte-level row comparison of a fresh forward pass against the original parquet, not merely re-argued)**: Error Dot, Negative-Cosine Fraction, Overlap-vs-Cosine Pearson/Spearman correlation, Within-Mean-Mixture-Set `D`/`C`/AggMSE, the full 31-Subset Head Oracle, the Soft/Hard Gap, and the Gradient Analysis (PHASE F, computed by an entirely separate, unmodified script). See `PHASE_D_UNAFFECTED_METRICS.md` for the full exact-match table.

## Final questions

**1. 기존 Head Error Cosine은 실제로 잘못 계산됐는가?** Yes, confirmed exactly -- a normalization-scope bug, not a conceptual error in the cosine formula itself.

**2. 오류의 정확한 Root Cause는 무엇인가?** `dot_mn` (Error Dot, correctly `/H`) was reused as the Error Cosine's numerator instead of the raw (un-`/H`) dot product; the cosine denominator (`norm_m` product) was never `/H`-scaled, so the two scalings became mismatched, silently shrinking every reported cosine by exactly `H`.

**3. 수정 후 Head Error Cosine 값은 얼마인가?** 0.844 (ETTh1_96), 0.952 (ETTh1_720), 0.650 (Weather_96), 0.809 (Weather_720) -- pooled mean over all 10 head pairs, test split. Medians are 0.88-0.98 in 3 of 4 cells (Weather_96's median is 0.879 but its lower tail dips negative).

**4. 수정 전후 해석이 어떻게 달라졌는가?** From "heads are nearly orthogonal, little redundancy, little room for a complementarity loss" to "heads are strongly redundant, substantial room exists in principle -- but the spec's own cautions (within-set `C` is a separate, unaffected quantity; negative-cosine/true-cancellation fraction is still small; degenerate-solution risk is real) mean this new fact alone does not establish that an explicit Complementarity Loss would help."

**5. Head 간 Error Redundancy가 존재하는가?** Yes, substantially, in every cell -- this is the headline factual correction.

**6. Error Complementarity를 직접 학습할 필요성이 확인됐는가?** No -- "redundancy exists" and "a specific loss term to reduce it would help AggMSE" are different claims, and this track (diagnostic-only, no training) only establishes the former. The revised verdict for L2 moves from NOT SUPPORTED to PARTIALLY SUPPORTED, reflecting that the question is now genuinely open rather than closed by a (wrong) fact, not that it has been answered affirmatively.

**7. 기존 31-Subset Oracle 결과는 여전히 유효한가?** Yes, byte-identical, reverified by a fresh forward pass (PHASE D1).

**8. Soft/Hard Gap과 Gradient 분석은 그대로 유효한가?** Yes, both byte-identical/untouched (PHASE D3/D4) -- neither code path ever depended on the buggy cosine computation.

**9. L1/L2/L3 중 어떤 실험을 우선 진행해야 하는가?** **Unchanged from the original report: L3 first, on ETTh1 cells first, with an explicit Stage2 gate before Weather or before adding L1/L2.** The reasoning for this priority (soft/hard correlation 0.97 on ETTh1 vs 0.49-0.59 on Weather; TRACK-M's Stage1-win/Stage2-NO-GO precedent) is entirely built on metrics confirmed UNCHANGED by this fix. L2 is now a live, not a closed, SECOND-priority question (test it as an ablation once L3 is validated, to see if it adds anything beyond L3 alone) -- still not a priority-1 candidate, because the within-set `C` term L3 would already optimize is the quantity that actually drives AggMSE, and that term's own numbers never moved.

**10. 어떤 기존 실험이 새로운 Loss 실험과 가장 많이 중복되는가?** Unchanged from the original report -- `TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01` remains the closest and most load-bearing precedent (aggregate-aligned loss alone already reduced an analogous cross-term without a separate complementarity term, AND its own Stage1-win failed to transfer to Stage2) -- this bugfix does not change that track's own numbers or relevance at all.

## Overall conclusion: **MODIFY** (same as the original report; the correction changes a fact, not the recommended next action)

Proceed with the original plan: test L3 (Aggregate-aligned Loss, using `R_soft`) alone first on ETTh1 H96/H720, with an explicit Stage2 evaluation gate, before touching Weather or adding L1/L2. Additionally, given this track's correction: when L2 is eventually tested as a secondary ablation (per the original report's own sequencing), it should be evaluated with the specific, now-sharpened question "does an explicit complementarity term reduce within-set `C` or improve Stage2 beyond what L3 alone achieves" -- not dismissed in advance, and not assumed necessary either.

## Completion checklist (spec section 11)

1. Error Cosine normalization bug fixed -- DONE (`scripts/diag_v5_head_objective01.py`).
2. Synthetic tests pass -- DONE, 10/10 (`tests/test_v5_head_objective_cosine_fix01.py`).
3. Corrected Error Cosine recomputed, all 4 cells -- DONE, fresh forward pass, no new training.
4. Before/after comparison -- DONE (`PHASE_C_CORRECTED_COMPLEMENTARITY.md`).
5. Unaffected-metric reproducibility confirmed -- DONE, byte-level row comparison (`PHASE_D_UNAFFECTED_METRICS.md`).
6. Phase D research interpretation revised -- DONE (`PHASE_E_REVISED_RESEARCH_VERDICT.md`, supersedes the original `PHASE_D_HEAD_COMPLEMENTARITY.md`'s "near-orthogonal" narrative; that original file is left in place, unmodified, per the standing never-delete-history rule, but is now superseded -- see note below).
7. Phase G Loss Feasibility re-evaluated -- DONE (L2 upgraded to Partially Supported; L1/L3 unchanged).
8. Final GO/MODIFY/STOP report -- DONE, MODIFY.
9. Commit/Push -- pending, performed immediately after this report per standing session policy.

**No new Head Loss training was run or will be auto-started following this track**, per spec's explicit instruction.

## Note on the original track's artifacts

`results/TRACK-V5-HEAD-OBJECTIVE-AUDIT01/PHASE_D_HEAD_COMPLEMENTARITY.md` and the Error-Cosine-dependent portions of its own `FINAL_REPORT.md` are **not deleted or edited** (standing rule: never delete/modify historical results) -- they remain on disk exactly as originally written, but their Error-Cosine-based narrative ("heads are nearly orthogonal") is **superseded by this track's `PHASE_E_REVISED_RESEARCH_VERDICT.md`**. Any future reader of the original Phase D report should be pointed to this correction.
