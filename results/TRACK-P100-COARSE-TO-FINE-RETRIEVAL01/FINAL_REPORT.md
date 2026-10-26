# TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- Final Report

Fixed Shared-P100 candidate identity in every arm, every Phase. Stage2
uses the Original CARTS Trainable Global Lambda
(`Y_final = B + lambda*(R-B)`, `lambda=sigmoid(a)`), reused byte-for
-byte unmodified (`scripts/train_r_stage2_lambda01.py`'s own
`load_tensors`/`GlobalLambdaGate`/`eval_gate`), validation-selected,
test evaluated once, identical Base Forecaster checkpoint per cell
across all 5 arms.

## Table 1 -- Stage1 Retrieval, test AggMSE@10 (lower is better)

| Dataset | H | V5 | Fixed Fusion (A) | Adaptive Fusion (B) | Residual V5 (C) | Set-aware (D) |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.4098 | 0.4098 | 0.4155 | 0.4259 | 0.4393 |
| ETTh1 | 720 | 0.6206 | 0.6206 | 0.6343 | 0.6174 | 0.6083 |
| Weather | 96 | 0.2643 | 0.2615 | 0.2558 | 0.2688 | 0.2183 |
| Weather | 720 | 0.6178 | 0.6002 | 0.6011 | 0.5835 | 0.5607 |

Phase A's "Fixed Fusion" equals V5 exactly on both ETTh1 cells
because the val-selected beta there is 1.0 (pure V5) -- confirmed, not
a bug (see Phase A report). No single new arm (A/B/C/D) beats V5 on
every cell; each wins on a different subset.

## Table 2 -- Stage2 Forecast, test MSE (lower is better)

| Dataset | H | Base (no retrieval) | V5 | Fixed Fusion (A) | Adaptive (B) | Residual (C) | Set-aware (D) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3924 | 0.3696 | 0.3696 | 0.3708 | 0.3760 | 0.3761 |
| ETTh1 | 720 | 0.5604 | 0.5149 | 0.5149 | 0.5197 | 0.5121 | **0.5096** |
| Weather | 96 | **0.1694** | 0.1767 | 0.1760 | 0.1751 | 0.1778 | 0.1714 |
| Weather | 720 | **0.3194** | 0.3855 | 0.3808 | 0.3802 | 0.3783 | 0.3715 |

Bold = the single best number in that row across ALL 6 columns
(including Base). **This is the most important and least comfortable
finding of the whole track: on both Weather cells, Base ALONE (no
retrieval at all) beats every one of the 5 retrieval arms, including
the best-performing one (Phase D).** On ETTh1, the opposite holds --
every retrieval arm beats Base, and the best retrieval arm per cell
(V5/A tied on H96, Phase D on H720) gives a real, meaningful
improvement over Base (5.8% on H96, 9.1% on H720). This is a cleaner,
more decisive version of the dataset-split pattern this whole track
kept finding inside the retrieval-only metric (Table 1) -- at the
Stage2 level, on Weather, retrieval augmentation of this kind is not
just "not very helpful," it is actively harmful relative to simply
not using it, regardless of which of the four new mechanisms (A-D) is
applied.

## Table 3 -- Candidate quality summary (test split; full per-cell tables in each Phase's own report)

| Arm | RetMSE@10 range | D range | C range | Recall@10 range | NDCG@10 range |
|---|---:|---:|---:|---:|---:|
| V5 | 0.456-0.943 | 0.046-0.094 | 0.219-0.526 | 0.255-0.321 | 0.825-0.890 |
| Fixed Fusion (A) | 0.540-0.943 | 0.054-0.094 | 0.208-0.526 | 0.247-0.321 | 0.822-0.890 |
| Residual V5 (C) | 0.486-0.935 | not computed | not computed | 0.248-0.313 | 0.822-0.895 |

(Phase B/D do not report this exact RetMSE/D/C decomposition in their
own output format; see `PHASE_B_ADAPTIVE_FUSION_REPORT.md` and
`PHASE_D_SET_RERANK_REPORT.md` for their own comparable metrics; the
Residual-V5 evaluation script did not compute the D/C aggregation
-bias decomposition.) No arm shows a systematically better RetMSE or
NDCG range than V5 -- consistent with Table 1's mixed, non-dataset
-aligned pattern.

## Table 4 -- Diversity (pairwise head Top10 overlap; applicable arms only)

| Cell | V5 (baseline, measured in `TRACK-V5-TQH-HEAD-OVERLAP-DIAG01`) | Residual V5 (C1, this track) |
|---|---:|---:|
| ETTh1_96 | 0.376 | 0.375 |
| ETTh1_720 | not measured in this track | 0.803 |
| Weather_96 | not measured in this track | 0.242 |
| Weather_720 | not measured in this track | 0.383 |

Residual-V5 reaches almost EXACTLY canonical V5's own head-collapse
level on the one cell where both are known (ETTh1_96: 0.375 vs
0.376) -- the residual architecture neither worsens nor meaningfully
improves head diversity; it just inherits whatever level V5 already
has. Phase A/B/D do not change V5's own head mechanism (A/B
post-process its frozen scores; D operates one level above, on
Residual-V5's already-fixed Top-20), so their own pairwise-overlap
numbers are identical to whichever upstream V5/Residual-V5 produced
the scores -- not independently reported here to avoid double
-counting.

## Table 5 -- Complexity

| Arm | Additional trainable params vs V5 | Stage1 wall-clock (per cell, approx.) | Peak VRAM (Stage1 train) |
|---|---|---:|---:|
| V5 | 0 (baseline) | -- | -- |
| Fixed Fusion (A) | 0 (frozen fusion, no training) | seconds (eval-only sweep) | negligible |
| Adaptive Fusion (B) | ~1-2K (tiny 2-layer MLP, 18 features) | ~1-30 min (feature collection dominates on Weather) | negligible (CPU-cacheable features) |
| Residual V5 (C) | exactly 1 (gamma scalar) | 116s (ETTh1_720) -- 1500s (Weather_96), full 10-epoch run, measured | 102.6 MB (ETTh1_96) -- 6146.4 MB (Weather_720) |
| Set-aware (D) | ~10-15K (2-hidden-layer reranker, ~100-dim input) | tens of minutes to ~1hr/arm on Weather (K=10 sequential on-policy steps per batch) | not separately logged; same order as C |

Residual-V5 (C) is by far the cheapest new mechanism in parameter
count (exactly 1 scalar) yet shows the most erratic cell-to-cell
AggMSE behavior (Table 1) of any arm -- complexity and
reliability are not correlated in this track's results.

## Case interpretation (spec section 36)

- **CASE 1** (Fixed Fusion alone wins) -- partially true only on
  Weather_96/720's RETRIEVAL metric (Table 1), but Fixed Fusion is
  NOT the single best Stage2 arm anywhere (Table 2); not supported as
  an overall conclusion.
- **CASE 2** (Adaptive beta > fixed fusion) -- NOT supported; Phase B
  never clears Phase A by a reliable, non-degenerate margin (Phase
  B's own report: the "adaptive" model collapses to a near-constant
  scalar in 3/4 cells).
- **CASE 3** (Residual-V5 > frozen fusion) -- partially true at the
  retrieval level on 2/4 cells (ETTh1_720, Weather_720) but reverses
  on the other 2/4; not a reliable pattern.
- **CASE 4** (Set-aware > all pointwise scorers) -- NOT supported;
  Phase D's own ablation (D1 vs D3) shows the set-state-tracking
  mechanism contributes nothing measurable in any cell, even where D1
  itself beats other arms.
- **CASE 5** (all new arms ~ or < canonical V5; stop this line) --
  **this is the closest fit, with an important addition Table 2
  surfaces that CASE 5 as originally written does not fully capture**:
  not only do the new arms fail to reliably beat V5, but on Weather,
  NEITHER V5 NOR any of the 4 new mechanisms beats simply not
  retrieving at all (Base alone). The practical ceiling for this
  scorer family, within fixed Shared-P100, is not just "V5" -- on
  Weather specifically, it may be "no retrieval."

## Final answers to the track's four questions (spec section 40)

1. **Is the coarse score itself a useful prior?** Partially --
   Phase A found a real, test-confirmed fixed-fusion benefit on
   Weather (not ETTh1) at the retrieval level, but Table 2 shows this
   never becomes the best STAGE2 arm anywhere, and on Weather even
   the best Stage2 arm loses to Base. The coarse score carries some
   signal V5 lacks, but not enough to make retrieval worthwhile on
   Weather once Stage2 is accounted for.
2. **Does coarse/V5 trust need query-level adaptation?** No --
   Phase B found large oracle headroom (12-38%) but the deployable
   adaptive model could not capture it; it collapsed to a near
   -constant scalar in 3/4 cells.
3. **Does training V5 as a residual correction help?** Mixed, not
   reliably -- 2/4 cells win, 2/4 lose, not aligned with dataset
   identity; the architecture preserves V5's own head-diversity level
   but does not improve on it.
4. **Is pointwise ranking the bottleneck, vs set-level aggregate
   selection?** No -- Phase D found enormous Set Oracle headroom
   (14-55%) but definitively showed (via the D1-vs-D3 ablation) that
   the specific mechanism built to exploit it (sequential set-state
   tracking) contributes nothing; whatever gain D1 shows comes from
   elsewhere.

## Overall verdict

Across all 4 Phases and all 4 cells, no single new mechanism
reliably improves on canonical V5, and -- the track's most important
finding, uncovered only once Stage2 was run -- **on Weather, none of
the 5 retrieval arms (including canonical V5 itself) improves on
simply not retrieving at all.** This closes the
coarse-to-fine-retrieval research line within fixed Shared-P100 as
specified: CASE 5, with the Weather/Base finding as the most
consequential addition to that conclusion. Per the track's own scope
rules, no extension to seed1/2 or H192/H336 is warranted, since no
method here "명확하게 V5를 이길" (clearly beats V5) on the 4 cells
tested.
