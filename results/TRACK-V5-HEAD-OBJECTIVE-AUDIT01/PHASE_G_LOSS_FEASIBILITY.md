# PHASE G -- Soft/Hard Top-10 Alignment and Loss-Design Feasibility

> **PARTIALLY SUPERSEDED (2026-10-27)**: G1/G2 (soft/hard gap) are
> confirmed unchanged by `TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01`. G3's "C"
> (Complementarity Loss) verdict relied on a buggy near-zero cross-head
> cosine and is revised from NOT SUPPORTED to PARTIALLY SUPPORTED -- see
> `results/TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01/PHASE_E_REVISED_RESEARCH_VERDICT.md`.

## G1. Prediction gap: Mean-Mixture hard Top-10 vs full-pool soft expectation

`R_soft = sum_{i in pool} p_mix(i) * V_i` (expectation over ALL 100 pool candidates, weighted by the full Mean-Mixture probability -- the most natural fully-differentiable relaxation of `R_mix`, since `p_mix` already exists and is already a valid probability distribution) vs `R_hard = mean(Top10(p_mix))` (what V5 actually outputs at inference).

| Cell | Hard AggMSE (actual) | Soft AggMSE (full-pool expectation) | Mean Abs Gap | Soft vs Hard | Query/Channel-level correlation |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.40981 | 0.39912 | 0.0696 | Soft better by 2.61% | 0.969 |
| ETTh1_720 | 0.62057 | 0.57958 | 0.0759 | Soft better by 6.61% | 0.971 |
| Weather_96 | 0.26433 | 0.22417 | 0.1380 | Soft better by 15.19% | 0.589 |
| Weather_720 | 0.61780 | 0.45393 | 0.2417 | **Soft better by 26.52%** | **0.485** |

A real, substantial, and consistent gap exists everywhere (never in favor of hard), growing sharply from ETTh1 (2.6-6.6%) to Weather (15.2-26.5%). The query/channel-level correlation between soft and hard AggMSE also drops sharply on Weather (0.49-0.59 vs 0.97 on ETTh1) -- on Weather, a query where the soft relaxation looks good is markedly less predictive of whether the actual hard Top-10 will look good, a second, independent symptom of the same underlying issue (hard selection throws away much more useful probability mass on Weather than on ETTh1).

This matches this track's own PHASE E oracle-gap pattern (Weather's 31-subset oracle gap is also far larger, 38.7-54.0% vs ETTh1's 9.0-20.8%) and the project's broader, repeatedly-observed ETTh1-vs-Weather divergence in retrieval-signal quality. It is NOT the same mechanism as K2's sequential-greedy soft/hard gap (`EXP-SET-LOSS01`/`TRACK-A-SET-LOSS-CONTROL02`, PHASE A) -- that gap comes from relaxing a multi-STEP greedy choice process; this one comes from relaxing a single-shot hard-top-K-of-a-fixed-distribution operation -- but both recur as the SAME qualitative finding across structurally different mechanisms in this codebase: **a non-trivial soft/hard gap that grows, not shrinks, on the harder (Weather) retrieval regime.**

## G2. Existing differentiable Top-K / SoftTopK review (per PHASE A, no new relaxation introduced here)

- `utils/set_loss_experimental.py` (Soft Regret Mass, Set-Utility SoftCE) relaxes the SEQUENTIAL greedy-choice Oracle-imitation loss, not Mean-Mixture's single-shot probability pooling -- structurally different target.
- `train_k_multislot_predictive_retrieval01.py`'s K2 arm uses a Top-32 future-blind-subset + tempered-softmax relaxation of the AGGREGATE MSE itself as an explicit TRAINING loss (not just a diagnostic) -- this is the closest existing precedent to what an "Aggregate-aligned Loss" (L3) for V5 would look like, and it already showed a real, measured soft/hard gap post-training (0.5049 vs 0.5469 AggMSE, 7.7% relative) even AFTER the model was trained to minimize the soft version -- i.e. training against a soft relaxation does not fully close the gap even when that relaxation IS the training objective.
- No prior script implements a differentiable relaxation of Mean-Mixture's OWN Top-10 selection (the `R_soft` computed in this track's PHASE G1 is new, diagnostic-only, never used as a training signal anywhere).

## G3. Loss design feasibility judgment (spec section 10.3)

**A. Head별 KL (unweighted `mean_h KL(p_T||p_h)`, L1's second term)**

PHASE F shows this provides a genuinely DIFFERENT (not redundant) gradient signal -- larger magnitude everywhere (1.04-1.55x per-head `W_m` norm), and non-trivially different DIRECTION on at least one cell (Weather_96 cosine 0.71). It also removes the Jensen's-inequality dilution that lets any one head coast on the others' average. **Verdict: PARTIALLY SUPPORTED** -- there is a real, measurable basis for "V5's current loss provides diluted, not absent, per-head pressure," but this diagnostic says nothing about whether applying stronger per-head pressure would improve AggMSE (PHASE C showed the heads already differ measurably in standalone quality without any such pressure, and PHASE E showed Mean-Mixture's own averaging sometimes HURTS AggMSE relative to a single good head -- so a mechanism that makes heads MORE individually teacher-aligned, without any aggregate-awareness, has no established causal link to AggMSE improvement, and TRACK-K's own K1 arm result -- "multi-slot structure alone, no aggregate loss, does NOT beat the single-slot baseline" -- is a close, discouraging precedent).

**B. Aggregate-aligned Loss (`MSE(R_soft, Y_q)`, L3's second term)**

PHASE G1 establishes the soft/hard gap is real and substantial, especially on Weather (where it matters most, since Weather is also where Mean-Mixture underperforms a single head the most, PHASE C/E). This is the single most directly-relevant new evidence this track produced for L3: **a differentiable aggregate signal (`R_soft`) is readily available (it is already computed from `p_mix`, no new machinery needed) and correlates well with the hard target on ETTh1 (r=0.97) but POORLY on Weather (r=0.49-0.59)** -- meaning on exactly the dataset where an aggregate-aligned loss is most needed, the natural soft relaxation is also the LEAST faithful proxy for what actually happens at inference. **Verdict: PARTIALLY SUPPORTED** -- real headroom and a ready-made relaxation exist, but the relaxation's own fidelity is weakest where the opportunity is largest, a genuine obstacle TRACK-K2/M's experience (soft-trained models still show a real soft-hard gap post-training) suggests will not resolve itself simply by training against it.

**C. Head Complementarity Loss (explicit error-interaction supervision, L2's third term)**

PHASE D found cross-head error cosine is already small (0.001-0.009) everywhere -- heads are already nearly orthogonal in error space WITHOUT any complementarity-specific pressure. There is very little obvious headroom left to capture by explicitly PUSHING heads further apart in error space (they are not redundant to begin with), and M's own prior finding (PHASE A) that an aggregate-aligned loss ALONE (no explicit complementarity term) already fully captures and exceeds K2's complementarity-driven gain is a direct, close precedent against adding a third explicit term. The spec's own required mathematical check (does `L_comp` duplicate information already inside the aggregate MSE loss?) resolves cleanly here: `AggMSE = D + C`, and `C` for the actual selected set already IS a (cross-term-shaped) function of exactly the pairwise interactions `L_comp` would separately supervise -- an aggregate-aligned loss (B) already carries a gradient signal through `C`, making a standalone `L_comp` term redundant with B rather than a complementary addition, UNLESS it is specifically needed to counteract a degenerate-collapse failure mode of `L_comp`-free aggregate training (e.g., all heads collapsing to the SAME picks) -- M's own result (both M1/M2 showed LOWER cross-term `C` than K2, with no explicit complementarity term beyond what the aggregate objective itself provides) suggests this collapse risk is not realized in practice either. **Verdict: NOT SUPPORTED** at the level of "needed as a separate term" -- mathematically substantially redundant with B, and the one prior track (M) that tested a closely analogous design chose NOT to add an explicit complementarity term and still exceeded the complementarity gain of the method that motivated this question (K2) using only a relevance-protection + aggregate-objective combination.

## Mandatory honest caveat

G1-G3 are diagnostic-only findings at a SINGLE frozen checkpoint / single seed, never a trained outcome. None of A/B/C above has been tested by actually training a model under the proposed loss -- TRACK-M's own Stage1-STRONG-GO-but-Stage2-NO-GO result (PHASE A) is the single most important reason this track does NOT recommend treating any PARTIALLY/NOT-SUPPORTED verdict above as final without an actual Stage2-evaluated training run, should one be approved as a follow-up.
