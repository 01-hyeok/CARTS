# PHASE C -- Trainable Residual-V5 Scorer

Shared-P100 candidate identity fixed. First Phase in this track that
trains Stage-1 from scratch. Architecture (the ONLY change from
canonical V5): `s_final_{h,i} = c_i + gamma * r_{h,i}`, where `c_i` is
the SAME coarse delta-last-cosine score Phase A/B reuse (unscaled --
already a cosine similarity in [-1,1], same range as `r_{h,i}`), `r_{h,i}`
is canonical V5's own per-head score `cosine(W_h z_q, z_i)` (unchanged
math), and `gamma` is exactly ONE new global scalar, shared by all 5
heads, init=0, trained jointly with the full encoder+SlotHeads under
the UNCHANGED canonical loss `KL(p_teacher || mean_h softmax(s_final_h/tau_s))`.
Code: `scripts/train_p100_residual_v501.py`.

Controls: **C0** = existing frozen canonical sign-fixed V5 (no new
run, numbers read from Phase A's beta=1.0 rows). **C1** = this
script's primary 10-epoch training, checkpoint selected by validation
Mean-Mixture RetMSE@10 (not round-robin, per explicit spec deviation
from `train_v_sharedtop100_01.py`'s own historical convention). **C2**
= a minimal, read-only control -- fresh random init, gamma FIXED at
1.0, zero training epochs, immediate val/test evaluation (per spec
section 19's "C2 가능하면 read-only/posthoc 또는 minimal control").

## Table 1 -- Test AggMSE@10, C0 vs C1 vs C2

| Cell | C0 (canonical V5) | C1 (Residual-V5, trained) | C1 vs C0 | C2 (fresh init, gamma=1, no training) | C2 vs C0 |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.4098 | 0.4259 | -3.93% (worse) | 0.4639 | -13.20% (worse) |
| ETTh1_720 | 0.6206 | 0.6174 | **+0.52% (better)** | 0.7489 | -20.65% (worse) |
| Weather_96 | 0.2643 | 0.2688 | -1.69% (worse) | 0.2480 | **+6.18% (better)** |
| Weather_720 | 0.6178 | 0.5835 | **+5.55% (better)** | 0.6546 | -5.96% (worse) |

(% = improvement; negative = worse than C0.) **C1 beats canonical V5
in exactly 2/4 cells (ETTh1_720 marginally, Weather_720
substantially) and loses in the other 2/4 (ETTh1_96, Weather_96)** --
this is NOT the clean ETTh1-vs-Weather dataset split Phase A found;
it does not line up with dataset identity at all (one ETTh1 cell
wins, one Weather cell wins; the other of each loses).

**C2's one genuinely surprising number**: on Weather_96, a completely
UNTRAINED, randomly-initialized encoder+SlotHeads, fixed gamma=1 (no
training at all), beats BOTH the trained Residual-V5 (C1) and
canonical V5 (C0) by over 6%. This is reported as observed, not
chased further (no new sweep was run to explain it, per the
pre-registered no-retuning rule) -- but Table 2's diagnostics below
offer a plausible mechanism: C2's head_top10_overlap is ~0.99 (heads
collapsed to near-identical rankings, since random projections of a
COMMON additive `c_i` term dominate over small per-head noise), so
C2's Mean-Mixture is effectively averaging 5 near-identical,
noise-perturbed copies of the SAME coarse ranking -- a form of
implicit ensembling/smoothing over the coarse score rather than a
meaningfully different ranking mechanism. In every other cell this
same mechanism does NOT help (C2 loses to C0 by 5.9-20.6%), so
Weather_96's result looks like an interaction specific to that cell's
own coarse-score quality rather than a general "training is
unnecessary" finding -- consistent with Weather_96 already being the
one cell where Phase A's coarse score carried unusual weight (the
best fixed beta there was 0.25, closest to coarse-only of any cell).

## Table 2 -- gamma trajectory (validation, per epoch)

| Epoch | ETTh1_96 | ETTh1_720 | Weather_96 | Weather_720 |
|---:|---:|---:|---:|---:|
| 1 | -0.216 | 0.187 | 0.327 | 0.416 |
| 2 | -0.316 | 0.306 | 0.497 | 0.726 |
| 3 | -0.399 | 0.408 | 0.642 | 1.033 |
| 4 | -0.464 | 0.497 | 0.775 | **1.340 (best epoch)** |
| 5 | -0.521 | 0.573 | 0.893 | 1.627 |
| 6 | -0.584 | 0.644 | 1.000 | 1.899 |
| 7 | -0.651 | 0.708 | 1.095 | 2.146 |
| 8 | -0.714 | 0.765 | 1.177 | 2.354 |
| 9 | -0.776 | 0.826 | 1.253 | 2.551 |
| 10 (best, 3 of 4 cells) | **-0.837** | **0.884** | **1.321** | 2.733 |

Gamma monotonically grows in magnitude in every single cell, never
oscillating back toward 0 -- the residual correction is consistently
being pushed to full strength or beyond (|gamma| ends between 0.84
and 2.73), never staying small. ETTh1 cells converge to NEGATIVE
gamma (the learned residual is SUBTRACTED from coarse); both Weather
cells converge to positive gamma around +1.3 to +2.7. Weather_720's
best checkpoint is actually epoch 4 (early stopping by validation
RetMSE@10, before gamma grew past 1.34) -- not epoch 10 like the
other 3 cells, the only cell where that happened.

## Table 3 -- mechanism diagnostics (test split, C1 vs C2)

| Cell | Arm | final-vs-coarse overlap | final-vs-residual overlap | head Top10 overlap | union size (of 100) | Spearman(final, teacher) |
|---|---|---:|---:|---:|---:|---:|
| ETTh1_96 | C1 | 0.230 | 0.002 | 0.375 | 26.9 | 0.395 |
| ETTh1_96 | C2 | 0.606 | 0.711 | 0.993 | 10.1 | 0.167 |
| ETTh1_720 | C1 | 0.206 | 0.824 | 0.803 | 14.5 | 0.496 |
| ETTh1_720 | C2 | 0.489 | 0.711 | 0.992 | 10.2 | 0.121 |
| Weather_96 | C1 | 0.295 | 0.618 | 0.242 | 34.5 | 0.385 |
| Weather_96 | C2 | 0.420 | 0.759 | 0.991 | 10.2 | 0.311 |
| Weather_720 | C1 | 0.239 | 0.686 | 0.383 | 27.3 | 0.393 |
| Weather_720 | C2 | 0.448 | 0.683 | 0.991 | 10.2 | 0.391 |

Two consistent patterns across all 4 cells:

1. **C1 (trained) always has dramatically lower head Top10 overlap
   than C2 (untrained)** -- 0.24-0.80 vs a flat ~0.99. Training lets
   the 5 heads genuinely differentiate under this architecture,
   consistent with `r_{h,i}`'s per-head `W_h` becoming meaningfully
   different after training, vs C2's random, undifferentiated `W_h`
   where the shared additive `c_i` term dominates every head equally.
   ETTh1_96's own C1 overlap (0.375) is close to this session's
   previously-measured canonical-V5 pairwise overlap on the same cell
   (0.376, `TRACK-V5-TQH-HEAD-OVERLAP-DIAG01`) -- i.e. Residual-V5
   reaches roughly the SAME head-diversity level as canonical V5 on
   ETTh1_96, not more, not less. Weather_720/Weather_96's C1 overlap
   (0.38, 0.24) is noticeably lower (more diverse) than that -- this
   track did not measure canonical V5's own head overlap on the
   Weather cells, so whether this is specific to the residual
   architecture or simply a Weather-cell characteristic is unresolved
   here.
2. **C1's final ranking always correlates with the Future-MSE teacher
   more strongly than C2's does** (Spearman 0.385-0.496 vs
   0.121-0.391) -- training clearly makes the SCORE more accurate by
   this measure in every cell, including Weather_96 where C2 still
   wins on the deployable AggMSE metric. This is the same
   dissociation Phase A's Table 5 already found (pointwise ranking
   correlation and realized aggregate-forecast accuracy are not the
   same thing) -- here it recurs in a completely different
   mechanism (a trained vs untrained scorer, not a beta sweep).

## Verdict

Per the pre-registered rule, no hyperparameter (gamma init, LR,
epochs, loss) was changed after seeing these results, and C2's
surprising Weather_96 number was not chased with a follow-up sweep.
**Residual-V5 does not consistently beat canonical V5**: 2/4 cells
win (one marginally, one substantially), 2/4 cells lose, and the
split does not even line up with dataset identity the way Phase A's
did. The architecture does succeed on its own narrower, explicitly
stated goal of preserving head diversity better than a collapsed
mixture would (head Top10 overlap 0.24-0.80 vs canonical V5's own
~0.38 baseline, where measured) and of producing a more
teacher-correlated ranking than an untrained residual does -- but
neither of those translates into a reliable AggMSE win. Per section
37, Phase D proceeds next regardless of this mixed result.
