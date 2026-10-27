# PHASE F -- Head KL Gradient Analysis

Script: `scripts/diag_v5_head_gradient01.py`. Diagnostic only -- the EXISTING canonical checkpoint is loaded, put into `.train()` mode only so gradients flow, 6 TRAIN batches x all channels are forward/backward'd TWICE per (batch,channel) with two different losses computed from the identical `scores`/teacher/mask, gradients read via `.grad`, then thrown away (`zero_grad` before each, no optimizer step, nothing saved, no checkpoint touched). Compares:

- **L0** (`KL(p_T || p_bar)`, `p_bar = mean_h p_h`) -- the loss V5 is ACTUALLY trained on.
- **L_head** (`mean_h KL(p_T || p_h)`) -- the unweighted per-head-KL alternative the spec's RQ1/L1 asks about (distinct from Soft Expert's responsibility-WEIGHTED version and Hard Expert's single-winner version, both already explored by retraining in prior tracks; this is the plain, equal-weight average, isolated as a pure gradient comparison at the SAME frozen point, never trained to convergence).

## Table D -- Gradient analysis (mean over all sampled query-channel pairs, train split)

| Cell | n pairs | Encoder grad norm (L0) | Encoder grad norm (L_head) | Gradient cosine (L0 vs L_head) |
|---|---:|---:|---:|---:|
| ETTh1_96 | 42 | 1.7274 | 2.1947 | 0.8533 |
| ETTh1_720 | 42 | 5.4776 | 5.6407 | 0.9897 |
| Weather_96 | 126 | 1.0500 | 1.3875 | 0.7106 |
| Weather_720 | 126 | 5.3925 | 6.3021 | 0.9098 |

## Per-head SlotHead `W_m` gradient norm (the only genuinely per-head parameter -- the encoder itself is shared, so a per-head encoder gradient does not exist as a separable quantity)

| Cell | Loss | H1 | H2 | H3 | H4 | H5 |
|---|---|---:|---:|---:|---:|---:|
| ETTh1_96 | L0 | 0.0650 | 0.0765 | 0.1229 | 0.0808 | 0.0852 |
| ETTh1_96 | L_head | 0.1008 | 0.1002 | 0.1287 | 0.1136 | 0.0998 |
| ETTh1_96 | ratio (L_head/L0) | 1.55 | 1.31 | 1.05 | 1.41 | 1.17 |
| ETTh1_720 | L0 | 0.0975 | 0.0988 | 0.0998 | 0.1005 | 0.0979 |
| ETTh1_720 | L_head | 0.1157 | 0.1042 | 0.1174 | 0.1054 | 0.1054 |
| ETTh1_720 | ratio | 1.19 | 1.05 | 1.18 | 1.05 | 1.08 |
| Weather_96 | L0 | 0.0513 | 0.0767 | 0.0550 | 0.0543 | 0.0557 |
| Weather_96 | L_head | 0.0741 | 0.1031 | 0.0636 | 0.0760 | 0.0717 |
| Weather_96 | ratio | 1.44 | 1.34 | 1.16 | 1.40 | 1.29 |
| Weather_720 | L0 | 0.2029 | 0.1796 | 0.1744 | 0.1792 | 0.2109 |
| Weather_720 | L_head | 0.2344 | 0.2648 | 0.2242 | 0.2008 | 0.2189 |
| Weather_720 | ratio | 1.16 | 1.47 | 1.29 | 1.12 | 1.04 |

## F3 -- answers to the required questions

**1. 기존 V5 Loss (`L0`)에서 특정 Head에 Gradient가 집중되는가?**
Mildly, and cell-specific, never a dramatic concentration. On ETTh1_96, H3's `W_3` gradient (0.1229) is ~1.9x H1's (0.0650) under L0 -- the largest within-cell spread found. ETTh1_720 is nearly uniform (0.0975-0.1005, <3% spread). Weather_96 shows H2 elevated (0.0767 vs 0.0513-0.0557 for the rest, ~1.4x). Weather_720 is close to uniform (0.1744-0.2109). No head is ever starved to near-zero gradient under L0 in any cell -- all 5 heads keep receiving a comparable, non-degenerate gradient magnitude, consistent with the previously-observed fact (PHASE C/`TRACK-HARD-EXPERT-V5-P100-ALLH01`) that no head "collapses" under plain V5 training.

**2. 특정 Head의 Gradient가 매우 작은가?** No -- every head's `W_m` gradient norm stays within the same order of magnitude in every cell (max/min ratio 1.05-1.9x, never >2x). Dead/vanishing heads are not observed at this checkpoint.

**3. Head별 KL (`L_head`)은 기존 V5 (`L0`)와 다른 Gradient를 제공하는가?** Yes, measurably, in both magnitude and (mildly) direction. `L_head`'s per-head `W_m` gradients are UNIFORMLY LARGER than `L0`'s for every head in every cell (ratios 1.04-1.55x) -- unsurprising since `L0` pools 5 heads' probabilities BEFORE taking KL (Jensen's inequality makes the mixture's KL to the teacher <= the average of the 5 heads' own individual KLs, so `L0`'s value, and apparently also its gradient magnitude on this data, is systematically smaller/weaker than `L_head`'s). The FULL gradient vector's cosine similarity between `L0` and `L_head` ranges from 0.71 (Weather_96) to 0.99 (ETTh1_720) -- always positive and often high, meaning the two losses mostly push parameters in similar overall directions, but NOT identically: Weather_96's 0.71 cosine is a real, non-trivial divergence in gradient direction, not just magnitude.

**4. 기존 V5 Loss가 Head들에 서로 다른 역할을 학습시킬 여지가 있는가?** The gradient-magnitude spread found in Q1 (up to ~1.9x within a cell) shows SOME differential pressure already exists under plain `L0`, even without any explicit per-head mechanism -- consistent with PHASE C/D's finding that the 5 heads DO end up with measurably different standalone quality (RetMSE spread up to 0.056 on ETTh1_96) and near-zero cross-head error cosine (PHASE D) despite training on an identical shared objective. This is an EMERGENT differentiation from training-dynamics noise and initialization asymmetry (`SlotHeads`' per-head-seeded small random init, `std=1e-3`, confirmed in source), not an engineered one -- `L0` does not actively suppress this differentiation, but it also does not reward it (the loss is defined purely on the pooled, order-invariant `p_bar`).

**5. Head별 KL이 모든 Head를 같은 Teacher에 맞추도록 하는 압력이 실제로 존재하는가?** Yes, by construction -- `L_head = mean_h KL(p_T||p_h)` applies an IDENTICAL, undiluted teacher-matching pressure to every head independently (each head's own softmax is compared to the SAME `p_T`), whereas `L0` only ever supervises the AVERAGE of the 5 heads -- any one head is free to drift away from the teacher as long as the other 4 compensate in the average. The consistently-larger `L_head` gradient magnitudes (Q3) are the direct numerical signature of this stronger, less-diluted pressure.

## Caveat (explicitly required by the spec)

Gradient DIRECTION similarity (cosine 0.71-0.99) does **not** imply the two losses would converge to the same trained model -- `L_head`'s per-head gradients are both larger in magnitude AND, per the sub-1.0 cosine, pointed in a measurably different direction (most visibly on Weather_96), so a full training run under `L_head` (not attempted here, out of this diagnostic-only Phase's scope) could plausibly diverge from `L0`'s trained behavior over many steps even where a single-step gradient comparison shows substantial directional agreement. This diagnostic establishes that a DIFFERENT, non-trivial training signal is available (RQ1 territory for the final verdict), not that training on it would help AggMSE -- that question belongs to PHASE G/the final L1 verdict and would require an actual training run this track does not perform.
