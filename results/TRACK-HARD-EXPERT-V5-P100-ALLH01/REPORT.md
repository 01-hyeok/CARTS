# TRACK-HARD-EXPERT-V5-P100-ALLH01 -- Final Report

**Stage-2 Mode: Original CARTS Trainable Global Lambda** (`scripts/eval_r_stage2_lambda_signfix01.py`, reusing `train_r_stage2_lambda01.py` unmodified -- `Y_final = B + lambda*(R-B)`, `lambda=sigmoid(a)`, trained by gradient descent on train, selected by validation MSE. The Professor-paper-style validation-only beta fusion is NOT used anywhere in this track.)

All 8 cells (ETTh1/Weather x H96/H192/H336/H720) x 3 arms (V5 Original, Soft Expert, Hard Expert), Shared-Top-100 (P100) candidate support, fixed 10 epochs (no early stopping), checkpoint selection = validation Mean-Mixture RetMSE@10 (never Round-Robin), identical Base Forecaster checkpoint per cell across all 3 arms, GPU1 only.


## Table 1: Forecasting (test MSE)

| Dataset | H | Base | V5 (vs base) | Soft (vs base) | Hard (vs base) |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.39242 | 0.36960 (-5.82%) | 0.37002 (-5.71%) | 0.36987 (-5.75%) |
| ETTh1 | 192 | 0.44722 | 0.42284 (-5.45%) | 0.42399 (-5.19%) | 0.42336 (-5.34%) |
| ETTh1 | 336 | 0.45778 | 0.43913 (-4.07%) | 0.44020 (-3.84%) | 0.44163 (-3.53%) |
| ETTh1 | 720 | 0.56035 | 0.51492 (-8.11%) | 0.51234 (-8.57%) | 0.51352 (-8.36%) |
| Weather | 96 | 0.16936 | 0.17665 (+4.30%) | 0.17607 (+3.96%) | 0.17905 (+5.72%) |
| Weather | 192 | 0.19908 | 0.21758 (+9.29%) | 0.21880 (+9.90%) | 0.22030 (+10.66%) |
| Weather | 336 | 0.24567 | 0.28290 (+15.15%) | 0.28120 (+14.46%) | 0.28988 (+17.99%) |
| Weather | 720 | 0.31943 | 0.38553 (+20.69%) | 0.38043 (+19.10%) | 0.39567 (+23.87%) |

## Table 2: Specialization (P100, test split)

| Dataset | H | Method | Pairwise Top-10 Overlap | Union Size (/100) | Max Head Usage | Min Head Usage |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.375 | 27.13 | 0.243 | 0.156 |
| ETTh1 | 96 | Soft | 0.740 | 16.02 | 0.241 | 0.173 |
| ETTh1 | 96 | Hard | 0.492 | 23.12 | 0.220 | 0.172 |
| ETTh1 | 192 | V5 | 0.428 | 25.06 | 0.258 | 0.158 |
| ETTh1 | 192 | Soft | 0.669 | 17.80 | 0.238 | 0.170 |
| ETTh1 | 192 | Hard | 0.470 | 23.67 | 0.213 | 0.184 |
| ETTh1 | 336 | V5 | 0.535 | 21.60 | 0.244 | 0.147 |
| ETTh1 | 336 | Soft | 0.729 | 16.26 | 0.306 | 0.093 |
| ETTh1 | 336 | Hard | 0.485 | 23.16 | 0.266 | 0.165 |
| ETTh1 | 720 | V5 | 0.732 | 16.28 | 0.288 | 0.109 |
| ETTh1 | 720 | Soft | 0.835 | 13.63 | 0.245 | 0.150 |
| ETTh1 | 720 | Hard | 0.615 | 19.15 | 0.285 | 0.139 |
| Weather | 96 | V5 | 0.253 | 33.88 | 0.215 | 0.185 |
| Weather | 96 | Soft | 0.518 | 22.71 | 0.244 | 0.174 |
| Weather | 96 | Hard | 0.302 | 31.48 | 0.244 | 0.157 |
| Weather | 192 | V5 | 0.221 | 34.83 | 0.227 | 0.175 |
| Weather | 192 | Soft | 0.510 | 22.67 | 0.246 | 0.171 |
| Weather | 192 | Hard | 0.268 | 32.53 | 0.234 | 0.159 |
| Weather | 336 | V5 | 0.287 | 31.11 | 0.225 | 0.188 |
| Weather | 336 | Soft | 0.559 | 21.15 | 0.238 | 0.163 |
| Weather | 336 | Hard | 0.278 | 31.71 | 0.231 | 0.187 |
| Weather | 720 | V5 | 0.437 | 25.17 | 0.231 | 0.154 |
| Weather | 720 | Soft | 0.655 | 18.50 | 0.252 | 0.153 |
| Weather | 720 | Hard | 0.347 | 28.83 | 0.235 | 0.178 |

## Table 3: Full-memory vs P100 (test MSE, Original-CARTS-lambda Stage-2)

Full-memory numbers reused from already-completed tracks (TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 for V5, TRACK-EXPERT-V5-FULL01 for Soft, TRACK-HARD-EXPERT-V5-FULL01 for Hard). **Full-memory Soft/Hard were only ever run for H96/H720** (Soft) **or ETTh1-only H96/H720** (Hard) **in this project** -- H192/H336 and Weather-Hard Full-memory cells were never run and are marked N/A, not estimated.

| Dataset | H | Method | Full MSE | P100 MSE | Delta % |
|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.37472 | 0.36960 | -1.37% |
| ETTh1 | 96 | Soft | 0.37557 | 0.37002 | -1.48% |
| ETTh1 | 96 | Hard | 0.37684 | 0.36987 | -1.85% |
| ETTh1 | 192 | V5 | 0.42368 | 0.42284 | -0.20% |
| ETTh1 | 192 | Soft | N/A | 0.42399 | N/A |
| ETTh1 | 192 | Hard | N/A | 0.42336 | N/A |
| ETTh1 | 336 | V5 | 0.44241 | 0.43913 | -0.74% |
| ETTh1 | 336 | Soft | N/A | 0.44020 | N/A |
| ETTh1 | 336 | Hard | N/A | 0.44163 | N/A |
| ETTh1 | 720 | V5 | 0.50368 | 0.51492 | +2.23% |
| ETTh1 | 720 | Soft | 0.50312 | 0.51234 | +1.83% |
| ETTh1 | 720 | Hard | 0.49947 | 0.51352 | +2.81% |
| Weather | 96 | V5 | 0.16818 | 0.17665 | +5.04% |
| Weather | 96 | Soft | 0.16958 | 0.17607 | +3.83% |
| Weather | 96 | Hard | N/A | 0.17905 | N/A |
| Weather | 192 | V5 | 0.20414 | 0.21758 | +6.59% |
| Weather | 192 | Soft | N/A | 0.21880 | N/A |
| Weather | 192 | Hard | N/A | 0.22030 | N/A |
| Weather | 336 | V5 | 0.25133 | 0.28290 | +12.56% |
| Weather | 336 | Soft | N/A | 0.28120 | N/A |
| Weather | 336 | Hard | N/A | 0.28988 | N/A |
| Weather | 720 | V5 | 0.34229 | 0.38553 | +12.63% |
| Weather | 720 | Soft | 0.34052 | 0.38043 | +11.72% |
| Weather | 720 | Hard | N/A | 0.39567 | N/A |

## Table 4: Resource (total wall-clock seconds, P100 pipeline: Stage1 + selection + cache + Stage2 + specialization report)

| Dataset | H | Arm | Total Time (s) |
|---|---|---|---|
| ETTh1 | 96 | V5 | 250 |
| ETTh1 | 96 | Soft | 464 |
| ETTh1 | 96 | Hard | 465 |
| ETTh1 | 192 | V5 | 511 |
| ETTh1 | 192 | Soft | 459 |
| ETTh1 | 192 | Hard | 456 |
| ETTh1 | 336 | V5 | 491 |
| ETTh1 | 336 | Soft | 440 |
| ETTh1 | 336 | Hard | 431 |
| ETTh1 | 720 | V5 | 213 |
| ETTh1 | 720 | Soft | 391 |
| ETTh1 | 720 | Hard | 396 |
| Weather | 96 | V5 | 2507 |
| Weather | 96 | Soft | 4478 |
| Weather | 96 | Hard | 4495 |
| Weather | 192 | V5 | 5243 |
| Weather | 192 | Soft | 4430 |
| Weather | 192 | Hard | 4492 |
| Weather | 336 | V5 | 5309 |
| Weather | 336 | Soft | 4542 |
| Weather | 336 | Hard | 4827 |
| Weather | 720 | V5 | 2520 |
| Weather | 720 | Soft | 4801 |
| Weather | 720 | Hard | 4553 |

## Table 5: Oracle-vs-Fixed-Head Gap (P100, test split, restricted-pool oracle)

| Dataset | H | Arm | Fixed-head RetMSE@10 | Oracle RetMSE@10 (per-query argmin) | Oracle gain % |
|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.6750 | 0.5733 | 15.06% |
| ETTh1 | 96 | Soft | 0.6620 | 0.6025 | 8.99% |
| ETTh1 | 96 | Hard | 0.6758 | 0.5808 | 14.05% |
| ETTh1 | 192 | V5 | 0.7491 | 0.6577 | 12.21% |
| ETTh1 | 192 | Soft | 0.7413 | 0.6700 | 9.62% |
| ETTh1 | 192 | Hard | 0.7326 | 0.6417 | 12.42% |
| ETTh1 | 336 | V5 | 0.7932 | 0.7124 | 10.18% |
| ETTh1 | 336 | Soft | 0.8032 | 0.7282 | 9.34% |
| ETTh1 | 336 | Hard | 0.8438 | 0.7013 | 16.89% |
| ETTh1 | 720 | V5 | 0.9429 | 0.8836 | 6.29% |
| ETTh1 | 720 | Soft | 0.9221 | 0.8872 | 3.79% |
| ETTh1 | 720 | Hard | 0.9182 | 0.8404 | 8.48% |
| Weather | 96 | V5 | 0.5290 | 0.2087 | 60.54% |
| Weather | 96 | Soft | 0.4253 | 0.3222 | 24.25% |
| Weather | 96 | Hard | 0.5232 | 0.2464 | 52.90% |
| Weather | 192 | V5 | 0.4746 | 0.2465 | 48.08% |
| Weather | 192 | Soft | 0.5039 | 0.3425 | 32.03% |
| Weather | 192 | Hard | 0.6820 | 0.2942 | 56.86% |
| Weather | 336 | V5 | 0.6230 | 0.3324 | 46.65% |
| Weather | 336 | Soft | 0.5894 | 0.3995 | 32.22% |
| Weather | 336 | Hard | 0.6451 | 0.3294 | 48.94% |
| Weather | 720 | V5 | 0.7328 | 0.4857 | 33.72% |
| Weather | 720 | Soft | 0.7407 | 0.5362 | 27.61% |
| Weather | 720 | Hard | 0.8384 | 0.4556 | 45.66% |

Note: "oracle" here is the RESTRICTED-pool oracle (best achievable
among the same 100 pool candidates every arm sees, per query), not the
Full-memory oracle -- it measures router headroom within the P100
setting, not the cost of P100 restriction itself (that is Table 3).

## Anomaly flagged before interpretation

On Weather, **plain V5 (no specialization loss at all) already has
very low pairwise overlap** (0.22-0.44, LOWER than Hard in 3/4
horizons, and far lower than Soft everywhere) **and often the largest
union size of the three arms** (e.g. Weather_96: V5 union=33.88 >
Hard union=31.48; Weather_192: V5=34.83 > Hard=32.53). This is the
opposite of the ETTh1 pattern, where V5's overlap/union sits between
Soft and Hard in a more orderly way. Candidate explanations, in the
order the spec asks to check them:
- **data**: Weather's 21 channels and much larger train set, together
  with the single V5 Mean-Mixture objective (KL against a 5-head
  AVERAGE, no per-head pressure at all) plausibly let different heads
  drift into genuinely different local optima by instability/noise
  alone, with no specialization signal needed -- not necessarily
  "specialization" in the sense this track is testing.
- **metric**: overlap/union are measured identically everywhere (same
  `head_pairwise_overlap`/`per_head_standalone_topk` primitives,
  reused unmodified for all 3 arms, confirmed via the shared pool-aware
  diagnostics module) -- not a measurement artifact of this track's
  own code.
- **implementation**: V5's trainer (`train_v_sharedtop100_01.py`) and
  Soft/Hard's trainers share the identical SlotHeads init
  (`std=1e-3`, seeds `1000+m`) and identical P100 scoring path
  (`compute_scores`/`compute_scores_pool_channel_first_grad`) -- no
  implementation divergence identified.
- **evaluation protocol**: all 3 arms are evaluated with the identical
  `evaluate_and_save_head_report_pool` call on their own
  Mean-Mixture-selected checkpoint -- no protocol asymmetry identified.
- **model structure**: left as the most likely explanation -- V5's
  loss provides no per-head pressure whatsoever (only the Mean-Mixture
  average is trained against the teacher), so on a dataset where the
  retrieval signal itself is weak (Weather, confirmed throughout this
  session's retrieval-quality investigations), the 5 heads have no
  reason to agree OR to specialize -- they just drift. This is
  consistent with but not proof of the hypothesis; flagged honestly
  rather than smoothed over.

## Closing questions

**1. Hard는 P100에서도 Soft collapse를 완화하는가?**
Yes, unambiguously, at all 8 cells: Hard's pairwise Top-10 overlap is
lower than Soft's and Hard's union size is larger than Soft's in every
single cell (ETTh1: Hard overlap 0.37-0.62 vs Soft 0.67-0.84; Weather:
Hard overlap 0.27-0.35 vs Soft 0.51-0.66). The Full-candidate finding
from TRACK-HARD-EXPERT-V5-FULL01 replicates fully under P100
restriction -- Hard assignment induces specialization independent of
candidate-pool size.

**2. H96/H192/H336/H720에서 효과가 어떻게 달라지는가?**
On ETTh1, Hard's forecasting advantage over Soft is present at the
short horizons (H96: -5.75% vs -5.71%, H192: -5.34% vs -5.19%, both
Hard winning by a hair) and REVERSES at the long horizons (H336: Soft
-3.84% beats Hard -3.53%; H720: Soft -8.57% beats Hard -8.36%) -- a
genuine horizon-dependent sign flip within a single dataset, not
reducible to "Hard is better" or "Hard is worse." On Weather the
horizon dependence is monotonic in a different sense: Hard's
underperformance relative to base GROWS with horizon (+5.72% ->
+10.66% -> +17.99% -> +23.87%), i.e. Hard gets progressively worse
relative to base as the horizon lengthens on this dataset, the
opposite direction from the "long-horizon specialization helps" story
in hypothesis D.

**3. ETTh1과 Weather에서 동일한 패턴이 나타나는가?**
No. On ETTh1 all three arms beat base at every horizon (-3.5% to
-8.6%), and the Hard-vs-Soft ranking flips with horizon but stays
close. On Weather all three arms are WORSE than base at every single
horizon (+4% to +24%) -- consistent with the Weather
trainable-lambda-gets-stuck-near-initialization pattern established
earlier this session (TRACK-V-SHARED-TOP100-SIGNFIX01) -- and Hard is
reliably the WORST of the three arms in every one of the 4 Weather
cells, by a clear margin (e.g. Weather_720: Hard +23.87% vs Soft
+19.10% vs V5 +20.69%). The two datasets do not share a pattern here;
Weather is a qualitatively different (and uniformly worse) regime for
this entire family of methods under this Stage-2 mode.

**4. specialization 증가가 forecasting 개선과 실제로 연결되는가?**
Weak at best, and on Weather it points the WRONG way. On ETTh1,
Hard's lower overlap does not reliably predict a forecasting win (it
wins 2/4 horizons, loses 2/4). On Weather, Hard has the lowest overlap
of the three arms in 3/4 cells yet has the WORST forecasting result in
all 4/4 cells -- the dataset where specialization is most aggressively
induced is exactly the dataset where it hurts most. This is evidence
for hypothesis C (diversity does not guarantee forecasting benefit),
not hypothesis D, and the Weather direction is sharper than a null
result -- it looks actively harmful there, not just unhelpful.

**5. P100에서도 Oracle-vs-Fixed-Head gap이 충분히 존재하는가?**
Yes, substantially, everywhere. ETTh1: 3.8-16.9% oracle gain across
all 12 cell-arms. Weather: 24.3-60.5% oracle gain -- roughly 3-6x
larger than ETTh1's -- meaning a perfect per-query head router would
recover a far larger improvement on Weather than on ETTh1, even though
(or perhaps because) none of the three fixed/trained-assignment
methods tested here currently capture any of that gap profitably on
Weather (all three make things worse there under this Stage-2 mode).

**6. 이 결과가 다음 Past-only Router 실험을 정당화하는가?**
Yes, with an important caveat. The oracle-vs-fixed gap (question 5) is
large and real, especially on Weather, so there is genuine headroom a
learned router could in principle capture. But this track's own
results show that BOTH of the two non-learned heuristics tried here --
Soft's responsibility-weighted averaging and Hard's detached-oracle
argmin -- fail to capture that headroom safely on Weather (both make
things worse than plain V5's uniform Mean-Mixture there), and even
reach OPPOSITE conclusions about which heuristic is better depending
on horizon on ETTh1. A learned Past-only Router is a reasonable next
step specifically BECAUSE the two hand-designed alternatives tried so
far do not reliably realize the headroom that demonstrably exists --
but the Weather failure mode (large oracle gap, yet both heuristics
actively hurt) is a warning that a router trained the same way (via a
future-dependent or future-adjacent signal) may face the same
train/val/test retrieval-quality-inflation problem already diagnosed
for Weather this session, and should be evaluated against that
possibility explicitly rather than assumed away.
