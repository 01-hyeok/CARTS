# TRACK-L-EVAL-ALIGNMENT-FULLTEST01

**Status: COMPLETE. NO TRAINING (pure re-evaluation of J0/J1/K1/K2's
existing, unchanged checkpoints on two aligned populations).** Fixes a
real evaluation-population mismatch discovered after TRACK-K: J0/J1's
saved test numbers came from a fixed 256-query probe (P256), K1/K2's
came from the full 2161-query test split (FULL2161) -- different
populations, so the original "+11.16%" K2-vs-J1 comparison was invalid.
This track evaluates all four arms on BOTH populations with one shared
evaluator. **Result: both prior conclusions are CONFIRMED, not
overturned, on the corrected full-to-full comparison** -- TRACK-J3's
complementarity finding is FULL-TEST CONFIRMED, and TRACK-K's NO-GO is
CONFIRMED (the aligned degradation is 11.48%, if anything slightly worse
than the original mismatched 11.16%).

## 0. Checkpoint audit and reproduction gates

See `research/L-eval-alignment/AUDIT.md` for the full checkpoint audit
(hashes, best epochs, confirmed-identical config across all four arms)
and the root-cause explanation of the original population mismatch.

All four reproduction gates passed at ~1e-8 (well under the 1e-5
requirement):

| Gate | retMSE diff | Agg diff | Recall diff |
|---|---:|---:|---:|
| J0 @ P256 | 1.3e-08 | 2.7e-08 | 2.5e-10 |
| J1 @ P256 | 6.3e-09 | 3.6e-09 | 7.1e-10 |
| K1 @ FULL2161 | 1.1e-08 | 4.9e-10 | 3.6e-10 |
| K2 @ FULL2161 | 7.9e-09 | 2.5e-09 | 1.8e-10 |

**Q1/Q2: YES** -- both existing-value reproductions pass exactly.

## 1. The central table: all four arms, both populations

### P256 (n=256 queries x 7 channels)

| Arm | retMSE@10 | D | C | AggMSE@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| J0 | 0.953599 | 0.095360 | 0.546371 | 0.641731 | 0.020257 |
| J1 | 1.008351 | 0.100835 | 0.463512 | 0.564347 | 0.021987 |
| K1 | 1.142388 | 0.114239 | 0.457915 | 0.572154 | 0.019754 |
| K2 | 1.111587 | 0.111159 | 0.432881 | 0.544039 | 0.026060 |

### FULL2161 (n=2161 queries x 7 channels) -- the primary, corrected table

| Arm | retMSE@10 | D | C | AggMSE@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| J0 | 0.955397 | 0.095540 | 0.548835 | 0.644375 | 0.022377 |
| J1 | 1.005504 | 0.100550 | 0.463476 | 0.564026 | 0.021855 |
| K1 | 1.139629 | 0.113963 | 0.459787 | 0.573750 | 0.020017 |
| K2 | 1.120915 | 0.112091 | 0.434852 | 0.546943 | 0.027461 |

`Agg = D + C` verified exactly (atol=1e-4) for every one of the
(256+2161) x 7 channels x 4 arms query-channel rows, never failed.

## 2. J3 complementarity: FULL2161 re-verification (Q3-Q7)

| | J0 | J1 | Delta (J1-J0) |
|---|---:|---:|---:|
| D | 0.095540 | 0.100550 | **+0.005005** |
| C | 0.548835 | 0.463476 | **-0.085359** |
| Agg | 0.644375 | 0.564026 | -0.080349 (= Delta D + Delta C, exact) |

**Q3 (D_J1 > D_J0 on FULL)**: **YES** (0.100550 > 0.095540).
**Q4 (C_J1 < C_J0 on FULL)**: **YES** (0.463476 < 0.548835).
**Q5 (Agg_J1 < Agg_J0 on FULL)**: **YES** (0.564026 < 0.644375).

**Q6**: \(R_{cross} = \frac{C_{J0}-C_{J1}}{Agg_{J0}-Agg_{J1}} =
\frac{0.085359}{0.080349} = \mathbf{1.0624}\) on FULL2161 (vs 1.071 on
P256 -- nearly identical).

Additional FULL2161-only diagnostics: mean pairwise error cosine
J0=0.6215 vs J1=(0.6215-0.1238=)0.4977 (bootstrap-confirmed below);
individual-worse-but-set-better fraction = **20.9%** of all 15,127
query-channel pairs (vs 19.7% on P256 -- consistent); fraction of pairs
where J1's aggregate is better than J0's = **60.6%** (vs 60.5% on P256).

**Bootstrap (query-clustered, 10,000 reps, all 2161 FULL2161 queries x 7
channels jointly)**:

| Metric (J1-J0) | mean diff | 95% CI | excludes 0? |
|---|---:|---|---|
| Agg MSE@10 | -0.0803 | [-0.0866, -0.0741] | Yes (favors J1) |
| D | +0.0050 | [0.0043, 0.0057] | Yes (favors J0) |
| C | -0.0853 | [-0.0910, -0.0797] | Yes (favors J1) |
| Mean pairwise error cosine | -0.1238 | [-0.1266, -0.1209] | Yes (favors J1) |

**Q7: J3 verdict = FULL-TEST CONFIRMED.** All four pre-registered H1-H4
conditions hold on the full 2161-query population, with bootstrap CIs
excluding zero on every relevant metric, at nearly identical magnitude
to the original P256-only result. The complementarity finding is not a
probe artifact.

## 3. TRACK-K re-verification: the corrected K2-vs-J1 comparison (Q8-Q10)

\[
\Delta_{ret} = \frac{retMSE_{K2}-retMSE_{J1}}{retMSE_{J1}}\times100
= \frac{1.120915-1.005504}{1.005504}\times100 = \mathbf{+11.48\%}
\]

(computed FULL-to-FULL, replacing the invalid original mismatched-
population "+11.16%" -- these are numerically close but the new value
is the only valid one, and this closeness is itself informative: it
shows the original conclusion's DIRECTION was not an artifact of the
population mismatch, even though the specific number must be replaced.)

**Q8: +11.48%.**
**Q9: YES, exceeds the pre-registered 10% threshold** (11.48% > 10%).
**Q10: TRACK-K verdict = CONFIRMED.** The original NO-GO stands on the
corrected, apples-to-apples comparison -- if anything the aligned
degradation is very slightly LARGER than what was originally (invalidly)
reported.

## 4. K2's aggregate improvement over J1 -- both populations (Q11, Q12)

\[
\Delta Agg_{K2-J1}^{FULL} = \frac{0.546943-0.564026}{0.564026}\times100 = -3.03\%
\]
\[
\Delta Agg_{K2-J1}^{P256} = \frac{0.544039-0.564347}{0.564347}\times100 = -3.60\%
\]

**Q11: YES** (K2's aggregate MSE is 3.03% better than J1's on FULL2161).
**Q12: YES** (3.60% better on P256). K2's aggregate advantage over J1 is
real and consistent across both populations -- this part of TRACK-K's
finding is NOT an artifact of the population mismatch. It is specifically
the **individual-quality cost** (Section 3) that triggers the
pre-registered NO-GO, not the aggregate result itself.

## 5. Ranking stability (Q13)

| | P256 | FULL2161 | Stable? |
|---|---|---|---|
| retMSE order (best->worst) | J0 < J1 < K2 < K1 | J0 < J1 < K2 < K1 | **Identical** |
| AggMSE order (best->worst) | K2 < J1 < K1 < J0 | K2 < J1 < K1 < J0 | **Identical** |
| Recall order (best->worst) | K2 > J1 > J0 > K1 | K2 > J0 > J1 > K1 | J0/J1 swap (2nd/3rd), K2 best and K1 worst in both |

**Q13: Very high.** Two of three rankings (retMSE, AggMSE) are byte-
identical between populations; Recall@10's ranking is stable at the
extremes (K2 best, K1 worst in both) with only a minor swap between J0
and J1 in the middle -- expected given Recall@10's much smaller absolute
magnitude and correspondingly higher relative sampling variance (see
Section 6). The P256 probe was, in practice, a reasonably faithful
predictor of arm ranking despite the invalid direct magnitude comparison
its numbers were originally used for.

## 6. Probe representativeness

| Arm | Metric | P256 | FULL2161 | Relative diff |
|---|---|---:|---:|---:|
| J0 | retMSE | 0.953599 | 0.955397 | -0.19% |
| J0 | Agg | 0.641731 | 0.644375 | -0.41% |
| J0 | Recall | 0.020257 | 0.022377 | **-9.48%** |
| J1 | retMSE | 1.008351 | 1.005504 | +0.28% |
| J1 | Agg | 0.564347 | 0.564026 | +0.06% |
| J1 | Recall | 0.021987 | 0.021855 | +0.60% |
| K1 | retMSE | 1.142388 | 1.139629 | +0.24% |
| K1 | Agg | 0.572154 | 0.573750 | -0.28% |
| K1 | Recall | 0.019754 | 0.020017 | -1.31% |
| K2 | retMSE | 1.111587 | 1.120915 | -0.83% |
| K2 | Agg | 0.544039 | 0.546943 | -0.53% |
| K2 | Recall | 0.026060 | 0.027461 | **-5.10%** |

retMSE, AggMSE, and C are all within ~1% between populations for every
arm -- P256 is a good estimator of these. Recall@10 is noticeably
noisier (up to -9.5% relative for J0) -- expected, since Recall@10 is a
much smaller-magnitude, higher-variance statistic (fraction of exact
Top-10 overlaps) that benefits more from the larger FULL2161 sample.
**Practical implication for future work**: P256 is adequate for
retMSE/AggMSE-based screening decisions, but Recall@10 comparisons
should prefer the full test split when precision matters.

## 7. Per-channel FULL2161 (does the J3 pattern hold per-channel?)

| Channel | J0 D | J1 D | J0 C | J1 C | J0 Agg | J1 Agg |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.1632 | 0.1464 | 0.9537 | 0.7791 | 1.1169 | 0.9254 |
| 1 | 0.0765 | 0.0768 | 0.4474 | 0.3200 | 0.5239 | 0.3967 |
| 2 | 0.1667 | 0.1540 | 0.9364 | 0.7900 | 1.1031 | 0.9440 |
| 3 | 0.0672 | 0.0763 | 0.3908 | 0.2844 | 0.4581 | 0.3607 |
| 4 | 0.1205 | 0.1359 | 0.6480 | 0.6271 | 0.7685 | 0.7629 |
| 5 | 0.0451 | 0.0754 | 0.2651 | 0.2407 | 0.3102 | 0.3161 |
| 6 | 0.0295 | 0.0392 | 0.2004 | 0.2031 | 0.2299 | 0.2423 |

(from `results/TRACK-L-EVAL-ALIGNMENT-FULLTEST01/ETTh1_720/j3_full_decomposition/channel.csv`)

Same pattern as the original P256-only channel breakdown: channels 0/1/2/3
show the clean H1 pattern (C drops substantially, Agg improves clearly);
channels 4/5/6 show weak-to-mixed effects (channel 5's Agg is actually
slightly worse for J1). The macro result is driven by roughly 4 of 7
channels in both populations -- **channel pattern confirmed stable**.

## 8. K1 vs K2 (unaffected by the original mismatch, re-verified anyway)

| | K1 (FULL) | K2 (FULL) | Delta |
|---|---:|---:|---:|
| retMSE | 1.139629 | 1.120915 | -0.018714 (K2 better) |
| D | 0.113963 | 0.112091 | -0.001871 (K2 better) |
| C | 0.459787 | 0.434852 | -0.024935 (K2 better) |
| Agg | 0.573750 | 0.546943 | -0.026807 (K2 better) |
| Recall | 0.020017 | 0.027461 | +0.007444 (K2 better) |

K2 beats K1 on every metric, exactly as originally reported (this
comparison was always same-population, so it was never affected by the
mismatch) -- confirmed again here for completeness.

## 9. Revised decision framework outcome

Per the spec's pre-registered scenarios (Section 25): **Scenario B**
applies exactly -- J3 is CONFIRMED on FULL2161 (Section 2), AND
`retMSE_{K2} > 1.10 * retMSE_{J1}` (1.1209 > 1.1061 = 1.10x1.0055) on
FULL2161. Therefore:

\[
\boxed{\text{TRACK-J3: CONFIRMED} \qquad \text{TRACK-K: NO-GO CONFIRMED}}
\]

The set-level individual-vs-aggregate misalignment TRACK-J3 identified is
real and survives full-test evaluation. The multi-slot surrogate tested
in TRACK-K (K2, at `lambda_agg=1.0`, `beta=0.05`, `L_soft=32`) recovers a
genuine, population-independent aggregate-quality improvement over J1
(~3% better on both P256 and FULL2161) and the best Recall@10 of any arm,
but at an individual-quality cost (~11.5%) that exceeds the
pre-registered acceptable budget on the now-correctly-aligned
comparison. This is not a reversal of either prior conclusion -- it is
confirmation that neither conclusion was an artifact of the evaluation
mismatch, now on solid, apples-to-apples footing.

## 10. Interpretation-rule compliance

Both verdicts above are re-confirmed, not re-decided from scratch --
this track's purpose was solely to remove the population confound, not
to re-litigate the underlying findings. No coefficient sweep, additional
seed, Weather, Stage-2, checkpoint re-selection, or new training was
performed, per spec's explicit prohibition.

## Artifacts

- Audit: `research/L-eval-alignment/AUDIT.md`
- Report: `research/L-eval-alignment/TRACK-L-EVAL-ALIGNMENT-FULLTEST01.md`
- Script: `scripts/eval_l_aligned_population01.py` (single shared
  evaluator for all four arms, both populations)
- Unit tests: `tests/test_l_eval_alignment_fulltest01.py` (14 tests, all pass)
- Raw results: `results/TRACK-L-EVAL-ALIGNMENT-FULLTEST01/ETTh1_720/`
  (`checkpoint_audit.json`, `reproduction_gates.json`, `P256/`,
  `FULL2161/`, `j3_full_decomposition/`, `population_comparison.csv`,
  `revised_verdict.json`, `exact_commands.txt`, `working_tree.diff`)
- No new checkpoints -- J0/J1/K1/K2's existing checkpoints reused,
  unmodified, never re-selected.
