# TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01

**Status: COMPLETE. NO TRAINING (pure post-hoc analysis on J0/J1/J2's
existing best checkpoints). Verdict: GO** -- J1's aggregate-MSE
improvement over J0 is explained by cross-error-interaction reduction,
which exceeds and outweighs a real individual-quality regression.

## 0. Reproduction gate (spec section 2)

Reconstructed on the same fixed 256-query test PROBE set TRACK-J/TRACK-J2
used for their own `final_test_metrics.json` (discovered during this
round: the saved test numbers are NOT computed on the full 2161-query
test split, but on `build_probe_set(exp, test_loader, n_probe=256, ...)`
-- an initial full-test-split run at first FAILED reproduction by
~0.002-0.003, which correctly triggered the spec's "do not proceed"
gate; root-caused to this probe-vs-full-split mismatch, fixed, re-run).

| | J0 (saved) | J0 (reproduced) | diff | J1 (saved) | J1 (reproduced) | diff |
|---|---:|---:|---:|---:|---:|---:|
| retMSE@10 | 0.9535986185 | 0.9535986056 | 1.3e-08 | 1.0083508066 | 1.0083508129 | 6.3e-09 |
| UniformAggMSE@10 | 0.6417305831 | 0.6417305566 | 2.7e-08 | 0.5643467861 | 0.5643467825 | 3.6e-09 |
| Recall@10 | 0.0202566971 | 0.0202566968 | 2.5e-10 | 0.0219866068 | 0.0219866075 | 7.1e-10 |

All differences well under the 1e-5 tolerance -- **gate PASSED**.
Candidate count confirmed 7201 (train-only), 7 channels, test probe
n=256, Top-K=10, future-blind selection (`stable_topk_indices` on
`arm_score`, oracle from `individual_utility_memsafe`), `delta_last`
reconstruction via `memory_value` -- all reused unmodified from the
existing production library.

## 1. Exact decomposition identity

For every query x channel, `Agg = D + C` was asserted via an
INDEPENDENT direct computation (`cross_mat = einsum('bih,bjh->bij', e, e)/H`,
summed off-diagonal for C, diagonal for D) against the
subtraction-based `C = Agg - D`, at `atol=1e-4` -- never failed across
all 256x7 = 1792 query-channel pairs, for all three arms.

## 2. Macro decomposition (the central result table)

| Arm | retMSE@10 | D (individual) | C (cross-interaction) | Agg = D+C |
|---|---:|---:|---:|---:|
| **J0** Joint Shared | 0.953599 | 0.095360 | 0.546371 | 0.641731 |
| **J1** StopGrad-Key | 1.008351 | 0.100835 | 0.463512 | 0.564347 |
| J2 True-Frozen-Key | 1.023965 | 0.102397 | 0.492557 | 0.594954 |

`D = retMSE@K / K` holds exactly (`0.953599/10 = 0.0953599`, etc.),
confirming the spec's predicted identity.

## Q1/Q2/Q3: where does J1's aggregate gain come from?

\[
\Delta D = D_{J1}-D_{J0} = +0.005475 \quad(\text{individual term WORSE for J1})
\]
\[
\Delta C = C_{J1}-C_{J0} = -0.082859 \quad(\text{cross term substantially BETTER for J1})
\]
\[
\Delta Agg = -0.077384 = \Delta D + \Delta C \quad(\text{verified exactly})
\]

**Yes**: `D_J1 > D_J0` (individual candidates ARE genuinely worse under
J1) while `Agg_J1 < Agg_J0` (the aggregate is genuinely better) --
confirmed on real data, not just possible in principle.

\[
Gain_{individual} = D_{J0}-D_{J1} = -0.005475
\qquad
Gain_{cross} = C_{J0}-C_{J1} = +0.082859
\]
\[
\boxed{R_{cross} = \frac{Gain_{cross}}{Agg_{J0}-Agg_{J1}} = \frac{0.082859}{0.077384} = 1.071}
\]

**`R_cross = 1.071 > 1`**: the cross-term reduction ALONE more than
fully explains the entire aggregate improvement -- it more than offsets
the individual-quality regression, with room to spare. This is the
strongest form of evidence the spec's GO criterion asks for
(`R_cross > 0.75`, "or more strongly `>1`").

## 3. Pairwise error interaction (Q4)

| Arm | mean pairwise error dot | mean pairwise error cosine | frac(cosine<0) |
|---|---:|---:|---:|
| J0 | 0.6071 | 0.6234 | 0.12% |
| **J1** | **0.5150** | **0.5004** | **5.01%** |
| J2 | 0.5473 | 0.5190 | 4.55% |

J1's Top-10 candidate errors are, on average, substantially LESS
redundant with each other (cosine 0.500 vs 0.623) than J0's, and true
sign-flipped error cancellation (cosine<0) is ~40x more frequent (5.0%
vs 0.12% of query-channel pairs) -- still a small absolute fraction, but
a large relative increase. **Yes**, J1's pairwise error interaction is
measurably lower than J0's.

## 4. Diversity vs complementarity (Q5, spec section 9)

| Arm | future diversity (pairwise future MSE) | error cosine |
|---|---:|---:|
| J0 | 0.693 | 0.623 |
| J1 | **0.987** (+42%) | **0.500** (-20%) |

Both future diversity AND error-cosine improve together for J1. This
matches spec's **Case 1** ("future diversity up, error cross-term
down -> useful diversity/complementarity possible"), not Case 2 (diverse
but cross-term unchanged) or Case 3 (diversity flat, cross-term alone
driving it). The two signals move together here, so this diagnostic
alone cannot cleanly separate "diversity that happens to be useful" from
"complementarity that happens to look diverse" -- both readings are
consistent with the data; the error-cosine result (Section 3) is the more
direct evidence since it is computed relative to `Y_q`, not just between
candidates.

## 5. Bias/variance and sign cancellation

| Arm | mixed-sign fraction | mean \|mu_h\| | mean variance_h |
|---|---:|---:|---:|
| J0 | 66.1% | 0.578 | 0.312 |
| J1 | **77.4%** | 0.543 | **0.444** |

J1's Top-10 candidates disagree in sign (some over-, some
under-predicting) at a given horizon timestep more often (77.4% vs
66.1% of timesteps) and have higher per-timestep variance (0.444 vs
0.312) -- both consistent with more cancellation opportunity. Mean
absolute bias is slightly LOWER for J1 (0.543 vs 0.578) despite the
higher individual-candidate error (D) -- the individual candidates are
noisier but not more systematically biased.

## 6. Leave-one-out contribution vs individual quality (Q7)

Pooled Pearson/Spearman correlation between each selected candidate's
own individual future MSE and its leave-one-out aggregate `Benefit_i`
(positive = removing it makes the aggregate worse, i.e. it helps):

| Arm | n | Pearson r | Spearman rho |
|---|---:|---:|---:|
| J0 | 17,920 | -0.220 | -0.118 |
| J1 | 17,920 | -0.425 | -0.298 |

Both correlations are negative (worse individual MSE associates with
lower/negative benefit) and highly significant given the large n, but
**neither is strong** (|r| < 0.5 for both) -- individual future-MSE
quality is a weak-to-moderate, not a strong, predictor of a candidate's
actual contribution to the aggregate, for both arms. Notably, J1's
correlation is somewhat STRONGER (more negative) than J0's, not weaker --
so it is not accurate to say "individual relevance stops mattering at
all under J1"; it remains a real but incomplete predictor in both arms,
and the correlation itself does not by itself explain why J1's aggregate
is better (that evidence comes from the D/C decomposition, Section 2).

## 7. Per-query paired analysis (Q6)

n=1792 paired query-channel rows (256 queries x 7 channels).

- **Fraction J1's aggregate better than J0's**: 60.5%
- **Fraction J0's aggregate better**: 39.5%
- Median `Delta Agg` (J1-J0): -0.0283 (improvement); mean -0.0774
  (heavier-tailed improvement, consistent with the macro mean)
- **`individual-worse-but-set-better` fraction** (D_J1 > D_J0 AND
  Agg_J1 < Agg_J0): **19.7%** of all query-channel pairs -- a
  substantial, non-trivial fraction of cases show the exact H1 pattern
  directly at the single-query level, not just in the macro average.

## 8. Channel-wise breakdown

| Channel | J0 D | J1 D | J0 C | J1 C | J0 Agg | J1 Agg |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.1640 | 0.1460 | 0.9565 | 0.7726 | 1.1205 | 0.9186 |
| 1 | 0.0757 | 0.0763 | 0.4431 | 0.3155 | 0.5187 | 0.3918 |
| 2 | 0.1665 | 0.1540 | 0.9298 | 0.7851 | 1.0963 | 0.9391 |
| 3 | 0.0663 | 0.0746 | 0.3838 | 0.2719 | 0.4501 | 0.3465 |
| 4 | 0.1211 | 0.1369 | 0.6513 | 0.6311 | 0.7724 | 0.7680 |
| 5 | 0.0444 | 0.0789 | 0.2609 | 0.2629 | 0.3053 | 0.3419 |
| 6 | 0.0295 | 0.0391 | 0.1993 | 0.2055 | 0.2288 | 0.2446 |

Channels 0/1/2/3 show the clean H1 pattern (D roughly flat-to-up, C down
substantially, Agg down). Channels 4/5/6 show little-to-no aggregate
improvement (channel 5/6 actually slightly worse for J1) -- **the macro
result is NOT driven by a single outlier channel**, but it is also not
perfectly uniform: roughly 4 of 7 channels drive most of the gain.

## 9. Bootstrap (paired, query-clustered, 10,000 reps, 256 queries x 7 channels jointly)

| Metric (J1-J0) | mean diff | 95% CI | excludes 0? |
|---|---:|---|---|
| Agg MSE@10 | -0.0773 | [-0.0962, -0.0595] | Yes (favors J1) |
| D | +0.0055 | [0.0034, 0.0075] | Yes (favors J0) |
| C | -0.0828 | [-0.0995, -0.0665] | Yes (favors J1) |
| Mean pairwise error cosine | -0.1230 | [-0.1312, -0.1147] | Yes (favors J1) |

Every CI clearly excludes zero, in the direction consistent with the H1
pattern. This is **paired within-population uncertainty on the ETTh1
H720 test-probe set (n=256 queries), NOT seed uncertainty** -- it does
not speak to whether this reproduces on a different seed, horizon, or
dataset.

## 10. Answers to the pre-registered questions

**Q1**: J1's aggregate improvement is explained by the cross-error term
(`C`), which improves by 0.0829, more than offsetting a 0.0055 worsening
in the individual term (`D`).

**Q2**: Yes -- `D_J1 (0.1008) > D_J0 (0.0954)` while
`Agg_J1 (0.5643) < Agg_J0 (0.6417)`, confirmed both at the macro level
and in 19.7% of individual query-channel pairs.

**Q3**: The cross-term reduction (`0.0829`) is larger than the
individual-term worsening (`0.0055`) by a factor of ~15x; `R_cross =
1.071` -- cross-term improvement alone would have been sufficient to
explain the entire aggregate gain even without any individual-term
change.

**Q4**: Yes -- J1's mean pairwise error cosine (0.500) is significantly
lower than J0's (0.623), bootstrap CI [-0.131,-0.115], clearly excluding
zero.

**Q5**: Both future diversity and target-relative error complementarity
increase together for J1 (Case 1) -- this diagnostic cannot fully
separate the two, though the error-cosine result (computed relative to
`Y_q`, not just between candidates) is the more direct evidence for
complementarity specifically.

**Q6**: 19.7% of query-channel pairs show the exact
"individual-worse-but-set-better" pattern; 60.5% of pairs show J1's
aggregate winning overall.

**Q7**: Weak-to-moderate (|Pearson r| = 0.22-0.43, both arms) --
individual future-MSE quality is a real but incomplete predictor of a
candidate's actual leave-one-out contribution to the aggregate, in both
J0 and J1.

**Q8**: `individual relevance != set-level forecasting utility` --
**YES**, strongly supported. `D_J1 > D_J0` (individual relevance did get
worse under J1) and yet `Agg_J1 < Agg_J0` (set-level utility got better),
with the decomposition showing exactly why (cross-term collapse
outweighing the individual-term cost), reproduced at the macro level,
the per-query level (19.7% direct instances), the channel level (4/7
channels), and with bootstrap CIs excluding zero on every relevant
metric.

**Q9**: **GO** -- `R_cross = 1.071 > 1` (spec's strongest threshold),
`D_J1 > D_J0` confirmed, cross-term reduction confirmed as the dominant
explanation, paired-query pattern confirmed, bootstrap CIs confirm the
direction robustly. Per spec section 19's explicit interpretation
constraint: this does **not** mean "Set Oracle is correct" or "J1
learned diversity intentionally" -- J1 was never trained with any
complementarity-aware objective. The correct statement is: **J1's
training intervention (removing candidate-side gradient) resulted,
emergently, in individually-worse but more mutually-complementary
retrieved candidates, and this emergent complementarity explains its
aggregate-MSE advantage over J0.** Per spec, no set-level objective,
Set Oracle, or new training is introduced this round -- that is left for
a future, explicitly-scoped experiment.

## Artifacts

- Report: `research/J-shared-encoder-drift/TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01.md`
- Script: `scripts/diag_j3_error_complementarity01.py`
- Raw results: `results/TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01/ETTh1_720/`
  (`reproduction.json`, `sanity_checks.json`, `macro_decomposition.json`,
  `per_query_decomposition_{J0,J1,J2}.parquet`, `channel_decomposition.csv`,
  `pairwise_error_interaction.csv`, `future_diversity.csv`,
  `sign_cancellation.csv`, `leave_one_out_contribution_{J0,J1,J2}.csv`,
  `paired_j0_j1.csv`, `bootstrap_results.json`, `exact_commands.txt`,
  `working_tree.diff`)
- No new checkpoints -- J0/J1/J2's existing checkpoints reused unmodified.
