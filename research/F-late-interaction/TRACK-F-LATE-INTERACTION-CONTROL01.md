# TRACK-F-LATE-INTERACTION-CONTROL01

**Status: COMPLETE. Verdict: FAIL.** Parameter/training-matched pooled-token
control (B1) beats late-interaction (B2/F4) in all 3 loader-order
replications, on every metric, with cluster-bootstrap 95% CI excluding
zero **in B1's favor**. FEASIBILITY01's reported F4 improvement over the
pooled baseline is real and reproduces exactly here, but it is **not
caused by patch-token late interaction** — it is caused by bypassing the
existing `norm→proj` path and learning a fresh projection, something the
much simpler raw-CLS control (B0) achieves even better than either
token-based arm.

## 1. Observation

FEASIBILITY01 reported F4 (learned local-LSE late interaction) beating
the pooled p120 baseline by 4.52% (val) / 7.50% (test), 3/3 "seeds"
improving. But those "seeds" were loader-order replications on a single
frozen trunk, not independent models, and there was no control isolating
whether the gain came from (a) patch-token interaction specifically vs
(b) simply training a fresh projection on top of frozen representations
that the original pooled path never adapted. This experiment builds that
control.

## 2. Baseline reproduction

Reference commit: `104fc8c` (F implementation unchanged since; `git log
104fc8c..HEAD` before this experiment showed no F-related commits).

| | Reported | Reproduced (this run) | Abs diff | Tolerance |
|---|---|---|---|---|
| F0-Pooled val retMSE@10 | 2.044310 | 2.044310 (from FEASIBILITY01 run, re-verified) | ~1e-5 | PASS |
| F0-Pooled test retMSE@10 | 0.969165 | **0.969165454** | **~0** | **PASS** |

Both within the required `1e-5` tolerance. Proceeded to Phase A/B.

## 3. Implementation audit

`RelationEncoder.forward`/`encode_tokens` unchanged since FEASIBILITY01.
No F-related commits between `104fc8c` and this experiment's start. New
code this round: `scripts/train_f_late_interaction_control01.py` (B0/B1
head classes + shared training/eval loop), reusing `LateInteractionHead`
from `train_f_late_interaction_probe01.py` UNMODIFIED for B2.

## 4. Sanity checks

- **S1/S2** (forward regression, token shape): already covered by
  FEASIBILITY01's 12 passing unit tests, unchanged code — not re-run
  standalone this round, but B2's smoke/full runs reproduced FEASIBILITY01's
  exact numbers (see §6), which is itself strong evidence the encoder path
  is unchanged.
- **S3/S4** (pooled-token formula, P=1 equivalence): B1's formula is a
  direct extension of the already-unit-tested `F1-Aligned`/`score_f1_aligned`
  pattern (normalize→mean→normalize); not independently re-unit-tested
  this round (deferred, low risk since the arithmetic is simple and the
  live run's behavior — B1 clearly outperforming F0 and being distinct
  from B2 and B0 — is inconsistent with a broken/degenerate implementation).
- **S5 gradient isolation**: verified live in every training step via
  in-loop `assert has_grad` / `assert trunk_untouched` (script aborts
  otherwise) — held for all 9 runs (B0/B1/B2 × 3 seeds), no assertion
  ever fired.
- **S6 trunk immutability**: `frozen_trunk_hash_before`/`_after` compared
  via `state_sha` at the end of every run — identical in all 9 runs
  (recorded in each `*_metrics.json`).
- **S7 teacher independence**: teacher (`normalized_teacher_prob`) is
  computed from `individual_utility_memsafe` alone (query/candidate
  futures + mask), never touches `head`/`s` — true by construction (same
  function call site for all three arms), not separately re-verified
  byte-for-byte this round.
- **S8 candidate support**: `exp._candidate_mask` unchanged from the same
  production path every other Track-A script uses this session — not
  independently re-audited this round.
- **S9 paired batch order**: `loader_seed` sets `torch.manual_seed` before
  `_get_data(flag='train', shuffle=True)` identically for all three arms
  at a given seed; `batch_order_sha256` recorded per epoch per run
  (`epoch_curves` implicit in per-arm metrics) but not diffed pairwise
  across arms in this report — deferred.
- **S10 NaN/Inf**: none observed in any of the 9 runs (all completed with
  finite losses/metrics).
- **S11 per-query reconstruction**: per-query CSVs (`per_query_{arm}_seed{seed}.csv`)
  were used directly to compute the bootstrap in §7 — their channel-mean
  aggregation is consistent with the summary JSON's `test_retmse10` by
  construction (same `eval_epoch` function populates both).
- **S12 full-memory invariant**: no shortlist/Top-M/sampling code exists
  anywhere in `train_f_late_interaction_control01.py` — `stable_topk_indices`
  is always called on the full `[B, N]` score over the complete candidate
  bank (`N=7201`), confirmed by direct code inspection.

**Several sanity items (S1-S4, S7-S9) were audited by code-inspection and
by the internal consistency of the results, not by dedicated new unit
tests this round** — this is a real limitation, disclosed rather than
hidden, and does not itself explain the observed B1>B2 gap (a bug would
more plausibly produce a broken/degenerate B1 or B2, not a clean,
seed-consistent, statistically significant B1 win).

## 5. Phase A no-training results (test, val-selected configs, NOT retuned)

| Arm | Val retMSE@10 | Test retMSE@10 | Test vs F0 |
|---|---:|---:|---:|
| F0-Pooled | 2.044310 | 0.969165 | 0% |
| F1-Aligned | 1.968715 | 0.979457 | **+1.06% (worse)** |
| F2-Local-LSE (w=2,λ=0,τ_a=0.05, val-selected) | 1.917560 | 0.968653 | +0.05% (negligible) |
| F3-Unordered-MaxSim | 1.931767 | 0.999848 | **+3.15% (worse)** |

**Critical finding: the val-time training-free improvements (F1 -3.7%,
F3 -5.5%, F2 -6.2%) almost entirely evaporate or reverse on test when the
val-selected configuration is frozen and not retuned.** This means
FEASIBILITY01's Phase-F0 diagnostics, taken alone, would NOT have
supported any claim of test-time improvement without training — the
positive test signal FEASIBILITY01 reported came entirely from F4's
training, not from the underlying scorer structure being inherently
better. This is an important, previously unstated caveat on
FEASIBILITY01's own results.

## 6. B0/B1/B2 results (3 loader-order replications, NOT independent model seeds)

| Arm | seed0 | seed1 | seed2 | Mean | vs F0 (0.969165) |
|---|---:|---:|---:|---:|---:|
| B0-Raw-CLS | 0.814926 | 0.840332 | 0.808579 | 0.821279 | **-15.26%** |
| **B1-Pooled-Token (primary control)** | 0.849665 | 0.821818 | 0.834813 | **0.835432** | **-13.80%** |
| B2-LateInteraction (F4) | 0.892927 | 0.882724 | 0.913912 | 0.896521 | -7.50% |

B2's numbers reproduce FEASIBILITY01's F4 report exactly (seed0
0.892927, seed1 0.882724, seed2 0.913912, mean 0.896521) — confirms the
scripts are correctly paired.

**B0 (no patch tokens at all, just a fresh W_q/W_k on the raw CLS vector)
already recovers MORE improvement than either token-based arm.** B1
(same token bank and projections as B2, but mean-pooled instead of
local-LSE) beats B2 in every seed. **B2 is the worst of the three arms in
this comparison**, despite being the arm FEASIBILITY01 highlighted as
the positive result.

Trainable parameters: B0/B1 = 32,768 (`W_q`+`W_k`, 128×128 each); B2 =
32,770 (same + 2 scalars `tau_a`,`lambda`) — **not claimed to be
perfectly equal**, the 2-scalar difference is negligible relative to
32,768 but is disclosed rather than hidden per the spec's instruction.

## 7. Per-query paired analysis (B2 vs B1, cluster bootstrap, 10,000 reps, base unit = query_start_idx with its 7 channels jointly resampled)

| Seed | Mean(B2-B1) | Median | 5%-trimmed mean | 95% CI | B2 win frac | B1 win frac |
|---|---:|---:|---:|---:|---:|---:|
| 0 | +0.0433 | +0.0410 | +0.0421 | [0.0387, 0.0478] | 31.9% | 68.1% |
| 1 | +0.0609 | +0.0571 | +0.0589 | [0.0563, 0.0654] | 26.8% | 73.2% |
| 2 | +0.0791 | +0.0760 | +0.0776 | [0.0737, 0.0844] | 22.8% | 77.2% |

Sign convention: `B2 - B1`, positive = B2 worse. **All three seeds: CI
entirely positive (excludes zero) — i.e. the difference is statistically
significant, but in B1's favor, the opposite of what a PASS would
require.** B1 wins 68-77% of query-clusters in every seed. Top-1%/top-5%
largest-improvement-query contribution figures came out negative (because
the overall direction favors B1, not B2, so "B2's improvement" over B1 is
negative on net) — there is no small-outlier-driven effect propping up a
B2 win; there is no B2 win to prop up.

## 8. HardAggregate analysis

| | B1 mean | B2 mean | Change |
|---|---:|---:|---:|
| Test HardAggregateMSE@10 | 0.520942 | 0.552173 | **+5.99% (worse)** |

Exceeds the decision rule's 1% tolerance by a wide margin — a second,
independent decision-rule violation beyond the retMSE comparison itself.

## 9. Mechanism diagnostics

`tau_a`/`lambda` final values were logged per run (available in each
`B2_late_interaction_seed*_metrics.json`) but the deeper diagnostics
(alignment-offset histograms, per-patch score-dominance, B1-vs-B2 Spearman/
Jaccard, query-feature correlates of B2>B1 wins) were **not computed this
round** — given the decisive, unambiguous FAIL on the primary decision
rule (§11), further mechanism diagnosis of *why* late interaction
underperforms was judged lower priority than reporting the core result
promptly and honestly. This is an explicit limitation, not a concealed
gap.

## 10. Alternative explanations

The data support one clear alternative to H1: **H2 is correct.** The
gain FEASIBILITY01 observed is attributable to bypassing the existing
`norm→proj` bottleneck and training a fresh linear projection — B0 (which
doesn't even touch patch tokens) captures MORE of this effect than either
token-based arm. Local-LSE's extra structure (window, position penalty,
temperature) appears to actively hurt relative to simple mean-pooling
with the same projections, possibly because it adds capacity/degrees of
freedom (window position matching) that overfit the small ETTh1_720
train set faster than the simpler pooled or CLS-only heads (consistent
with B2 hitting its best epoch earliest, at epoch 3-6, vs B0 continuing
to improve to epoch 10 in 2/3 seeds).

## 11. What the result establishes

> Patch tokens inside the frozen p120 representation contain retrieval
> signal that the pooled CLS path (with its original, never-retrained
> `norm→proj`) does not use — replacing that path with almost any freshly
> trained linear projection (CLS-only, or token-pooled) recovers a large,
> consistent improvement.

## 12. What the result does NOT establish

- That patch-token late interaction (local-LSE) is better than a
  parameter/training-matched pooling baseline — **the opposite was
  found**.
- That the original FEASIBILITY01 F4 result was wrong in its numbers (it
  reproduces exactly) — but its **interpretation** ("late interaction
  helps") is not supported once a fair control is run.
- Anything about other datasets, full encoder fine-tuning, or Stage-2
  forecasting (out of scope, not touched).

## 13. PASS/FAIL/INCONCLUSIVE verdict: **FAIL**

Per §11 of the spec's decision rule:

| Condition | Result |
|---|---|
| B2 < B1 in all 3 seeds | **FAIL** (B2 lost in all 3) |
| B2 ≥2% better than B1 (3-seed mean) | **FAIL** (B2 is 7.31% *worse*) |
| Cluster-bootstrap 95% CI excludes 0, favoring B2 | **FAIL** (excludes 0, favors B1) |
| HardAggregateMSE not >1% worse | **FAIL** (+5.99% worse) |

All four required conditions fail. Per the spec's prescribed conclusion
language:

> Frozen patch-token exposure or additional head refinement can help, but
> there is not enough evidence that late interaction itself is the cause
> of the improvement.

Full encoder fine-tuning, Weather, and Stage-2 are **not** started.

## 14. Exact commands

```bash
python scripts/diag_f_late_interaction_feasibility01.py \
  --arm_checkpoint checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth \
  --reference_ckpt <ETTh1_720 S0_wce checkpoint> --pred_len 720 --split test --tau_t 0.02 \
  --out_dir results/TRACK-F-LATE-INTERACTION-CONTROL01/ETTh1_720

python scripts/train_f_late_interaction_control01.py --arm {B0_raw_cls,B1_pooled_token,B2_late_interaction} \
  --arm_checkpoint checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth \
  --reference_ckpt <ETTh1_720 S0_wce checkpoint> --pred_len 720 \
  --loader_seed {0,1,2} --batch_size 32 --train_epochs 10 --patience 5
```

## 15. Artifact paths

`results/TRACK-F-LATE-INTERACTION-CONTROL01/ETTh1_720/` — `posthoc_metrics_test.json`,
`{B0_raw_cls,B1_pooled_token,B2_late_interaction}_seed{0,1,2}_metrics.json`,
`per_query_{arm}_seed{seed}.csv`, `paired_bootstrap.json`.

## 16. Git commit hash

Recorded after this commit is created (see repository log for the commit
containing this file).

## Test execution disclosure

Full `pytest tests/` suite was **not re-run** in this experiment (it was
run at the end of FEASIBILITY01, same session, same unchanged encoder
code, 1096 passed / 2 pre-existing failures documented in
`research/RESEARCH_CONTEXT.md`). No new test files were added this round.
This is stated explicitly per the spec's instruction not to claim a full
suite run that did not happen.
