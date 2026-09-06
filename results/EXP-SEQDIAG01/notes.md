# EXP-SEQDIAG01 -- factual notes (final, 2026-09-06)

Status: **COMPLETE.** All 4 arms (ETTh1 Trainable [reused], ETTh1 Frozen-B0,
Weather Trainable, Weather Frozen-B0) have Stage-2 results. Full entry:
`research/EXPERIMENT_LOG.md` -> `## EXP-SEQDIAG01`. This file is the
supporting working notes; the log entry is the canonical factual record.

## Design

Frozen-B0 control arm (B0's own encoder weights, never updated; only the new
`SetConditioner` trains) vs. the existing/new Trainable arm, on both ETTh1
and Weather, H96, to separate:

- **H1**: the sequential full-memory cross-entropy objective collapses the
  encoder representation, and that collapse is what prevents generalization.
- **H2**: even with the representation held stable, past-only information
  does not let this mechanism learn a discrete greedy set-construction rule
  that transfers from training queries to held-out ones.

## Stage-2 results table

| Arm | Dataset | b0_unforced_mse | sequential_forced_mse | gap_recovery | seq_set_recall@10 |
|---|---|---:|---:|---:|---:|
| Trainable (reused, EXP-SEQFULL01) | ETTh1 | 0.37312 | 0.40277 | -0.404 | 0.01354 |
| Frozen-B0 (Arm B) | ETTh1 | 0.373122 | 0.387924 | -0.248 | 0.01242 |
| Trainable (Arm C) | Weather | 0.175529 | 0.268140 | -0.816 | 0.00917 |
| Frozen-B0 (Arm D) | Weather | 0.175529 | 0.217331 | -0.608 | 0.02293 |

`duplicate_rate` / `invalid_rate` are 0.0 on every arm -- the sequential
selector is not cheating via duplicates or invalid picks, it is simply not
recovering the teacher's set membership.

Source files: `results/EXP-SEQFULL01/metrics.csv`,
`results/EXP-SEQDIAG01/armB_etth1_frozen_stage2.json`,
`results/EXP-SEQDIAG01/armC_weather_trainable_stage2.json`,
`results/EXP-SEQDIAG01/armD_weather_frozen_stage2.json`.

## Effective-rank diagnostic (post-hoc, `effective_rank_diagnostic.py` in
this directory; uses `utils.rank_losses.embedding_geometry`, full candidate
bank, channel 0 -- **not** logged by the training script itself)

| Encoder | Dataset | n_candidates | effective_rank (of 128) | mean pairwise cosine |
|---|---|---:|---:|---:|
| B0 | ETTh1 | 8449 | ~20.8 (EXP-3 closure figure, not recomputed here) | UNKNOWN |
| Trainable (EXP-SEQFULL01) | ETTh1 | 8449 | 5.433 | 0.0122 |
| B0 | Weather | 36696 | 19.576 | 0.8123 |
| Trainable (Arm C) | Weather | 36696 | 4.297 | 0.7063 |

Weather's trained encoder collapses by ~78% (19.576 -> 4.297), ETTh1's by
~74% (20.8 -> 5.433) -- consistent relative-magnitude collapse across both
datasets, independent evidence alongside the `gap_recovery` pattern below.
Note mean pairwise cosine does **not** move the same direction on Weather
(B0's own cosine, 0.812, is already high; the trained arm's, 0.706, is
*lower*) -- cosine alone is not a reliable collapse indicator on Weather;
effective rank is the metric that agrees in direction across both datasets.
Frozen arms (B, D) trivially have B0's own effective rank/cosine by
construction (weights never updated) -- no separate computation needed.

## HardAggregateMSE@10 (post-hoc, unweighted aggregate; `compute_hard_aggregate_mse_seqdiag01.py`)

`= MSE(mean_i y_i, y_q)`, equal-weight, distinct from `A_weighted`'s
B0-weighted aggregate above. Not computed by the production eval script.

| Arm | Dataset | individual_oracle | set_oracle | sequential | B0 (unforced) | seq/b0 |
|---|---|---:|---:|---:|---:|---:|
| Trainable (reused) | ETTh1 | 0.19101 | 0.30503 | 0.59017 | 0.40741 | 1.449x |
| Frozen-B0 (Arm B) | ETTh1 | 0.19101 | 0.30503 | 0.51901 | 0.40741 | 1.274x |
| Trainable (Arm C) | Weather | 0.03631 | 0.12417 | 4.30396 | 0.23421 | 18.38x |
| Frozen-B0 (Arm D) | Weather | 0.03631 | 0.12417 | 0.92057 | 0.23421 | 3.93x |

Larger freeze effect than `gap_recovery` showed, especially on Weather:
freezing cuts the sequential arm's unweighted aggregate MSE by 4.7x on
Weather (4.304 -> 0.921) vs. only 1.14x on ETTh1 (0.590 -> 0.519). Both
datasets move the same direction (freezing helps) -- a third, independent
corroborating metric alongside `gap_recovery` and effective rank -- but the
size of the effect is markedly asymmetric across datasets, and no arm beats
its own B0 under this metric either.

## Structural invariants (independently re-verified, not just training-time-asserted)

**Encoder freeze, byte-level:** SHA256 of every `encoder.*` tensor in the
frozen-arm checkpoints matches the source B0 checkpoint exactly --
`b7cc8b56...` (ETTh1 B0 == Arm B) and `ecfcc2e9...` (Weather B0 == Arm D).
Arm C's (trainable) encoder hash differs from B0 as expected. This is
independent of, and consistent with, the training-time
`requires_grad=False` + zero-gradient assertion already built into
`scripts/train_seqfull01.py`.

**Full-memory, by code inspection:** `step_logits()` in
`models/SequentialSetRetriever.py` scores the entire `candidate_embeddings`
tensor (N=8449 ETTh1 / N=36696 Weather) at every one of K=10 steps; only
already-selected/invalid positions are masked to `-inf`. No shortlist stage
exists in `train_seqfull01.py` or `eval_seqfull01_stage2.py`.

**Encoder freeze != Stage-1/Stage-2 separation:** every arm here (Trainable
and Frozen alike) already separates Stage-1 (produces a hard Top-10 via
`set_forced_selection`) from Stage-2 (B0's own unchanged forecaster/gate/
fusion/aggregation; forecasting loss never backpropagates into Stage-1).
"Frozen" only means the Stage-1 encoder's own weights don't update during
Stage-1 training. True Stage-1+Stage-2 joint end-to-end training was not
implemented here and is out of scope for this experiment.

## Reading (all 4 arms complete)

Freezing the encoder moved `gap_recovery` toward zero (less negative) on
**both** datasets: ETTh1 -0.404 -> -0.248 (delta +0.156), Weather -0.816 ->
-0.608 (delta +0.208); HardAggregateMSE@10 confirms the same direction with
a much larger, dataset-asymmetric magnitude (Weather 4.7x, ETTh1 1.14x).
Same-direction pattern on both datasets and across two independent metrics,
matching the experiment's pre-registered "Case A" framing (frozen improves
over trainable, consistently across datasets) -- but on neither dataset does
the frozen arm approach `gap_recovery = 0` (matching B0), and no arm beats
its own B0 under either aggregate metric: **every one of the four arms
remains net negative / net worse than B0**. This is read as **partial
support for H1** (collapse is a genuine, consistent contributing cause)
**without full support** -- H2 (the discrete greedy rule does not transfer
to held-out queries even from a stable, high-quality representation)
remains necessary to explain the residual failure once the encoder is held
fixed at B0's values. See `research/EXPERIMENT_LOG.md` for the full
write-up; the decision between competing explanations, novelty assessment,
and next-experiment recommendation belong to the reviewer (ChatGPT/Codex),
not this file.

## Fingerprint checks

`b0_unforced_final_mse` is independently re-derived by every arm's own
evaluation run (each script re-runs B0's own unforced selection through the
identical eval loop before reporting the forced-selection number). ETTh1:
0.373122 (Arm B) matches 0.37312 (EXP-SEQFULL01's recorded B0 baseline) to 5
significant figures. Weather: 0.175529 reproduced identically by Arm C and
Arm D independently (same B0 checkpoint, same eval code, two separate runs).
See `checkpoint_fingerprints.txt` for sha256 of every checkpoint/cache file
used.

## Sanity

`pytest tests/`: 466 passed, 2 failed -- both pre-existing at HEAD
(`test_topk_coverage_reuses_target_indices_across_relations`,
`test_identity_retrieval_uses_raw_target_source_relation_without_encoder`),
no regression. `tests/test_exp_seqdiag01.py` (8 new tests) covers the
frozen-encoder gradient-isolation contract and the corrected `overlap_at_k`
metric definition.

## Server reboot mid-campaign (2026-09-06, ~03:2x UTC)

All background processes were cleanly SIGTERM'd before a planned reboot (no
mid-write), GPU 1 confirmed free before shutdown. Arm D's Stage-1 was 2/10
epochs in; the 2-epoch checkpoint was discarded (not resumed --
`train_seqfull01.py` has no epoch-resume logic) and Arm D was rerun from
scratch after reboot, completing all 10 epochs cleanly. Arms B and C were
unaffected (both completed before the reboot). Full resume-state record:
`logs/exp_seqdiag01/RESUME_STATE.md`.

## ERRATUM inherited from EXP-SEQFULL01

The "chance" baseline formula for `overlap_at_k` used earlier in this
project's write-ups was wrong (`K^2/N` instead of `K/N` for the full-memory
case). Corrected before this experiment started; see
`research/EXPERIMENT_LOG.md`'s ERRATUM under EXP-SEQFULL01 for the full
table. All `seq_set_recall_at_k` values reported in this file/experiment use
the corrected formula.
