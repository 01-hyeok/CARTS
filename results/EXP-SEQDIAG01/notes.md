# EXP-SEQDIAG01 -- interim results (as of 2026-09-06, Arm D still running)

Status: **IN PROGRESS.** Arm D's Stage-2 evaluation is running in the
background (GPU 1, pid 16248 at time of writing); this file will be
completely rewritten once all four rows are final, per the after-experiment
procedure in `CLAUDE.md`. Treat every number below as provisional except
where marked complete.

## Design

Frozen-B0 control arm (B0's own encoder weights, never updated; only the new
`SetConditioner` trains) vs. the existing Trainable arm, on both ETTh1 and
Weather, H96. Decision rule (Case A/B/C) is in
`research/EXPERIMENT_LOG.md` -> EXP-SEQDIAG01; not evaluated yet, pending
Arm D.

## Stage-2 results table (gap_recovery = how much of the B0-to-set-oracle gap
the sequential arm's forced selection recovers; negative = worse than B0,
i.e. moves in the wrong direction)

| Arm | Dataset | Stage-1 | Stage-2 | b0_unforced_mse | sequential_forced_mse | gap_recovery | seq_set_recall@10 |
|---|---|---|---|---:|---:|---:|---:|
| Trainable (reused, EXP-SEQFULL01) | ETTh1 | done | done | 0.37312 | 0.40277 | -0.404 | 0.01354 |
| Frozen-B0 (Arm B) | ETTh1 | done | done | 0.37312 | 0.38792 | -0.248 | 0.01242 |
| Trainable (Arm C) | Weather | done | done | 0.17553 | 0.26814 | -0.816 | 0.00917 |
| Frozen-B0 (Arm D) | Weather | done (10/10 epochs, 2026-09-06 05:08 UTC) | **running** | -- | -- | **pending** | -- |

Source files: `results/EXP-SEQFULL01/metrics.csv`,
`results/EXP-SEQDIAG01/armB_etth1_frozen_stage2.json`,
`results/EXP-SEQDIAG01/armC_weather_trainable_stage2.json`,
`results/EXP-SEQDIAG01/armD_weather_frozen_stage2.json` (not yet written).

## Reading so far (ETTh1 + Weather Trainable + ETTh1 Frozen only -- 3 of 4 rows)

- On ETTh1, freezing the encoder made `gap_recovery` **less negative**
  (-0.404 -> -0.248): consistent with H1 (representation collapse
  contributes to the failure) contributing *something*, but it does not come
  close to closing the gap to B0 -- both ETTh1 arms remain net negative.
- Weather's Trainable arm (-0.816) is substantially worse than either ETTh1
  arm on the same `gap_recovery` scale (never compare raw MSE across
  datasets; `gap_recovery` is the normalized quantity meant to be
  comparable).
- Whether Weather's Frozen arm (Arm D, pending) shows the same
  improving-but-insufficient pattern seen on ETTh1 (supporting a consistent
  cross-dataset H1 contribution) or fails to improve at all (leaning toward
  H2 dominating, or dataset-dependence / Case C) is the open question this
  arm's completion will answer. **No conclusion should be drawn before it
  exists.**
- `seq_set_recall_at_k` is at or below the small-N chance baseline
  (0.04167) on every arm reported so far, and `duplicate_rate` /
  `invalid_rate` are 0.0 on all arms -- the sequential selector is not
  cheating via duplicates/invalids, it is simply not recovering the
  teacher's set membership.

## Not yet done (blocking final write-up)

1. Arm D Stage-2 eval (running).
2. Fingerprint re-verification of the reused ETTh1 Trainable arm
   (`results/EXP-SEQFULL01/`) for this campaign.
3. Effective-rank diagnostic for the Weather Trainable encoder (Arm C).
4. Full comparison table with Frozen-minus-Trainable deltas per dataset.
5. Case A/B/C decision-rule application.
6. `pytest tests/` full regression (expect ~466 passed: 458 baseline + 8 new
   in `tests/test_exp_seqdiag01.py`; same 2 pre-existing failures).
7. `research/EXPERIMENT_LOG.md` entry + full `research/REVIEW_FOR_CHATGPT.md`
   refresh.

See `logs/exp_seqdiag01/RESUME_STATE.md` for the exact resume commands and
full provenance of what ran before/after the 2026-09-06 server reboot.
