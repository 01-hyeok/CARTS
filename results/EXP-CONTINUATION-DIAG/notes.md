# EXP-CONTINUATION-DIAG -- factual notes (2026-09-07)

Status: **COMPLETE** for 3 cells (ETTh1 H96, Weather H96, ETTh1 H720) × 3
first-anchor policies (dense_first, b0_first, oracle_first) = 9
combinations, 500 queries each. Weather H720 not run (no EXP-MARGUTIL01
checkpoint exists for it — cancelled, see D-0013).

**Full interpretation, the primary table, all 7 required questions
answered with numbers: `REPORT.md`.** This file only indexes the raw
artifacts.

## Artifacts

- `comparison.csv` — the primary summary table (9 rows, one per combo)
- `continuation_diag_<cell>_<anchor>.csv` — query-level rows (9 files,
  500 rows each), full column set per the experiment spec section 11
- `<cell>_<anchor>_summary.json` — cell×anchor summary statistics (9 files)
- `<cell>_<anchor>_top20_catastrophic.json` — 20 highest-`ratio_dense`
  queries per combo (9 files)
- `REPORT.md` — full write-up, primary table, Q1-Q7 answered
- `command.txt`, `env.txt`, `git_commit.txt`, `checkpoint_fingerprints.txt`
  (identical checkpoints to EXP-MARGUTIL01/EXP-FIRSTANCHOR-DIAG — no new
  training here)

## One-line summary

Extreme top-tail ranking correlation (`Top1% ρ`) is negative in all 9
combinations tested; global correlation is mixed. A good first anchor does
not fix, and on 2/3 cells actively worsens, t=2 continuation failure. See
`REPORT.md` Q1-Q7 for the full evidence-based breakdown.
