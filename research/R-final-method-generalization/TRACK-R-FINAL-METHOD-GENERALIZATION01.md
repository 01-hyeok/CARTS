# TRACK-R-FINAL-METHOD-GENERALIZATION01

**STATUS: INTERIM REPORT -- 2 of 24 planned settings complete.** Not
architecture exploration: this track fixes the final method
(Relevance-Constrained Multi-Slot M2 + Uniform aggregate + Mixture
fusion + Trainable Global Lambda) and tests generalization across
datasets (ETTh1, Weather) x horizons (96/192/336/720) x seeds (0/1/2).
Per an explicit user redirect mid-track, execution order was changed
from "seed-first" to "horizon/dataset-breadth-first" (see AUDIT.md) --
this report covers the first two completed settings (**ETTh1 H720
seed0**, **ETTh1 H96 seed0**) at the user's request, ahead of the full
grid. No seed replication yet (n=1 per setting) -- mean±std across
seeds is NOT reported here; that requires later settings.

Full audit: `research/R-final-method-generalization/AUDIT.md`.

## Phase A: clean pipeline reproduction (ETTh1 H720 seed0)

**PASS.** Stage1 M2: `retMSE=0.988612` (ref 0.9886), `C=0.424731` (ref
0.4247), `Agg=0.523593` (ref 0.5236) -- exact match. J1's checkpoint
reproduced **byte-identical** to the historical TRACK-J2 checkpoint
(same SHA-256). Retrieval caches reproduce all known values exactly.
Base forecaster (freshly trained per PART 11, not reused from any prior
track) landed at `test_mse=0.560354`, ~14.6% worse than TRACK-M's
historical frozen S0 (0.488790) -- root-caused to construction-order
-dependent random init (verified bit-for-bit deterministic within this
track's own pipeline; not a bug). Final M2 Stage2 MSE (0.484746) is
correspondingly higher than the old reference (~0.464), inheriting the
base gap proportionally. **User reviewed this finding and approved
proceeding as-is** (AUDIT.md).

## Main table (MSE, test split)

| Horizon | Base | Cosine+Stage2 | J1+Stage2 | M2+Stage2 |
|---|---:|---:|---:|---:|
| 96 | 0.392423 | 0.380437 | **0.374193** | 0.379448 |
| 720 | 0.560354 | 0.550745 | 0.491584 | **0.484746** |

(seed0 only; mean±std across seeds not yet available)

## Stage1 table

| Horizon | Method | retMSE | D | C | AggMSE | Recall@10 |
|---|---|---:|---:|---:|---:|---:|
| 96 | Cosine | 0.865244 | 0.086524 | 0.424266 | 0.510790 | 0.033070 |
| 96 | J1 | 0.712228 | 0.071223 | 0.345246 | 0.416469 | 0.045427 |
| 96 | M2 | 0.724295 | 0.072429 | 0.359059 | 0.431489 | 0.031716 |
| 720 | Cosine | 1.359586 | 0.135959 | 0.736605 | 0.872563 | 0.012005 |
| 720 | J1 | 1.005504 | 0.100550 | 0.463476 | 0.564026 | 0.021855 |
| 720 | M2 | 0.988612 | 0.098861 | 0.424731 | 0.523593 | 0.028479 |

`Agg=D+C` identity confirmed to <1e-3 for every row (unit test item 10).

## Learned global lambda

| Horizon | lambda_cosine | lambda_J1 | lambda_M2 |
|---|---:|---:|---:|
| 96 | 0.196 | 0.404 | 0.414 |
| 720 | 0.354 | 0.448 | 0.449 |

## Paired bootstrap (test split, query_start_idx unit, 10,000 reps, per-setting)

**H720 (seed0)**: M2 significantly beats Base (`-0.0756`, CI
`[-0.0782,-0.0730]`), Cosine (`-0.0660`), AND J1 (`-0.0068`, CI
`[-0.0083,-0.0054]`) -- M2 is the clear winner.

**H96 (seed0)**: M2 significantly beats Base (`-0.0130`, CI
`[-0.0148,-0.0112]`), is statistically TIED with Cosine (`-0.0010`, CI
`[-0.0024,0.0005]`, includes zero), and is significantly **WORSE** than
J1 (`+0.0053`, CI `[0.0041,0.0064]`, entirely positive).

## Interim finding: horizon-dependent reversal (M2 vs J1)

At H720, M2's Stage1 quality advantage over J1 (lower D, C, AggMSE;
higher Recall) translates cleanly into a Stage2 forecasting win. At H96,
M2's Stage1 quality is actually slightly **worse** than J1's on every
metric (`retMSE` 0.7243 vs 0.7122, `C` 0.3591 vs 0.3452, `AggMSE` 0.4315
vs 0.4165, `Recall` 0.0317 vs 0.0454) -- and this reverses cleanly into
J1 significantly beating M2 downstream too. In both horizons, Stage1
ranking and Stage2 ranking agree with each other (internally
consistent), but the ranking ITSELF flips between horizons. This is
exactly the kind of dataset/horizon-dependent pattern PART 24 anticipates
and explicitly forbids reacting to with architecture changes -- reported
here as an honest, unresolved open finding pending the rest of the
grid (n=1 setting per horizon so far; PART 21's Weak-Evidence pattern --
"H720 only strong, short horizons weak/no gain" -- may be emerging, but
two data points cannot yet establish this).

## What this report does NOT yet establish

With only 2 of 24 settings complete (no Weather, no H192/H336, no
seed 1/2 replication), none of PART 33's RQ1-RQ5 can be answered with
the intended statistical rigor:

- RQ3 (robustness across horizons/datasets/seeds) is explicitly NOT
  answerable yet -- the one horizon-reversal finding above is a single
  data point per horizon, not a trend.
- RQ4 (M2 vs J1 generally) is genuinely mixed so far: M2 wins clearly at
  H720, loses clearly at H96.
- Seed variance is completely unmeasured (PART 25 requires 0/1/2; only
  seed0 run so far, per the user's own "breadth first" redirect).

This interim report exists at the user's explicit request to checkpoint
progress after 2 settings, not as a generalization verdict. The next
settings (Weather H96/H720 seed0 in progress; ETTh1 H192/H336 and all
seed1/2 replication still pending) are required before PART 33's
questions can be answered.

## STOP rule compliance

No Stage1 retriever modification, no loss/gamma/delta changes, no
architecture changes in response to the H96 J1>M2 finding -- reported
as-is per PART 24's explicit instruction.
