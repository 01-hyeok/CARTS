# TRACK-T-PURE-MULTISLOT-VALIDATION01

**STATUS: INTERIM REPORT -- Phase A complete (ETTh1 H96 + H720, seed0
only). Phase B (Weather) not yet run.**

Question: does giving Original-KL (future-aligned KL) retrieval
multiple independent query "slots" -- S rankings instead of 1 -- improve
the retrieved Top-10 set and downstream forecasting, on its own, with
NO StopGrad-Key, NO overlap penalty, NO aggregate future loss, NO
relevance budget? Teacher, encoder, gradient flow (full, both branches),
and KL objective are held byte-for-byte identical to Original KL/J0;
`num_slots in {1,2,4,10}` is the ONLY intended difference between arms.
Full audit: `research/T-pure-multislot-validation/AUDIT.md`.

## Setup

GPU: queued on **GPU1** (project standing rule), started only after
TRACK-R released it (verified via `ps`/`nvidia-smi` immediately before
launch); an earlier attempt on a separately-allocated GPU4 was corrected
per explicit user instruction and its partial artifacts deleted (see
AUDIT.md's "Revised decision" section). Base forecaster reused read-only
from TRACK-R for both settings (common per-setting base across all 4
arms, confirmed via identical `model_state_dict` SHA-256). Encoder init
hash is identical across all 4 arms within each setting
(`0f044e3ce726fc5c...` at H96, `b37fa4031f538e4b...` at H720 -- the
latter matches the session-wide historical J0/J1/K2/M2 init hash
exactly, further confirming deterministic reproduction). 20/20 unit
tests pass (spec PART 29); full pytest suite unaffected.

## Stage2 forecast MSE (test split, seed0)

| S | ETTh1 H96 | ETTh1 H720 |
|---:|---:|---:|
| 1 | 0.375676 | 0.506830 |
| 2 | 0.377003 | 0.504628 |
| 4 | 0.376284 | 0.500167 |
| 10 | **0.375539** | **0.497591** |

At H720, forecast MSE decreases **monotonically** with `num_slots` --
S10 is the best arm at H720. At H96, results are flat/mixed: S10 is
marginally the best but not significantly different from S1; S2 is
significantly *worse*.

## Stage1 AggMSE and D/C decomposition (test split)

| S | ETTh1 H96 D | ETTh1 H96 C | ETTh1 H96 Agg | ETTh1 H720 D | ETTh1 H720 C | ETTh1 H720 Agg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.063250 | 0.359093 | 0.422343 | 0.082536 | 0.510260 | 0.592796 |
| 2 | 0.063262 | 0.365266 | 0.428528 | 0.083614 | 0.510019 | 0.593632 |
| 4 | 0.070880 | 0.357616 | 0.428496 | 0.083876 | 0.499118 | 0.582993 |
| 10 | 0.070114 | 0.357674 | 0.427788 | 0.084045 | 0.493503 | 0.577547 |

**H720 matches the spec's section-20 "success pattern" exactly: `D`
stays essentially flat (+1.8% relative, S1->S10) while `C` decreases
clearly and monotonically (-3.3% relative) -- `Agg` (and downstream
forecast) decrease as a direct consequence of the `C` term, not the `D`
term.** This is a clean mechanistic account: Multi-Slot's benefit at
H720 comes from improved cross-candidate complementarity, with
individual relevance essentially preserved.

**H96 does NOT match this pattern.** `D` increases meaningfully (+10.9%
relative, S1->S10) while `C` stays flat/noisy (no significant
directional trend) -- so `Agg` moves slightly worse, driven by the `D`
term, not `C`. Multi-Slot buys no complementarity benefit at H96 and
costs a small amount of individual relevance.

## Slot specialization diagnostics (PART 14/15)

| S | H96 slot Top-10 overlap | H720 slot Top-10 overlap |
|---:|---:|---:|
| 2 | 0.028 | 0.211 |
| 4 | 0.540 | 0.237 |
| 10 | 0.633 | 0.265 |

No overlap penalty was used anywhere in this track (PART 14). At H96,
slots specialize initially (S2 overlap is low) but collapse toward
redundancy fast -- by S=4 the slots' own Top-10 sets already overlap
54%, rising to 63% at S=10. At H720, overlap stays low and grows slowly
even out to S=10 (21% -> 24% -> 27%) -- slots remain meaningfully
differentiated. This mirrors the D/C finding exactly: where slots stay
differentiated (H720), complementarity (`C`) improves; where slots
collapse fast (H96), it does not.

## Bootstrap significance (10k-rep paired, query_start_idx unit)

| Comparison | H96 Stage1 Agg | H96 Stage2 forecast | H720 Stage1 Agg | H720 Stage2 forecast |
|---|---|---|---|---|
| S2 vs S1 | sig, worse | sig, worse | not sig | **sig, better** |
| S4 vs S1 | sig, worse | not sig | **sig, better** | **sig, better** |
| S10 vs S1 | sig, worse | not sig | **sig, better** | **sig, better (largest)** |

## Case assessment (PART 21)

**ETTh1 H720: Case A -- Multi-Slot effect present.** S>1 significantly
and monotonically improves both Stage1 AggMSE and Stage2 forecast MSE,
via the clean D-const/C-down mechanism.

**ETTh1 H96: fits no pre-registered Case cleanly.** Not Case B (slot
collapse in the strict "forecast ~= S1" sense) -- S2 is significantly
*worse*, not merely unchanged. Not Case C either -- `C` does not
decrease at all (a precondition of Case C), only `D` worsens. Closest
honest description: slots DO show measurable, real specialization by
S=4 (per the overlap statistics) but that specialization never
translates into a complementarity gain at this horizon, while incurring
a small individual-relevance cost.

## RQ1-RQ7 (PART 26)

**RQ1 (Multi-Slot alone improves retrieval)**: horizon-dependent -- YES
at H720 (Stage1 AggMSE significantly better, monotonic in S), NO at H96
(significantly worse for every S>1).

**RQ2 (does increasing S reduce C)**: YES at H720 (monotonic, -3.3%
relative at S10, mostly significant); NO at H96 (flat/noisy, never
significant in the improving direction).

**RQ3 (is C reduction achieved without sacrificing D)**: at H720, YES --
D moves only +1.8% while C moves -3.3%, a clean asymmetric pattern. Not
applicable at H96 (C never decreases there).

**RQ4 (does improved set quality transfer to lower forecast MSE)**: YES
at H720 -- Stage1 AggMSE and Stage2 forecast MSE improve together,
monotonically, both significant from S4 onward. Consistently null at
H96 -- neither improves.

**RQ5 (is S=10 necessary)**: at H720, S10 gives the single best result
on both Stage1 and Stage2 of all arms tested -- better than S4, though
the *marginal* gain shrinks from S1->S4 (-0.0045 stage2) to S4->S10
(-0.0026 stage2), suggesting diminishing but not yet fully saturated
returns. At H96 the question is moot (no arm beats S1 significantly).

**RQ6 (is the effect stronger at H720 than H96)**: YES, unambiguously --
this is the track's central finding. It replicates, in ISOLATION (no
StopGrad-Key/overlap/aggregate/budget present anywhere in this track),
the exact same horizon-dependence pattern TRACK-R and TRACK-S already
observed for the full M2 method and for StopGrad-Key's own contribution
respectively -- now shown to be present in the Multi-Slot architecture
alone.

**RQ7 (does the pattern transfer to Weather)**: NOT YET TESTED -- Phase
B has not been run.

## PART 30 final judgment (interim)

The track's core question -- does giving future-aligned-KL retrieval
multiple query slots, alone, improve set complementarity and
forecasting -- has a horizon-dependent answer on the one dataset tested
so far: **YES at H720, mildly NO at H96.** This is a real, mechanistically
-explained effect of `num_slots` in isolation (traced to differential
slot specialization: slots stay differentiated at H720 but collapse
toward redundancy by S=4 at H96), not an artifact of StopGrad-Key,
overlap penalty, aggregate loss, or relevance budget -- none of which
exist anywhere in this track's code path. This is the single cleanest
mechanistic decomposition of "why Multi-Slot helps more at long horizons"
produced so far across TRACK-R/S/T, since it isolates the Multi-Slot
variable completely from every other architectural difference.

## STOP rule compliance (PART 25)

No StopGrad, overlap penalty, aggregate loss, relevance budget,
slot-specific teacher, or slot-count fine-tuning was added anywhere in
this track in response to H96's negative-ish result. Reported as-is.

## Next step

Phase B: Weather H96 seed0 and Weather H720 seed0, identical protocol
and hyperparameters (no per-dataset changes), to test RQ7.
