# TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01

**STATUS: INTERIM REPORT -- Phase A complete (ETTh1 H96 + H720, seed0
only). Weather not started, per PART 20's validity gate.**

TRACK-T's own S1 arm was not TRUE Original KL: it used
`SlotHeads(n_slots=1, std=1e-3)`, a trainable query-projection matrix
`W_1`, optimizer-updated every step. So TRACK-T's `S1 -> S10` comparison
actually measured *single-projected-head -> multi-slot*, not
*Original KL -> multi-slot*. This track adds a genuine **T0** (zero
projection parameters, the literal `train_j_shared_encoder_drift01.py`
reference implementation, run completely unmodified) to cleanly
separate:

```
G_Proj = MSE(T0) - MSE(T1)          projection effect
G_MS(S)= MSE(T1) - MSE(T_S)         multi-slot effect, S=2,4,10
```

T1/T2/T4/T10 are **not retrained** -- read read-only from the existing,
untouched `TRACK-T-PURE-MULTISLOT-VALIDATION01` results. Full audit:
`research/T2-projection-multislot-decomposition/AUDIT.md`.

## Setup and reproduction check

GPU1 only (standing rule), confirmed fully free via `ps`/`nvidia-smi`
immediately before launch. Encoder init hash identical across all 5
arms within each setting (`0f044e3ce726fc5c...` at H96,
`b37fa4031f538e4b...` at H720). T0's checkpoint carries no
`slot_heads_state_dict` (runtime-asserted). Checkpoint criterion
identical across all 5 arms: min val retMSE@10 (T0's own, native
criterion). 10/10 equivalence unit tests pass (score/loss/gradient
equivalence to the Original-KL reference pattern, optimizer-parameter
check, full-gradient-both-branches, S-independent teacher, shared
batch-order-construction path, K=10, exact Top-10 reduction).

**Reproduction (PART 16/17)**: T0's freshly-trained Stage2 MSE matched
TRACK-S's historical Original-KL value **exactly** at both horizons
(H96: 0.373682 = 0.373682; H720: 0.529134 = 0.529134) -- pipeline
determinism reconfirmed, no STOP/VALIDITY trigger encountered.

## Main result table

**ETTh1 H96 (seed0)**

| Arm | Proj. heads | Recall@10 | retMSE | D | C | Agg | Stage2 MSE | λ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 True KL | 0 | 0.0535 | 0.6807 | 0.06807 | 0.34585 | 0.41392 | **0.373682** | 0.511 |
| T1 | 1 | 0.0548 | 0.6325 | 0.06325 | 0.35909 | 0.42234 | 0.375676 | 0.501 |
| T2 | 2 | 0.0493 | 0.6326 | 0.06326 | 0.36527 | 0.42853 | 0.377003 | 0.483 |
| T4 | 4 | 0.0452 | 0.7088 | 0.07088 | 0.35762 | 0.42850 | 0.376284 | 0.384 |
| T10 | 10 | 0.0465 | 0.7011 | 0.07011 | 0.35767 | 0.42779 | 0.375539 | 0.383 |

**ETTh1 H720 (seed0)**

| Arm | Proj. heads | Recall@10 | retMSE | D | C | Agg | Stage2 MSE | λ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 True KL | 0 | 0.0224 | 0.9554 | 0.09554 | 0.54884 | 0.64437 | 0.529134 | 0.552 |
| T1 | 1 | 0.0572 | 0.8254 | 0.08254 | 0.51026 | 0.59280 | 0.506830 | 0.551 |
| T2 | 2 | 0.0528 | 0.8361 | 0.08361 | 0.51002 | 0.59363 | 0.504628 | 0.551 |
| T4 | 4 | 0.0524 | 0.8388 | 0.08388 | 0.49912 | 0.58299 | 0.500167 | 0.552 |
| T10 | 10 | 0.0504 | 0.8404 | 0.08404 | 0.49350 | 0.57755 | **0.497591** | 0.552 |

## Effect decomposition (PART 12)

| Horizon | G_Proj (T0->T1) | G_MS (T1->T10) | G_total (T0->T10) | G_Proj % of total | G_MS % of total |
|---|---:|---:|---:|---:|---:|
| H96 | +0.001995 (harmful) | -0.000138 (n.s.) | +0.001857 (net harmful) | 107% | n.s. |
| H720 | **-0.022304** (large, sig) | -0.009239 (sig) | -0.031543 | **70.7%** | 29.3% |

**Headline finding: at H720, the single trainable projection alone
accounts for more than twice the improvement that multi-slot adds on
top.** The track's original framing ("Multi-Slot helps at long
horizons") is not wrong that multi-slot helps -- it does, significantly
-- but it substantially overstates multi-slot's share of the total
improvement. Most of what TRACK-T attributed to "Multi-Slot" was
actually the confounded single-projection effect.

## D/C mechanism, decomposed by step

| Horizon | Step | delta D | delta D % | delta C | delta C % |
|---|---|---:|---:|---:|---:|
| H96 | T0->T1 (proj) | -0.00482 | -7.1% | +0.01325 | +3.8% |
| H96 | T1->T10 (multi-slot) | +0.00686 | +10.9% | -0.00142 | -0.4% |
| H720 | T0->T1 (proj) | **-0.01300** | **-13.6%** | **-0.03858** | **-7.0%** |
| H720 | T1->T10 (multi-slot) | +0.00151 | +1.8% | -0.01676 | -3.3% |

At H720, the projection step improves **both** `D` and `C`
substantially -- the strongest, cleanest win of any single step in this
track. The "D approx constant, C down" pattern TRACK-T originally
reported applies specifically to the **multi-slot increment on top of
an already-improved T1 baseline**, not to the full T0->T10 gain as
previously framed. At H96, the projection step trades D for C in the
wrong direction (D improves, but C worsens by more, net harmful) while
multi-slot changes almost nothing.

## Recall pattern (PART 20, Q5)

At H720, recall@10 **more than doubles** from T0 to T1 (0.0224 ->
0.0572) moving in the SAME direction as D/C/forecast (all improve
together) -- no dissociation there. But from T1 to T10, recall **drops**
11.9% relative (0.0572 -> 0.0504) while `C` and forecast MSE **keep
improving** -- this is the interesting recall/complementarity
dissociation, and it is specific to the multi-slot step, not the
projection step. At H96, recall's small movements track roughly with D,
not with any downstream benefit (there is none to track).

## Bootstrap significance (10k-rep paired, query_start_idx unit)

| Comparison | H96 Stage1 | H96 Stage2 | H720 Stage1 | H720 Stage2 |
|---|---|---|---|---|
| T1 vs T0 | sig, worse | sig, worse | **sig, better (large)** | **sig, better (large)** |
| T2 vs T1 | sig, worse | sig, worse | not sig | sig, better |
| T4 vs T1 | sig, worse | not sig | sig, better | sig, better |
| T10 vs T1 | sig, worse | not sig | sig, better | sig, better |
| T10 vs T0 | sig, worse | sig, worse | sig, better (largest) | sig, better (largest) |

## Case assessment (PART 15)

**ETTh1 H720**: closest to **Case B** (projection + multi-slot both
contribute, clean monotonic `T0 > T1 > T4 > T10` ordering, `C` decreases
at every step) -- but the spec's Case B language doesn't capture the
*size* asymmetry: projection contributes 70.7% of the total gain,
multi-slot 29.3%. This should be read as "projection-dominant, with a
real but secondary multi-slot contribution," not as two roughly-equal
contributors.

**ETTh1 H96**: fits none of the four pre-registered cases, all of which
implicitly assume `T0 <= T1` (projection helps or is neutral). Here
projection is **actively harmful** (T0 significantly beats T1 at both
Stage1 and Stage2), and multi-slot on top is statistically negligible.
An unregistered fifth pattern: *projection actively hurts, multi-slot
neither recovers nor worsens it further.*

## Q1-Q7 (PART 20)

**Q1 (is existing S1 actually different from true Original KL)**: YES,
substantively, and in **opposite directions per horizon**. At H96, true
T0 is significantly *better* than the old confounded S1 (0.373682 vs
0.375676). At H720, true T0 is significantly *much worse* (0.529134 vs
0.506830, 4.2% relative) -- the old S1 was never a valid Original-KL
baseline at either horizon.

**Q2 (does projection give an independent gain)**: horizon-dependent
and itself sign-reversed. YES at H720 (large, highly significant,
`G_Proj=-0.0223`). NO at H96 -- significantly *harmful*
(`G_Proj=+0.0020`).

**Q3 (do multiple slots add gain beyond single projection)**: YES at
H720 (T2/T4/T10 all significantly beat T1, monotonically), but this
additional gain (29.3% of the total T0->T10 effect) is smaller than the
projection step's own gain (70.7%). NO at H96 (T4/T10 not significantly
different from T1; T2 significantly worse).

**Q4 (does H720's improvement still mean D~const, C down)**: only for
the multi-slot increment. The projection increment instead improves
*both* D (-13.6%) and C (-7.0%) together -- a different, stronger
mechanism than the original "D const, C down" story, which now applies
only to the smaller, second half of the total effect.

**Q5 (does recall drop while C/Agg/forecast improve)**: YES, but
specifically for the multi-slot step at H720 (recall -11.9% relative
while C and forecast keep improving) -- not for the projection step,
where recall, D, C, and forecast all move together in the same
direction.

**Q6 (which stage fails at H96)**: both fail to help, but projection
actively *hurts* (not merely fails) -- this is the primary source of
H96's small net degradation. Multi-slot on top is statistically
neutral.

**Q7 (can Multi-Slot still be maintained as the main contribution)**:
**not as previously framed.** At H720, multi-slot IS a real, independent,
statistically significant contributor (29.3% of the total gain) and
survives the confound removal. But it is not the *dominant* mechanism --
the single trainable query projection contributes more than twice as
much. The defensible claim going forward is "query projection, with a
smaller but real multi-slot contribution on top," not "Multi-Slot
drives the H720 improvement." At H96, neither component helps, and the
projection component is actively harmful.

## STOP/VALIDITY compliance (PART 19)

No trigger was encountered: T0 score/loss/gradient equivalence to the
Original-KL reference pattern verified (10/10 unit tests), init hash
identical across all 5 arms per setting, T0 optimizer contains no
slot-head parameters, checkpoint criterion identical across all arms,
shared TRACK-R base checkpoint confirmed via SHA-256, teacher
S-independent, batch-order construction path shared and deterministic,
and both T0 reproductions matched their historical reference values
exactly. No StopGrad/overlap/aggregate/budget/Set-Oracle/new-scorer/
hyperparameter-tuning was added in response to any result, per PART 4's
prohibitions.

## Next step

Weather H96 seed0 and Weather H720 seed0, identical protocol and
hyperparameters, to test whether "projection dominant at H720,
projection harmful at H96" transfers to a second dataset -- withheld
until this ETTh1 validity audit was complete, per PART 20's gate.
