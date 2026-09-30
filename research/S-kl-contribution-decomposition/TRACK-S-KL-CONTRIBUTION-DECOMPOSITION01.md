# TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01

**STATUS: INTERIM REPORT -- 2 of (eventually) 4 ETTh1 horizons complete
(H96, H720; seed0 only).** Runs in parallel with
TRACK-R-FINAL-METHOD-GENERALIZATION01 (GPU1, untouched throughout) on a
separate, fully free GPU2. Not a new architecture: decomposes the
observed forecasting gain into three sequential steps -- Raw Cosine ->
**Original KL** (new arm this track) -> StopGrad-Key KL (=J1, reused
from TRACK-R) -> Multi-Slot M2 (=M2, reused from TRACK-R). Full audit:
`research/S-kl-contribution-decomposition/AUDIT.md`.

## Setup

GPU2 confirmed fully free at track start (0% util, 4 MiB used) --
`gpu_allocation_log.txt`. `train_j_shared_encoder_drift01.py` (TRACK-J's
"J0" script) audited line-by-line and confirmed to implement S1's exact
definition already, unmodified: single shared encoder, BOTH query and
candidate branches gradient-ON (`E = encode_raw(model, exp.memory_x,
c)`, no `.detach()`), same `kl_loss`/`normalized_teacher_prob`/`tau_t=
tau_s=0.1` as J1, and -- critically -- the **identical checkpoint
-selection criterion** as J1/J2 (`val_metric = val retMSE@10`), so no
reconciliation was needed for PART 10/18's "same criterion" requirement.
No `EXPECTED_INIT_HASH`-style hardcoded constant exists in this script,
so no bypass flag was needed either. Base/Cosine/J1/M2 all reused
read-only from TRACK-R (never retrained). Both Original KL checkpoints
this track produced reproduced **byte-identical** (SHA-256 match) to
the historical TRACK-J J0 checkpoint at H720 -- further confirming the
whole session's training pipelines are genuinely deterministic.

## Stage2 comparison table (test split, seed0)

| Horizon | Base | Cosine | Original KL | J1 | M2 |
|---|---:|---:|---:|---:|---:|
| 96 | 0.392423 | 0.380437 | **0.373682** | 0.374193 | 0.379448 |
| 720 | 0.560354 | 0.550745 | 0.529134 | 0.491584 | **0.484746** |

At H96, Original KL is (very slightly, not significantly) the BEST
arm of all four -- StopGrad-Key and Multi-Slot add nothing measurable.
At H720, the ranking is completely different: Original KL clearly beats
Cosine but trails J1 and M2 substantially.

## Stage1 comparison table

| Horizon | Method | retMSE | D | C | AggMSE | Recall@10 |
|---|---|---:|---:|---:|---:|---:|
| 96 | Cosine | 0.865244 | 0.086524 | 0.424266 | 0.510790 | 0.033070 |
| 96 | Original KL | 0.680699 | 0.068070 | 0.345848 | 0.413918 | 0.053516 |
| 96 | J1 | 0.712228 | 0.071223 | 0.345246 | 0.416469 | 0.045427 |
| 96 | M2 | 0.724295 | 0.072429 | 0.359059 | 0.431489 | 0.031716 |
| 720 | Cosine | 1.359586 | 0.135959 | 0.736605 | 0.872563 | 0.012005 |
| 720 | Original KL | 0.955397 | 0.095540 | 0.548835 | 0.644375 | 0.022377 |
| 720 | J1 | 1.005504 | 0.100550 | 0.463476 | 0.564026 | 0.021855 |
| 720 | M2 | 0.988612 | 0.098861 | 0.424731 | 0.523593 | 0.028479 |

Original KL's H720 numbers exactly match TRACK-J3's original J0
reference values (`retMSE=0.955397`, `D=0.095540`, `C=0.548835`,
`Agg=0.644375`) -- expected, since the checkpoint is byte-identical.

## Gain decomposition (`G_total = MSE_Base - MSE_M2`)

| Horizon | G_total | G_KL (Cosine->KL) | G_SG (KL->J1) | G_MS (J1->M2) |
|---|---:|---:|---:|---:|
| 96 | 0.012976 | 0.006755 (**52.1%**) | -0.000511 (-3.9%, n.s.) | -0.005255 (-40.5%) |
| 720 | 0.075608 | 0.021611 (28.6%) | 0.037550 (**49.7%**) | 0.006839 (9.0%) |

(n.s. = not statistically significant; all other entries are, per the
10k-rep paired bootstrap, `bootstrap.csv`)

## The mechanistic finding: StopGrad-Key's value is a pure complementarity (C) effect, and it is horizon-dependent

`horizon_trend.csv`: `delta_C_SG = C_J1 - C_OriginalKL` is essentially
zero at H96 (`-0.0006`) but large and negative at H720 (`-0.0854`,
i.e. J1's `C` is dramatically better than Original KL's there) --
**this tracks the downstream `G_SG` pattern exactly**. Meanwhile `D`
(individual relevance) is essentially UNCHANGED or even slightly worse
for J1 vs Original KL at both horizons (Original KL's `D` is actually
lower/better at both H96 and H720 -- StopGrad-Key does NOT improve
individual candidate quality, consistent with TRACK-J3's original
finding). **StopGrad-Key's entire downstream benefit, where it exists,
routes through improved cross-candidate complementarity, not individual
relevance -- and that complementarity benefit itself appears only at
the longer horizon tested so far.** This directly extends TRACK-J3's
original D/C decomposition (which only ever looked at H720) to a second
horizon, and shows the effect is not horizon-invariant.

Similarly, `delta_C_MS = C_M2 - C_J1` is `+0.0138` (M2 makes `C` WORSE)
at H96 but `-0.0387` (M2 makes `C` further BETTER) at H720 -- M2's own
complementarity mechanism shows the same horizon-dependent sign flip
TRACK-R's own report already noted, now confirmed to be a `C`-specific
(not `D`-specific) effect at both ends.

## Hypothesis verdicts (interim, n=2 horizons, 1 seed)

- **H_A (future-aligned KL is main contribution)**: partially supported
  at H96 (52% of total gain, the largest single component there) but
  NOT at H720 (29%, smaller than `G_SG`).
- **H_B (StopGrad-Key also core)**: strongly supported at H720 (`G_SG`=
  50%, the single LARGEST component, highly significant) but NOT
  supported at H96 (`G_SG`~0, not significant).
- **H_C (Multi-Slot = long-horizon extension)**: supported so far (H96:
  `G_MS` significantly negative; H720: `G_MS` significantly positive) --
  but this is only 2 horizons; needs H192/H336 to distinguish a real
  trend from a two-point artifact.
- **H_D (M2 wins everywhere)**: NOT supported -- M2 loses to J1 at H96
  (significant, matching TRACK-R's own finding).

## Case assessment (PART 22)

No single pre-registered Case (1-4) cleanly fits both horizons at once.
H96 alone resembles Case 1 (future-aligned KL is the whole story;
StopGrad-Key and Multi-Slot add nothing or hurt). H720 alone resembles a
Case 2/3 blend (StopGrad-Key is the single largest contributor, AND
Multi-Slot adds a smaller, real, significant further gain on top). The
honest reading with only 2 of 4 horizons and 1 seed: **contribution
appears to be genuinely horizon-dependent, not yet a settled single
verdict.** H192/H336 and seed1/2 replication (pending TRACK-R) are
needed before committing to one Case-level claim.

## Interim answers (PART 21)

**Q1 (Cosine->Original KL improvement)**: real and significant at both
horizons (H96: -0.0068, H720: -0.0216) -- future-aligned KL
representation learning clearly helps on its own, at every horizon
tested so far.

**Q2 (Original KL->J1 additional improvement)**: exists and is large at
H720 (-0.0375, highly significant) but is statistically ZERO at H96
(CI includes zero) -- StopGrad-Key's benefit is not universal.

**Q3 (J1->M2 gain only at H720?)**: yes so far -- positive and
significant at H720, negative and significant at H96.

**Q4 (largest single step in Base->M2 gain)**: at H96, it's the
Cosine->OriginalKL step (52%); at H720, it's the OriginalKL->J1 step
(50%) -- the identity of the "main contribution" itself changes with
horizon.

**Q5 (where the data currently places the main contribution)**: not yet
resolvable with 2 horizons -- the data says "it depends on horizon,"
which is itself the most defensible current answer. Extending to
H192/H336 is the direct next step to determine whether this is a smooth
trend (support for H_C) or a step function at some horizon boundary.

## STOP rule compliance

No new loss, teacher, slot design, coefficient tuning, or Stage2
redesign anywhere in this track. TRACK-R was never stopped, paused, or
resource-starved (GPU1 untouched throughout; verified via `nvidia-smi`
before and during TRACK-S's runs).
