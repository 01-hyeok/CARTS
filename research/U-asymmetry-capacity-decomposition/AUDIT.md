# TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01 -- AUDIT.md

## PART 0: the two competing hypotheses

TRACK-T2 found Query-only Projection (a single trainable `D x D`
matrix `W_q` applied only to the query side) beats True Original KL by
a large margin at H720 (0.529134 -> 0.506830). That single change
simultaneously (1) adds `D^2` learnable parameters and (2) breaks the
query/candidate symmetric-comparison constraint. This track adds two
controls that hold parameter count fixed at `D^2` while varying only
whether/where symmetry is broken:

- **U1 (Shared Symmetric)**: the SAME `W` applied to both query and
  candidate -- `D^2` params, symmetry PRESERVED (`cos(Wz_q, Wz_k)`).
- **U2 (Query-only)**: TRACK-T2's own T1 architecture -- `D^2` params,
  symmetry BROKEN, query side transformed.
- **U3 (Key-only)**: the mirror of U2 -- `D^2` params, symmetry BROKEN,
  candidate side transformed.

`H_capacity` predicts U1 ~= U2 ~= U3 (parameter count is what matters).
`H_asym` predicts U2 and/or U3 >> U1 (breaking symmetry is what
matters).

## GPU (PART 2)

Standing rule: GPU1 only. Verified fully free (`ps`/`nvidia-smi`, 0%
util, 4 MiB used) immediately before launch. `CUDA_VISIBLE_DEVICES=1`
throughout.

## All four arms retrained fresh (departure from TRACK-T2)

Unlike TRACK-T2 (which reused T1/T2/T4/T10 read-only), **all four arms
here are trained fresh** within this track's own pipeline. Reason: U1/
U2/U3 require a bit-identical initial projection weight tensor across
arms (PART 6) -- a control the existing TRACK-T/TRACK-T2 checkpoints
were never built to satisfy (TRACK-T's own `SlotHeads` used a
per-slot-index seed, not a single shared-across-arms seed). U0 is also
retrained fresh (rather than reusing TRACK-T2's T0) so this track's own
artifact tree is fully self-contained and its own equivalence tests
(PART 8) can be verified against artifacts this track itself produced.

## Implementation (`scripts/train_u_asymmetry_capacity01.py`)

`SingleProjection`: one `D x D` matrix, **no bias** (verified by unit
test 4), `W = I + eps`, `eps ~ N(0, std^2)`, generated via
`torch.Generator().manual_seed(PROJECTION_INIT_SEED=1000)` inside the
constructor -- calling this constructor in any of U1/U2/U3's separate
process invocations yields a **bit-identical** initial `W` (verified
live: `proj_init_sha=07e4bdd983fb4332` for all three arms in the smoke
test, and by unit test 5). `compute_scores(model, projection, ..., arm)`
selects which side(s) receive the projection: U0 neither (plain
cosine, `projection=None`), U1 both (symmetric), U2 query only, U3
candidate only -- **no `.detach()` anywhere**, query AND candidate
encoder gradients both flow in every arm (verified by unit test 9,
parametrized over all four arms). Single score vector, ordinary
`stable_topk_indices` Top-10 (NO multi-slot round-robin -- PART 4
explicitly forbids Multi-Slot in this track). Checkpoint criterion:
min val retMSE@10, identical across all four arms (matches Original
KL's own criterion, PART 10).

## Efficiency fix applied before real execution

`scripts/build_u_cache01.py`'s Stage1 metrics (D/C/Agg/Recall/NDCG/
Spearman) are only ever SAVED from the `test` split
(`stage1_metrics.json`), but the initial implementation computed the
expensive per-query `scipy.stats.spearmanr` diagnostic for train/val
splits too, discarding the result -- a pure waste that made cache
-building take 4+ minutes per arm in a tiny pipeline-integration test.
Fixed to only compute Spearman when `split == 'test'` before any real
run was launched -- cut cache-build time to ~1 minute in the same test,
with the saved `stage1_metrics.json` numerically unchanged (verified:
`test` split's `retmse10`/`agg_mse10` identical before and after the
fix in the integration test).

## Pre-execution equivalence checks (PART 8, 10 items)

All 10 written in `tests/test_u_asymmetry_capacity_decomposition01.py`,
16 test functions (some parametrized over all 4 arms), all pass:
U0 score/loss/gradient exactly match a direct `arm_score`+`kl_loss`
reference computation; projection parameter counts are exactly
`0/D^2/D^2/D^2` with no bias; U1/U2/U3's initial `W` tensors are
bit-identical (`torch.equal`, not just close); teacher has zero
arm-dependence; batch order is reproducible via the one shared loader
-construction path (all four arms use the same script); `compute_scores`
never touches candidate-mask/memory/split construction (source-level
check); query- and candidate-side encoder gradients are both nonzero
for every one of the four arms (isolated branch-detach test,
parametrized); K=10 ordinary Top-10 with the `Agg=D+C` identity holds
for every arm.

## Pipeline integration test (pre-flight, PART CLAUDE.md sanity-check requirement)

Ran a real 1-epoch/3-batch U0 and U2 training -> cache -> Stage2 chain
on GPU1 before launching the full 8-run (4 arms x 2 settings) job;
completed without error, `Agg=D+C` identity held, Stage2 produced a
sane `test_mse`. Smoke-tested all 4 arms' gradient flow (`--smoke_test`)
separately -- all PASS, `n_proj_params=16384` (`128^2`, matching
`d_model=128`) for U1/U2/U3, `0` for U0, identical `proj_init_sha`
across U1/U2/U3.

## STOP/VALIDITY checks (PART 20)

No trigger encountered pre-execution: parameter counts verified equal
across U1/U2/U3 and zero for U0; projection initialization verified
bit-identical across U1/U2/U3; encoder init hash shared via the same
`build_model` call every arm already uses session-wide; gradient flow
verified both branches nonzero for all four arms; teacher identical by
construction (one shared function, no arm parameter); candidate
memory/mask/split identical by construction (one shared
`build_experiment` call path, `compute_scores` never touches them);
checkpoint criterion identical (`min val retmse10`) across all four
arms; Stage2 base checkpoint is the same TRACK-R artifact already used
by every other track this session, referenced via one shared
`$R_BASE_CKPT` variable in the driver script's `for ARM in U0 U1 U2 U3`
loop (same pattern TRACK-T/TRACK-T2 used, same fairness guarantee).
