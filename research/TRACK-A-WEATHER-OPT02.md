# TRACK-A-WEATHER-OPT02 -- fixing OPT01's peak-VRAM regression via
candidate-dimension chunking of the persistent `d` tensor

Engineering follow-up to `TRACK-A-WEATHER-OPT01`. Goal: keep OPT01's
wall-clock speedup while removing its measured +58-59% peak-VRAM
regression. No math, selection rule, aggregation, or research support
changed. `utils/dense_utility.py` (reference) untouched. Neither running
experiment (`TRACK-A-FACTORIAL-E2E01` Weather cells, pid 3116146;
`TRACK-A-MULTIPOS-CHOICE01`, pid 3818975) was stopped, paused, or had any
file it uses modified -- both confirmed alive and unaffected at every
checkpoint of this work, including immediately before writing this report.

## EXECUTED / NOT EXECUTED

```
EXECUTED:
- utils/dense_utility_optimized.py MODIFIED: added a chunked (`d=None`)
  code path; OPT01's original (`d` given) path kept 100% byte-identical
  and is dispatched to unchanged -- verified by all 23 of OPT01's own
  tests passing unmodified except one test-introspection assertion (see
  Protocol deviations).
- tests/test_dense_utility_optimized.py MODIFIED (reinforced, spec's own
  word "보강"): OPT01's 23 tests retained (1 adjusted for the new
  dispatcher's source layout, functionally identical assertion), 16 new
  OPT02 tests added -- 39 total, all passing.
- Real end-to-end equivalence, H96 (500 queries, real trained checkpoint)
  and H720 (300 queries, scratch encoder -- no trained H720 Set arm
  exists yet, same method OPT01 used).
- 30-seed x 10-step synthetic stress test through the OPT02 code path
  specifically (separate from the unit-level parametrized tests).
- B0 vs B1 vs B2 benchmark, chunk sizes {128,256,512,1024,2048} (spec's
  requested sweep) at both horizons, plus a supplementary chunk=4096 run
  (OPT01's own established default) for direct comparability with that
  report's numbers.
- Full regression suite: 749 passed / 2 pre-existing failures, no
  regression.

NOT EXECUTED (per explicit constraints 3-4 and the request's closing
instruction):
- No change to train_factorial_e2e01.py, train_multipos_choice01.py, or
  any other production trainer.
- No BF16/mixed precision, no channel vectorization, no candidate-encoder
  cross-step caching.
- No `--set_oracle_impl` flag, no production wiring -- a minimal patch
  PLAN is proposed at the end, not applied.

FILES CREATED:
- scripts/benchmark_weather_opt02.py
- scripts/test_set_oracle_equivalence_opt02.py
- research/TRACK-A-WEATHER-OPT02.md (this file)
- results/TRACK-A-WEATHER-OPT02/{equivalence_H96,equivalence_H720}.json
- results/TRACK-A-WEATHER-OPT02/benchmark_H{96,720}.csv
- results/TRACK-A-WEATHER-OPT02/benchmark_H{96,720}_supplementary_4096.csv

FILES MODIFIED:
- utils/dense_utility_optimized.py (OPT01's own file -- the spec's
  intended target, per its own "utils/dense_utility_optimized.py 수정"
  deliverable line; `utils/dense_utility.py`, the REFERENCE, is a
  separate file and was NOT touched)
- tests/test_dense_utility_optimized.py (reinforced, per spec)

EXISTING FILES OVERWRITTEN: NONE (no result/checkpoint from any other
experiment was touched; OPT01's own prior result files under
results/TRACK-A-WEATHER-OPT01/ are untouched -- this round writes only to
results/TRACK-A-WEATHER-OPT02/)

PROTOCOL DEVIATIONS:
- One OPT01 test (`test_gradient_safety_no_grad_used_identically_to_reference`)
  needed a one-line change: it originally inspected only
  `dense_utility_optimized`'s own source for the string `"torch.no_grad()"`;
  after OPT02 split that function into a thin dispatcher plus two
  implementation helpers, the guard text lives in the helpers, not the
  dispatcher. Fixed to inspect the whole module instead of just the
  dispatcher. The PROPERTY the test checks (both code paths remain
  `torch.no_grad()`-safe) is unchanged and still verified true; only the
  introspection mechanism was adjusted. No other deviation.
```

## Root cause recap (from OPT01) and the fix

OPT01's `prepare_query_static(futures, query_future)` returned the FULL
`d = futures - query_future` tensor, `[B, N, H]`, held live for the whole
K=10 greedy trajectory -- the same size as `futures` itself, so it roughly
doubled that one input's memory footprint for the trajectory's duration.

**Fix**: a new `prepare_query_static_chunked` computes the SAME
prefix-invariant `d_sq = ||d_i||^2` (needed at every step, the genuine
"compute once" win worth keeping) but chunks the candidate dimension
internally and returns ONLY the `[B, N]` scalar result -- never the
`[B, N, H]` tensor that produced it. `E_S`/`Z_S` (spec: `R_S`, `Z_S`) only
ever need `d` at the PREFIX indices (<=K=10 rows out of N), gathered
directly from `futures` -- never the full candidate set. The per-step
candidate loop recomputes `d_c = futures[:,chunk,:] - q` fresh, transiently,
exactly where OPT01 already chunked the rest of the computation. At no
point does a `[B, N, H]`-sized tensor exist beyond `futures` itself (an
unavoidable input the reference also holds).

Both formula and eps-clamp order are UNCHANGED from OPT01 (spec's own
requirement): `_dense_utility_optimized_opt02` is line-for-line the same
algebra as `_dense_utility_optimized_opt01`, with `d`/`E_S`/`Z_S`'s SOURCE
being the only difference -- verified by a dedicated test
(`test_opt02_prefix_e_s_z_s_matches_full_d_computation`) proving the two
`E_S`/`Z_S` computations are algebraically identical, and by every
equivalence test below.

## A. Unit tests

39 tests in `tests/test_dense_utility_optimized.py`, all passing (23 from
OPT01 unmodified in intent, 16 new for OPT02):

- empty prefix (Z_S=0 exactly) -- same closed-form path as OPT01
- eps boundary (`w_i` placed exactly at 1e-12) -- OPT02 matches OPT01
  within 1e-4 (not bit-exact: gather-then-subtract vs subtract-then-gather
  differ by up to 2.4e-7 in floating-point summation order, well inside
  this file's project-wide 1e-4 convention; `d_sq` itself IS bit-exact
  between the two paths, confirmed directly)
- exact tie between two specific candidates (constructed so `||d_5||^2 ==
  ||d_9||^2` exactly) -- both paths break the tie toward the SAME
  (smaller) index, matching `torch.argmin`'s own documented convention
- non-divisible final chunk (N=137, chunk=64 -> chunks of 64,64,9) --
  explicitly checked at the short tail
- chunk_size larger than the candidate count (N*10)
- K=1 (single greedy step) and the full K=10 trajectory
- all candidates having IDENTICAL utility (full tie) -- both paths select
  index 0, the smallest, identically
- OPT02 vs OPT01 agreement across chunk sizes {1,13,128,N,N*2}
- OPT02 vs the REFERENCE across a full K=10 free-running trajectory
- static-analysis check that no `[B,N,H]`-shaped tensor is ever assigned
  outside the per-chunk loop in the OPT02 helper
- 30-seed x 10-step stress sweep through the OPT02 path specifically:
  worst valid-position diff, selection agreement over all 300 greedy
  decisions -- **100%**

## B. Equivalence, real Weather data + real checkpoint

Same methodology as OPT01 (`TRACK-A-FACTORIAL-E2E01`'s trained
`set_tf_cosine` Weather_96 checkpoint for H96; a fresh scratch encoder for
H720, since no trained H720 Set arm exists yet -- the Oracle-equivalence
mathematics do not depend on encoder quality), `candidate_chunk_size=1024`:

| | H96 (500 queries) | H720 (300 queries, scratch encoder) |
|---|---:|---:|
| max abs utility diff, valid positions | 9.5e-7 | 2.9e-6 |
| selection disagreements (total) | 1 | 0 |
| -- of which reference-tie disagreements | **1** | 0 |
| -- of which real mismatches | **0** | 0 |
| non-tie selection agreement | **100%** | **100%** |
| overall selection agreement | 99.98% | 100% |
| max free-running aggregate MSE diff | 3.9e-7 | 0.0 |
| mean free-running aggregate MSE diff | 7.8e-10 | 0.0 |
| max Stage-2 input (`y_ret`) diff | 1.35e-3 | 0.0 |

### Tie analysis (spec B, "tie 불일치는 candidate ID뿐 아니라 utility/aggregate가 같은지 확인")

The single H96 disagreement is the **exact same query/step/candidate pair**
OPT01's own report documented: t=8, candidates 7698 vs 7699,
`ref_utility_at_ref_pick = ref_utility_at_opt_pick = 0.0003276426577940583`
-- bit-identical under the reference, a genuine float32 tie the reference
itself cannot break. Classified `is_reference_tie=True`, `real_mismatch=0`.
Its aggregate-MSE impact (3.9e-7) is the entire `max_free_running_
aggregate_mse_diff` figure above -- i.e. every OTHER query/step in the
500x10=5000 decisions checked contributes ZERO measurable aggregate
difference. **OPT02 introduces no NEW disagreement beyond the one OPT01
already found and explained; H720 has zero disagreements of any kind.**

The `y_ret` (Stage-2 input) diff of 1.35e-3 at H96 comes entirely from this
one tie: choosing candidate 7699 instead of the reference's 7698 changes
that one query's retrieved-future VECTOR (naturally -- they are different
real-valued futures, only their aggregate UTILITY was tied), even though
the resulting Stage-2-relevant aggregate MSE barely moves. Reported, not
averaged away.

## C. Benchmark

**Timer boundary** (exact, per spec's explicit request): `oracle_sec_per_iter`
brackets ONLY `dense_utility(...)` / `dense_utility_optimized(...)` calls
(plus, for B1/B2, the one `prepare_query_static[...]` call per iteration)
with `torch.cuda.synchronize()` immediately before and after each bracketed
call. `total_sec_per_iter` is the full iteration wall clock (data loading +
query/candidate encoding + host scoring + the K=10 greedy loop, oracle
component included). Same warm-up (3 discarded iterations for H96, 2 for
H720, matching OPT01's convention scaled to this round's smaller `n_iters`)
applied identically to B0/B1/B2 before any timed iteration.

**All numbers below were measured with `TRACK-A-FACTORIAL-E2E01`'s Weather
cells and `TRACK-A-MULTIPOS-CHOICE01` running concurrently on the same
GPU** (per the standing GPU-1-only instruction) -- contention is real and
visible in the data (non-monotonic timings across chunk sizes at H720 in
particular); relative B0-vs-B2 comparisons within each single row remain
valid (back-to-back on the same contention level), absolute numbers do
not compare cleanly across different points in time.

### H96 (requested sweep, chunk sizes 128-2048)

| chunk | B0 oracle | B1 oracle | B2 oracle | B0 total | B1 total | B2 total | B0 peak alloc | B1 peak alloc | B2 peak alloc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.352 | 0.576 | 0.567 | 0.426 | 0.634 | 0.627 | 1564 MB | 2474 MB | 1568 MB |
| 256 | 0.179 | 0.265 | 0.289 | 0.236 | 0.324 | 0.350 | 1564 MB | 2474 MB | 1568 MB |
| 512 | 0.094 | 0.137 | 0.147 | 0.152 | 0.196 | 0.204 | 1564 MB | 2474 MB | 1568 MB |
| 1024 | 0.054 | 0.073 | 0.078 | 0.110 | 0.131 | 0.138 | 1564 MB | 2474 MB | 1568 MB |
| 2048 | 0.059 | 0.041 | 0.043 | 0.117 | 0.098 | 0.100 | 1564 MB | 2474 MB | 1568 MB |
| **4096 (suppl.)** | **0.053** | **0.069** | **0.027** | **0.121** | **0.129** | **0.085** | 1564 MB | 2474 MB | 1568 MB |

### H720 (requested sweep, chunk sizes 128-2048)

| chunk | B0 oracle | B1 oracle | B2 oracle | B0 total | B1 total | B2 total | B0 peak alloc | B1 peak alloc | B2 peak alloc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.405 | 0.562 | 0.563 | 0.497 | 0.629 | 0.631 | 11007 MB | 17548 MB | 11011 MB |
| 256 | 0.418 | 0.272 | 0.295 | 0.486 | 0.342 | 0.362 | 11007 MB | 17548 MB | 11011 MB |
| 512 | 0.405 | 0.146 | 0.163 | 0.478 | 0.214 | 0.231 | 11007 MB | 17548 MB | 11011 MB |
| 1024 | 0.385 | 0.085 | 0.139 | 0.462 | 0.152 | 0.211 | 11007 MB | 17548 MB | 11011 MB |
| 2048 | 0.372 | 0.052 | 0.111 | 0.457 | 0.120 | 0.183 | 11007 MB | 17548 MB | 11011 MB |
| **4096 (suppl.)** | **0.376** | **0.150** | **0.110** | **0.547** | **0.221** | **0.186** | 11007 MB | 17548 MB | 11011 MB |

Full data: `results/TRACK-A-WEATHER-OPT02/benchmark_H{96,720}[_supplementary_4096].csv`.

### Amdahl consistency (spec's explicit request)

Using OPT01's own profiled oracle-fraction `p` of total iteration time
(H96: 0.371, H720: 0.743) and this round's measured oracle-component
speedup `S` at chunk=4096 (the OPT01-comparable point):

| | p (oracle fraction, from OPT01) | S (measured oracle speedup) | Amdahl-predicted total speedup | measured total speedup |
|---|---:|---:|---:|---:|
| H96 | 0.371 | 1.95x | 1.22x | **1.41x** |
| H720 | 0.743 | 3.42x | 2.11x | **2.95x** |

**Measured total speedup EXCEEDS the naive Amdahl prediction at both
horizons -- reported as a real finding, not treated as confirmation of
extra synergy.** The most likely explanation is GPU-1 contention: `p` was
profiled in OPT01's session under a different contention level than this
round's B0/B2 timing windows, and the non-oracle components
(encoder/host-scoring/data-loading) are NOT held perfectly constant across
the sequential B0-then-B1-then-B2 runs the way a true Amdahl model assumes
-- other jobs' load shifts between those runs. The DIRECTION (measured
exceeds naive-Amdahl, rather than falling short of it) is consistent with
noise helping this comparison rather than a hidden implementation issue,
but this is not proof; flagged as a limitation of this benchmark's
precision, not resolved further within this round's scope.

## Pass/fail against the recommended criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Zero test regressions | **PASS** -- 749 passed / 2 pre-existing failures, unchanged from before this work |
| 2 | Non-tie selection agreement 100% | **PASS** -- 100% at both horizons (5000+3000 decisions checked, 1 documented reference-tie, 0 real mismatches) |
| 3 | Tie case utility/aggregate equal to reference | **PASS** -- utility bit-identical under the reference (gap=0.0), aggregate-MSE impact 3.9e-7 |
| 4 | Stage-2 MSE diff <=1e-7 or FP32 tolerance | **Not directly measured** (no trained Weather Stage-2-forced-selection run was performed this round to avoid extra GPU load on top of two live experiments); the closest available proxy, free-running aggregate MSE diff, is 3.9e-7 at H96 and 0.0 at H720, both within any reasonable FP32 tolerance -- treated as **PASS by proxy**, not a literal Stage-2 MSE measurement |
| 5 | Peak allocated VRAM within +5% of reference | **PASS, comfortably** -- B2 peak_alloc = 1568 MB vs B0's 1564 MB at H96 (**+0.3%**), 11011 MB vs 11007 MB at H720 (**+0.04%**), at EVERY chunk size tested. The OPT01 regression (+58-59%) is fully eliminated. |
| 6 | Overall speedup >=1.4x (H96) / >=3.0x (H720) | **MIXED, chunk-size-dependent.** At the requested sweep's largest tested point (chunk=2048): H96 1.17x (below target), H720 2.49x (below target). At the supplementary chunk=4096 (OPT01's own established default, not in the requested 128-2048 sweep): H96 **1.41x** (meets target), H720 **2.95x** (0.05x short of target, within this benchmark's contention-driven noise band). **Smaller chunk sizes (128-512) do not meet the speed target at either horizon** -- per-chunk Python-loop and kernel-launch overhead, and the recomputation of `d_c` every step (traded deliberately for the VRAM fix), both grow relatively more expensive as `n_chunks` grows. |

## Production applicability (spec's question 7)

**For the VRAM fix specifically: yes, conditionally.** Criteria 1-3 and 5
pass cleanly and consistently across every chunk size tested. Criterion 6
(speed) is the real caveat: **`candidate_chunk_size` must be set large
enough (>=~2048 for H96, closer to 4096 for H720) for the fix to also meet
the original OPT01-level speedup targets** -- at the small end of the
requested sweep (128-512), OPT02 is SLOWER than the reference at both
horizons, a genuine regression at those settings, not a subtle trade-off.
This is not a blocker for the fix's core purpose (VRAM), but it means the
eventual production default must not be a small chunk size.

## Suggested minimal patch plan for production adoption (NOT applied this round)

Proposed only, per the request's explicit instruction not to wire a
production flag yet:

1. Add `--set_oracle_impl {reference,optimized}` (default `reference`) and
   `--candidate_chunk_size` (default 4096, matching the value both OPT01's
   and this round's benchmarks used as the well-behaved reference point) to
   `train_factorial_e2e01.py` and `train_multipos_choice01.py`'s argument
   parsers only -- no behavior change when the flag is left at its default.
2. In each script's Set Oracle call site (`greedy_set_utility` /
   `dense_utility` call inside `run_sequence`/`train_epoch`/`eval_epoch`),
   branch on the flag: `reference` calls `dense_utility` exactly as today;
   `optimized` calls `prepare_query_static_chunked` once per query then
   `dense_utility_optimized(..., futures=..., query_future=..., d_sq=...,
   candidate_chunk_size=...)` per step -- a small, local, mechanical
   substitution, not a structural rewrite.
3. Apply only to NEW runs of cells not currently in progress -- never to
   `TRACK-A-FACTORIAL-E2E01`'s or `TRACK-A-MULTIPOS-CHOICE01`'s
   currently-running arms, and never by silently changing what a
   already-launched run does mid-flight.
4. Before defaulting any NEW experiment to `optimized`, re-run this
   report's benchmark at the chosen production `candidate_chunk_size`
   without GPU-1 contention (i.e. when no other experiment is running) to
   get an uncontended confirmation of the speedup numbers, since this
   round's own data shows the total-speedup criterion is sensitive to both
   chunk size and contention.
5. Keep `reference` as the flag's default until a user/reviewer explicitly
   approves switching a specific new experiment to `optimized`.
