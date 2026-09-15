# TRACK-A-WEATHER-OPT01 -- exact optimization of the Weather Greedy Set Oracle

Engineering/optimization work, not a new research experiment. Goal: reduce
the wall-clock/VRAM cost of Weather's full-memory Greedy Set Oracle
computation while producing algebraically **identical** retrieval results.
`FULL MEMORY -> DIRECT TOP-K` preserved throughout: no candidate reduction,
no channel reduction, no shortlist, no ANN, no Oracle-definition change, no
loss change, no K reduction, no horizon change.

**Ran entirely read-only / inference-only against existing artifacts.**
Did not kill, modify, or overwrite any running experiment, checkpoint, or
result. `TRACK-A-FACTORIAL-E2E01`'s Weather cells (orchestrator pid
3116146) and `TRACK-A-MULTIPOS-CHOICE01` (pid 3818975) ran on GPU 1
throughout this work, per standing instruction (GPU 1 only) -- **every
profiling/benchmark number below was measured under that contention** and
absolute per-component percentages carry noise from it; the reference-vs-
optimized comparisons within each single benchmark run are still valid
(both paths measured back-to-back under the same contention level).

## Executed / Not executed

```
EXECUTED:
- Component-level profiling of the reference Set Oracle pipeline, H96 + H720
- Exact algebraic reformulation of the Set Oracle (utils/dense_utility_optimized.py)
- 23 unit equivalence tests (synthetic, multi-seed) + 30-seed x 10-step stress test
- Real-checkpoint, real-Weather-data end-to-end equivalence, H96 (500 queries)
  and H720 (300 queries, scratch encoder -- no trained H720 Set arm exists yet)
- B0 (reference) vs B1 (optimized) benchmark, H96 + H720
- One real numerical bug found and fixed BEFORE any production use (see below)

NOT EXECUTED:
- Chunking `prepare_query_static` to fix the peak-VRAM regression (reported,
  not fixed -- time-boxed; see "Suggested follow-up")
- BF16/mixed precision (B2) -- spec explicitly makes this secondary/optional;
  not reached given the scope already delivered
- Candidate-encoder caching across optimizer steps (spec S6's harder case --
  requires confirming gradient-graph safety per training script; not
  attempted here, encoder re-encoding was NOT the dominant bottleneck anyway)
- Production path switch -- `--set_oracle_impl` flag / wiring the optimized
  path into `train_factorial_e2e01.py`/`train_multipos_choice01.py`'s
  actual training loop. NOT done: those scripts are currently mid-run for
  two live experiments; per spec S22 ("기존 진행 중인 experiment ... config
  수정 금지") this is deliberately left as a followup requiring explicit
  approval, after those runs finish.

FILES CREATED:
- utils/dense_utility_optimized.py (reference utils/dense_utility.py UNTOUCHED)
- tests/test_dense_utility_optimized.py (23 tests)
- scripts/test_set_oracle_equivalence.py
- scripts/benchmark_weather_opt01.py
- results/TRACK-A-WEATHER-OPT01/{profile_reference,profile_optimized}_H{96,720}.json
- results/TRACK-A-WEATHER-OPT01/equivalence_H{96,720}.json
- results/TRACK-A-WEATHER-OPT01/benchmark_H{96,720}.csv
- results/TRACK-A-WEATHER-OPT01/summary.md

FILES MODIFIED: NONE (production files untouched)
EXISTING FILES OVERWRITTEN: NONE
PROTOCOL DEVIATIONS: NONE
```

## 1. Profiling (spec S1) -- real bottleneck, both horizons

Weather_96 (real trained `set_tf_cosine` checkpoint), 60 iterations;
Weather_720 (scratch encoder -- no trained H720 Set arm exists at
`TRACK-A-FACTORIAL-E2E01`'s current progress -- see method note below), 30
iterations. Component times via `torch.cuda.synchronize()`-bracketed
`time.perf_counter()`, one Weather channel (channel 0; all 21 channels are
processed by an outer Python loop in production, identical cost per
channel, so profiling one channel and scaling is representative of the
per-channel bottleneck shape, not the absolute per-batch wall clock).

| Component | H96 sec/iter (B0) | H96 % | H720 sec/iter (B0) | H720 % |
|---|---:|---:|---:|---:|
| data_loading | 0.0424 | 29.4% | 0.0430 | 7.7% |
| query_encoder | 0.0009 | 0.6% | 0.0012 | 0.2% |
| candidate_encoder | 0.0213 | 14.7% | 0.0570 | 10.2% |
| host_scoring | 0.0109 | 7.6% | 0.0196 | 3.5% |
| **oracle_utility (Set Oracle, K=10 total)** | **0.0535** | **37.1%** | **0.4163** | **74.3%** |
| greedy_argmin | 0.0069 | 4.8% | 0.0078 | 1.4% |
| other (SetConditioner + score head) | 0.0083 | 5.8% | 0.0152 | 2.7% |
| **Total** | **0.1850** | 100% | **0.6589** | 100% |

**Set Oracle utility computation is the largest single component at both
horizons and dominates overwhelmingly at H720 (74%).** Per spec S19's own
instruction to defer to the profiler rather than assume: the profiler
confirms the Set Oracle IS the right target, more so as H grows -- no
re-prioritization to encoder/transfer optimization was warranted.
`candidate_encoder` is the second-largest component (10-15%); it is called
exactly ONCE per iteration already (outside the K=10 loop, in both the
reference and this profiling harness), so no stale/repeated re-encoding
within a single greedy trajectory was found -- see section 6 discussion.

## 2. Current Greedy Set Oracle implementation (spec S2)

`utils/dense_utility.py::dense_utility` (read, not modified). Uses the
existing project identity `Z_S = sum_{j in S} w_j`, `M_S = sum_{j in S}
w_j*Y_j`, `A(S+{i}) = MSE((M_S+w_i*Y_i)/(Z_S+w_i), Y_q)` -- already avoids
a naive `[B,N,H]` full materialization by chunking over candidates, but
**every greedy step reconstructs the full aggregate `y_ret` `[B, chunk, H]`
from scratch** (elementwise add, elementwise multiply, divide, subtract,
square, mean-reduce over H) for every chunk, every step, every channel.
Nothing about `y_i` (the candidate future) is precomputed or reused across
the K=10 steps of one query's trajectory. This matches the profiling
result exactly: the per-step `[B,chunk,H]` materialization-and-reduction is
the expensive part, worse as H grows (H720 vs H96: oracle_utility grew from
0.053s to 0.416s, ~7.8x, roughly tracking the ~7.5x H increase).

Weight semantics confirmed (spec S4): `candidate_weights` (`utils/dense_utility.py:67`)
computes `w_i = exp((s_i-max)/tau)`, zeroed at invalid positions; the
Stage-2 host's own forced-selection aggregation
(`utils/retrieval_ops.py:90-94`) uses the identical `softmax(score/tau)`
weighting. The Set Oracle in every current Track-A experiment uses this
SAME host-fixed weighting (not each arm's own score) as the aggregation --
confirmed unchanged, not touched by this work.

## 3-4. Exact algebraic reformulation (spec S3-S4)

`utils/dense_utility_optimized.py` (new file; `dense_utility.py` left
completely untouched, spec S13). Full derivation in the module docstring.
Summary: with `d_i = y_i - y_q` (candidate residual against the query
future, **prefix-invariant**) and `E_S = M_S - Z_S*y_q = sum_{j in S}
w_j*d_j` (= `R_S` in the pre-registered spec, expressed via the project's
own `M_S`/`Z_S` naming so `prefix_weighted_sums` -- the ALREADY-tested
production helper -- can be reused unmodified on `d` instead of `futures`,
rather than reimplementing it):

```
A(S+{i}) = [ ||E_S||^2 + 2*w_i*(E_S . d_i) + w_i^2*||d_i||^2 ] / (H*(Z_S+w_i)^2)
```

Algebraically IDENTICAL to `dense_utility` for every candidate, step, and
query -- not an approximation, derivation verified exact by 23 passing
equivalence tests plus a 30-seed stress sweep (section 5).

`||d_i||^2` and `d_i` itself are computed ONCE per query
(`prepare_query_static`) and reused verbatim across all K=10 greedy steps
-- confirmed by a dedicated test that two separate calls to
`prepare_query_static` return bit-identical tensors and that the function
signature has no prefix argument at all. Per step, the candidate loop
becomes one batched dot product (`torch.einsum('bh,bnh->bn', E_S, d_c)`,
GEMV-shaped) plus O(chunk) scalar combination -- no `[B,chunk,H]`
intermediate is ever materialized for `y_ret`. Chunking preserved exactly
(spec S7-S8): every candidate is visited in the same chunk order, output
shape/semantics identical, `argmin` on either function's output selects the
same candidate (verified, section 5) -- no approximate search, ANN,
shortlist, or prefiltering introduced anywhere.

**One numerically dangerous special case, found and handled (not by luck):**
at an empty prefix (t=1), `Z_S = E_S = 0` exactly, and the general formula's
`w_i^2` term is numerically fragile -- `candidate_weights`' softmax
numerator spans many orders of magnitude, and squaring a very small (but
legitimately nonzero) weight can underflow to exactly `0.0` in float32
while `w_i` itself would not, corrupting the ratio. Handled by an EXACT
closed-form special case (`A(i) = ||d_i||^2/H`, independent of `w_i`, which
is what the general formula provably converges to at `Z_S=0`) rather than
by evaluating the fragile general formula and hoping. This was caught by
the equivalence tests, not assumed correct by construction.

**A second, more consequential bug was caught and fixed before any
production use:** the first draft applied the eps-clamp (needed to avoid
literal division-by-zero) AFTER squaring `(Z_S+w_i)`, where the reference
applies its own clamp BEFORE the implicit squaring (inside the MSE). For
small-but-representable `(Z_S+w_i)` values (e.g. a prefix containing a
near-zero-weight candidate), squaring-then-clamping distorted the result by
up to several MSE units versus squaring-after-clamping, which matches the
reference exactly. Fixed to clamp `(Z_S+w_c)` before squaring, matching the
reference's own clamp point precisely; regression-tested (`den = h*(z_s +
w_c).clamp_min(eps).pow(2)`).

## 5. Equivalence (spec S14-S16)

**Unit level** (`tests/test_dense_utility_optimized.py`, 23 tests, all
passing): exact equality at every valid position across prefix lengths
{0,1,3,7} x chunk sizes {None,17,64,200}; full K=10 greedy-trajectory
selection agreement (synthetic); masked-candidate exclusion; chunking
never drops a candidate; the empty-prefix special case; a **documented,
investigated, and explicitly bounded** floating-point corner case in the
REFERENCE implementation itself (eps-clamp asymmetry for weights below
1e-12 under unbounded scores) that is mathematically proven unreachable
for every current Track-A experiment, all of which fix Score=cosine
(bounded [-1,1]) at tau=0.1 -- worst-case weight ratio `exp(-2/0.1) ~=
2.06e-9`, three orders of magnitude above the 1e-12 eps floor.

**30-seed x 10-step synthetic stress test** (bounded, production-realistic
score range): worst valid-position utility diff **1.19e-6**, selection
agreement **100%** (300/300 greedy decisions across 30 independent seeds).

**Real end-to-end, real Weather checkpoint** (spec S16 -- not unit-level
only): `scripts/test_set_oracle_equivalence.py`, loading
`TRACK-A-FACTORIAL-E2E01`'s own trained `set_tf_cosine` Weather_96
checkpoint (Greedy Set Oracle, on-policy, cosine -- the exact production
configuration this work targets), full K=10 free-running trajectories on
REAL test-split Weather data:

| | H96 (500 queries, trained checkpoint) | H720 (300 queries, scratch encoder) |
|---|---:|---:|
| max abs utility diff, valid positions | 9.5e-7 | 2.9e-6 |
| selection agreement | 4999/5000 = **99.98%** | 3000/3000 = **100%** |
| max free-running aggregate MSE diff | 3.9e-7 | 0.0 |

H720 used a fresh, untrained scratch encoder (spec S16 does not require a
trained checkpoint for a numerical-equivalence/speed check, and
`TRACK-A-FACTORIAL-E2E01` has not reached a trained Weather_720 Set arm at
the time of this work -- using it, rather than waiting, avoids blocking on
or disturbing that still-running experiment). The Oracle-equivalence
mathematics do not depend on encoder quality.

**The single H96 disagreement, investigated per spec S15 ("그냥 무시하지
마세요"), not swept aside:** an EXACT float32 tie under the reference
(candidates 7698/7699 at step t=8, `utility_gap_under_reference = 0.0`
bit-exact) that floating-point summation-order breaks in opposite
directions under the two paths (`utility_gap_under_optimized = 1.16e-10`).
Resulting aggregate-MSE impact: 3.9e-7 -- negligible, and this is precisely
the "floating tie" scenario the spec anticipated, not a systematic
disagreement.

## 6. Speed (spec D, S17-S18)

| | H96 | H720 |
|---|---:|---:|
| B0 (reference) sec/iter | 0.1850 | 0.6589 |
| B1 (optimized) sec/iter | 0.1149 | 0.1831 |
| **Total iteration speedup** | **1.0x -> 1.61x** | **1.0x -> 3.60x** |
| Oracle-component-only speedup | 2.39x | **11.50x** |
| B0 peak VRAM | 1564 MB | 11007 MB |
| B1 peak VRAM | 2475 MB | 17548 MB |

Speedup grows sharply with H (2.4x -> 11.5x oracle-only from H96 to H720),
exactly as the derivation predicts: the eliminated per-step `[B,chunk,H]`
materialization scales with H, while the retained per-step cost (a GEMV)
scales the same way but with far better constants and no large-tensor
allocation churn.

**Peak VRAM increased, not decreased, at both horizons (+58-59%) -- reported
honestly, not hidden.** Root cause: `prepare_query_static` holds `d` (same
shape as the `futures` input) live for the entire K=10 trajectory, on top
of `futures` itself, roughly doubling that one tensor's footprint for the
trajectory's duration; the reference only ever holds per-chunk temporaries
that are freed after each step. This is a genuine, measured trade-off
(wall-clock for memory), not an artifact of contention -- reproduced
consistently at both horizons with the same relative magnitude. **Not
fixed here** (time-boxed): the fix (chunk `prepare_query_static` too, or
recompute `d` per-chunk-per-step from `futures` directly, forfeiting part
of the "compute once" benefit for a smaller peak footprint) is a
straightforward follow-up, offered as a suggestion below, not implemented
without approval.

## 7. Channel vectorization, transfer, allocation (spec S9-S11)

**Not implemented in this round**, reported rather than silently skipped.
The profiler (section 1) shows `candidate_encoder`+`host_scoring`+
`data_loading` together account for 30-50% of iteration time at both
horizons -- real, but smaller than the Oracle component this work already
addressed, and each requires touching the production training-loop
structure (`train_factorial_e2e01.py`'s per-channel Python loop) rather
than a self-contained new module, which raises the risk of disturbing the
two experiments currently running on that exact code path. Left as an
explicitly separate follow-up (below), not attempted here.

## 8. Mixed precision (spec S12, B2)

**Not reached.** Spec explicitly marks this secondary/optional relative to
the FP32-exact optimization (S12: "BF16은 exact optimization의 필수 조건이
아닙니다. 먼저 FP32 optimized implementation이 ... 동일함을 검증하세요");
that FP32 verification (sections 5-6) is complete and is the deliverable
this round prioritizes.

## 9. Optional flag / production path (spec S13, S22)

The optimized path exists ONLY as the separate, importable
`utils/dense_utility_optimized.py` module plus two standalone verification
scripts. **It is not wired into any production training script.**
`train_factorial_e2e01.py`, `train_multipos_choice01.py`, and every other
existing trainer are completely unmodified and continue to call
`dense_utility` (or `greedy_set_utility`, which itself calls
`dense_utility`) exactly as before. Switching a production path to the
optimized implementation, or adding a `--set_oracle_impl
{reference,optimized}` flag to the trainers, is deliberately left for a
separate, explicitly-approved change -- per spec S22, the two experiments
currently mid-run must not have their code paths touched.

## Suggested follow-up (NOT executed -- offered only)

1. **Fix the peak-VRAM regression**: chunk `prepare_query_static` (or fold
   `d`'s construction into the existing per-step chunk loop, recomputing it
   from `futures` each step -- sacrificing part of the "compute once"
   saving for a smaller footprint) so B1 no longer exceeds B0's peak VRAM.
   Cheapest, highest-value next step given the numbers above.
2. **Wire the verified-exact optimized path into `train_factorial_e2e01.py`
   / `train_multipos_choice01.py` behind an explicit `--set_oracle_impl`
   flag**, defaulting to `reference` (spec S13), only after (1) and only
   for cells/arms not currently running.
3. **`candidate_encoder` caching** (spec S6): it is already called once per
   iteration, not per greedy step, so the remaining question is whether it
   can be cached ACROSS optimizer steps for a fixed checkpoint (evaluation/
   validation only, per spec S20) -- worth a separate, smaller change once
   (1)-(2) are settled.
4. Channel vectorization (spec S9) -- the second-largest untouched
   component; requires touching the production per-channel loop, so should
   follow, not precede, (2).

## Final answers (spec's 7 questions)

1. **What was the real bottleneck?** Set Oracle utility computation --
   37% of iteration time at H96, **74% at H720** -- confirmed by direct
   component profiling, not assumed.
2. **Is the algebraic reformulation exactly applicable?** Yes, exactly --
   derived from the project's own existing `Z_S`/`M_S` identity, verified
   by 23 unit tests, a 30-seed stress sweep, and two real-checkpoint
   end-to-end runs. Two real implementation bugs (an eps-clamp-order error,
   and a numerically fragile empty-prefix special case) were caught by
   this verification process before either could have reached production.
3. **What is the reference-vs-optimized selection agreement?** **99.98%**
   at H96 (4999/5000, the one disagreement a documented exact float32 tie
   with negligible aggregate impact) and **100%** at H720 (3000/3000).
4. **How much did H96/H720 runtime improve?** Total iteration: **1.61x**
   (H96), **3.60x** (H720). Oracle-component-only: **2.39x** (H96),
   **11.50x** (H720).
5. **How much did peak VRAM improve?** It did **not** -- it increased
   58-59% at both horizons, a real and reported trade-off, root-caused in
   section 6, with a concrete un-implemented fix suggested above.
6. **Can BF16 be used without changing selection?** Not evaluated this
   round (spec marks it secondary; not reached within scope).
7. **Can the optimized path be safely applied to existing Weather
   experiments going forward?** For the Oracle computation specifically:
   the evidence supports yes, PENDING the VRAM fix (suggestion 1) and an
   explicit approval + `--set_oracle_impl` flag (suggestion 2) rather than
   a silent default switch, and only once the two currently-running
   experiments are no longer using the code paths this would touch.
