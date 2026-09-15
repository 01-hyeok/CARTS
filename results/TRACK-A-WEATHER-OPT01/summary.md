# TRACK-A-WEATHER-OPT01 -- summary

Engineering optimization only. No experimental hypothesis, dataset, horizon,
channel count, candidate pool, K, Oracle definition, loss, or research
support was changed. `FULL MEMORY -> DIRECT TOP-K` preserved throughout.

## A. Bottleneck (real profiler, both horizons, GPU 1, under contention
from TRACK-A-FACTORIAL-E2E01's Weather cells and TRACK-A-MULTIPOS-CHOICE01
running concurrently -- see caveat in the main report)

| Component | H96 % of iter (B0) | H720 % of iter (B0) |
|---|---:|---:|
| data_loading | 29.4% | 7.7% |
| query_encoder | 0.6% | 0.2% |
| candidate_encoder | 14.7% | 10.2% |
| host_scoring | 7.6% | 3.5% |
| **oracle_utility (Set Oracle)** | **37.1%** | **74.3%** |
| greedy_argmin | 4.8% | 1.4% |
| other (SetConditioner+score) | 5.8% | 2.7% |

Set Oracle utility computation is the single largest component at both
horizons, and dominates overwhelmingly at H720 (74%) -- confirms the
premise, does NOT need to be second-guessed by profiler-driven re-priority
(spec S19): Oracle really is the hotspot, more so as H grows.

## B. What changed

`utils/dense_utility_optimized.py` (NEW FILE; `utils/dense_utility.py`
untouched) reformulates `A(S+{i})` algebraically (derivation in the module
docstring, matches spec S3 exactly): `d_i = y_i - y_q` (prefix-invariant)
and its norm are computed ONCE per query instead of re-derived every one of
the K=10 greedy steps; each step then costs one batched dot product
(`E_S . d_i`, GEMV-shaped) plus O(N) scalar combination, instead of
materialising a full `[B, chunk, H]` intermediate (`trial_num`/`y_ret`) and
running an elementwise divide-subtract-square-mean chain on it. Real,
measured effect: **oracle-component speedup 2.4x (H96) / 11.5x (H720)**.

One genuine trade-off found and NOT hidden: peak GPU memory INCREASED
(1.56GB->2.47GB at H96, 11.0GB->17.5GB at H720), because
`prepare_query_static` holds `d` (same shape as `futures`) live for the
whole K=10 trajectory rather than the reference's per-step, per-chunk
temporaries. See "Suggested follow-up" in the main report -- not
implemented here (time-boxed), reported as a known limitation.

## C. Equivalence (real Weather data + a real trained checkpoint)

| | H96 (500 queries, `set_tf_cosine`, TRACK-A-FACTORIAL-E2E01) | H720 (300 queries, fresh scratch encoder, no trained arm exists yet) |
|---|---:|---:|
| max abs utility diff (valid positions) | 9.5e-7 | 2.9e-6 |
| selection agreement | 4999/5000 = **99.98%** | 3000/3000 = **100%** |
| max free-running aggregate MSE diff | 3.9e-7 | 0.0 |

The single H96 disagreement is a documented, investigated, EXACT float32
TIE under the reference (`utility_gap_under_reference = 0.0` bit-exact
between the two candidates) that floating-point summation-order breaks in
opposite directions under the two implementations (`utility_gap_under_optimized
= 1.16e-10`) -- resulting aggregate-MSE impact 3.9e-7, i.e. below the
tolerance in spec S15's own example, reported rather than ignored.

30-seed x 10-step synthetic stress test (bounded cosine-range scores,
matching production): worst valid-position diff 1.19e-6, **selection
agreement 100%** across all 300 greedy decisions.

A real, load-bearing bug in the FIRST draft of the optimized formula was
caught by this equivalence suite before any production use: an eps-clamp
applied AFTER squaring `(Z_S+w_i)` instead of before (matching the
reference's own clamp point), which silently corrupted low-weight
candidates' utility by up to several MSE units. Fixed; regression-tested.

## D. Speed

| | H96 | H720 |
|---|---:|---:|
| B0 (reference) sec/iter | 0.1850 | 0.6589 |
| B1 (optimized) sec/iter | 0.1149 | 0.1831 |
| **Total speedup** | **1.61x** | **3.60x** |
| Oracle-component-only speedup | 2.39x | 11.50x |
| B0 peak VRAM | 1563 MB | 11007 MB |
| B1 peak VRAM | 2475 MB (+58%) | 17548 MB (+59%) |

Full detail: `profile_reference_H{96,720}.json`, `profile_optimized_H{96,720}.json`,
`benchmark_H{96,720}.csv`, `equivalence_H{96,720}.json`.
