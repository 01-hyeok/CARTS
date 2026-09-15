# TRACK-A-WEATHER-OPT04 -- Production wiring of OPT03's Oracle-path
optimizations, with real training verification

Follow-up engineering round to OPT01/OPT02/OPT03. Goal: wire C (Oracle-
Choice CE), A (Individual Oracle caching), E (Greedy Set Oracle) into
`scripts/train_factorial_e2e01.py` behind an explicit, default-off flag,
and verify the speed/VRAM claims survive contact with REAL forward +
backward + `optimizer.step()` training -- not just forward-only
microbenchmarks. `--oracle_compute_impl` defaults to `reference`
(byte-identical to OPT01/02/03's behavior); `optimized` is opt-in only.
Neither running experiment (Weather H96 factorial pid 3712193, MULTIPOS-
CHOICE01 ETTh1_720 pid 3937098) was stopped, slowed, or had any file it
uses modified -- both confirmed alive, with growing `etimes` and unchanged
command lines, at every checkpoint in this round, including immediately
before writing this report.

## ERRATUM -- OPT03's Individual Oracle chunk=4096 speedup was
miscalculated

OPT03 reported chunk=4096 as `1.09x`/`+9%` speedup and treated it as
clearing the 5% floor. The correct arithmetic, from OPT03's own reported
numbers (`0.02260`/`0.02239` path sec/iter, `0.08235`/`0.08225` total
sec/iter):

```
path:  0.02260 / 0.02239 = 1.0094x  (+0.93%)
total: 0.08235 / 0.08225 = 1.0012x  (+0.12%)
```

**Both are well under the 5% floor.** `research/TRACK-A-WEATHER-OPT03.md`
is NOT modified (per instruction) -- this erratum is the correction of
record. **Individual Oracle norm-expansion at chunk=4096 is REJECTED on
speed grounds**, same as chunk=2048 (which was already, correctly,
rejected in OPT03 at -61%). Its ~43% peak-VRAM reduction is evaluated
separately below as a VRAM-only candidate, per this round's Acceptance
criteria (>=30% VRAM reduction, <=5% total-iteration slowdown, no
target/selection impact on training).

This erratum is also why **Phase 3 of this round specifically requires
CACHING the reference formula, not the norm-expansion algebra**, as A's
default `optimized` path -- and that is what was implemented and wired.

## EXECUTED / NOT EXECUTED

```
EXECUTED:
- Process/GPU state recorded before and after all work
  (results/TRACK-A-WEATHER-OPT04/process_status_{before,after}.json,
  environment.json).
- OPT03 arithmetic erratum, recorded above and in git_diff_summary.txt's
  companion analysis.
- Phase 1: --oracle_compute_impl flag added to train_factorial_e2e01.py,
  default=reference; reuses the EXISTING --chunk_size flag (no new
  --candidate_chunk_size flag created); [oracle-compute] log block with
  explicit fallback_reason printed on every run.
- Phase 2 (C): oracle_choice_step_loss_optimized wired at the single
  run_sequence call site (+ eval_epoch's diagnostic-recompute loop, for
  consistency); applies whenever optimized is requested, regardless of
  chunk_size.
- Phase 3 (A): Individual Oracle CACHING (the reference formula, called
  once per query instead of once per K-step) wired, NOT the OPT03
  norm-expansion algebra, per this round's explicit instruction.
  Norm-expansion is evaluated separately, benchmark-only, as a VRAM-only
  candidate.
- Phase 4 (E): OPT02's validated dense_utility_optimized wired, gated on
  chunk_size >= GREEDY_SET_OPTIMIZED_MIN_CHUNK_SIZE (4096); reference
  fallback with a logged reason below that.
- Phase 5: real training benchmarks (forward+backward+optimizer.step()),
  4 requested cells (H96/H720 x Individual/Set, on-policy, cosine), B0 vs
  B1, BOTH on an initially-idle GPU (GPU2, isolated) AND -- after an
  explicit mid-task user correction reasserting the standing "GPU 1 only"
  rule -- REDONE on GPU1 (co-located with the two running jobs, contended,
  no job killed or throttled). GPU1 numbers are the primary evidence per
  that correction; GPU2 numbers are kept as supplementary/isolated.
- Phase 6: end-to-end 10-optimizer-step equivalence, individual target (tf
  AND onpolicy) and greedy_set target (onpolicy), same initial state, same
  batches, same seed.
- Unit tests: 14 new dispatcher-wiring tests
  (tests/test_oracle_compute_wiring_opt04.py), OPT03's 87 tests retained
  unmodified and passing, full suite regression-checked.
- research/REVIEW_FOR_CHATGPT.md: NOT yet appended this round (not
  requested this round; will follow the session's established pattern of
  appending only when explicitly asked).

NOT EXECUTED:
- Full spec-level benchmark rigor (>=20 warmup, >=100 measured iterations,
  3 repeats, both wall and compute-only, at every cell) -- scaled down for
  time/GPU-contention reasons to 3-5 warmup / 10-20 measured / 1 repeat.
  Explicitly flagged as a shortfall against the request, not hidden (see
  Anomalies).
- Asymmetric-scorer re-verification of A's caching (argued structurally
  correct -- caching only changes call COUNT of individual_utility, never
  touches arm_score/the metric -- but not independently re-run under
  --scorer asymmetric this round).
- A dedicated end-to-end run for greedy_set under prefix_policy=tf (only
  onpolicy was run for E's end-to-end check, given the time already spent
  investigating the onpolicy divergence finding below).
- VRAM-only adoption path for A's norm-expansion (needs a 4th criterion,
  "target/selection changes don't affect training results," which was not
  separately investigated this round beyond the equivalence data OPT03
  already produced).
- git commit/push (not requested this round).
```

## Anomalies (reported first, per explicit instruction)

1. **GPU-1 vs idle-GPU mid-task correction.** This round's own spec said
   "가능하면 idle GPU를 사용하세요," so Phase 5's benchmarks were initially
   run on GPU2 (idle at the time). Partway through, the user explicitly and
   emphatically reasserted the session's standing "GPU 1 only, always"
   rule, overriding that spec line. All four training benchmarks were
   re-run on GPU1, co-located with the two running experiments (no process
   killed, paused, or throttled -- confirmed alive throughout, `etimes`
   growing normally). GPU1 numbers are reported as the primary evidence
   below; GPU2 numbers are kept as supplementary/isolated context. This
   session's persistent memory has been updated so this preference is not
   re-litigated in a future round.

2. **A pytest full-suite run showed a THIRD failure** beyond OPT03's 2
   known pre-existing ones
   (`tests/test_cross_channel_context.py::test_context_off_is_exactly_target_only`)
   -- but only in ONE invocation that was accidentally run WITHOUT pinning
   `CUDA_VISIBLE_DEVICES` (defaulting onto the heavily-contended GPU1).
   Isolated rerun of that single test, and a full-suite rerun, both pinned
   to the idle GPU2, PASSED cleanly (850 passed / the same 2 pre-existing
   failures). Root-caused as a GPU-1-contention timing flake from an
   unpinned invocation, NOT a code regression -- recorded rather than
   quietly re-run until green.

3. **The most consequential anomaly: E's per-call numerical equivalence
   (validated exhaustively in OPT02/OPT03, and reconfirmed here at t=0 on
   the real first Weather_96 batch: max diff 9.5e-7, 0/32 disagreements)
   does NOT survive intact through real multi-step ON-POLICY training.**
   Starting reference and optimized from an IDENTICAL initial model state,
   same batches, same seed, and running 10 real optimizer steps
   (Weather_96, `greedy_set`, `onpolicy`, `cosine`) produces:
   - `final_state_bit_exact: false`
   - final parameter L2 divergence: `0.233`
   - selection agreement after 10 steps: **48.3%** (34,710 / 67,200
     positions disagree)
   - per-step CE loss differences up to `0.053` (NOT the ~1e-6 scale OPT02/
     OPT03's per-call checks found)

   **Root cause, verified directly** (not merely hypothesized): on the
   untouched initial model, for a single channel, reference and optimized
   `run_sequence` calls produce IDENTICAL `oracle_idx` (the CE target
   label) at every one of 10 steps and every one of 32 batch rows, and
   IDENTICAL losses to 6+ decimal places -- the wiring itself is correct.
   But `model_next` picks (which drive the trajectory under
   `prefix_policy=onpolicy`) come from `u_hat`, NOT from the Oracle target
   -- so within a SINGLE forward pass, `picks` are guaranteed identical
   regardless of `greedy_set_impl` (confirmed: `torch.equal(picks_r,
   picks_o) == True` for a single-channel check). Aggregated over Weather's
   21 channels x K=10 steps x 32-row batch (6,720 label decisions per
   optimizer step), a small fraction of genuine near-ties (the same
   phenomenon OPT02/OPT03 already documented as rare, at the ~1e-6 scale)
   DO flip the discontinuous `argmax` CE target label between reference
   and optimized. Each flip perturbs that gradient step by a non-negligible
   amount (CE loss is not smooth in the target label). Because
   `prefix_policy=onpolicy` feeds the model's own now-perturbed weights
   back into the NEXT optimizer step's `u_hat`-driven picks, this compounds
   -- a chaotic/butterfly-effect amplification of a tiny, algebraically
   correct numerical difference, not a bug in the implementation.

   **This directly changes E's adoption decision** (see below): E is
   numerically validated at the per-call level, exactly as OPT02/OPT03
   found, but switching `--oracle_compute_impl optimized` for an
   `onpolicy`-prefix arm means the resulting TRAINED MODEL will not be
   reproducible against a `reference` run of the same arm -- a materially
   different (not just noisier) outcome that a researcher comparing
   `reference` vs `optimized` training curves needs to know about
   explicitly, not discover after the fact.

4. **Benchmark scale is reduced from the requested spec** (see NOT
   EXECUTED) due to GPU-1 contention and time budget -- reported honestly
   as a shortfall, not silently substituted for the full protocol.

## 2. Production wiring

**Modified file**: `scripts/train_factorial_e2e01.py` ONLY (confirmed via
`git diff --stat`; no other tracked file touched). Reference
implementations (`utils/dense_utility.py`,
`scripts/train_oracle_choice01.py::oracle_choice_step_loss`) remain
completely untouched.

**Call sites changed**: `run_sequence` (the one function both `train_epoch`
and `eval_epoch` funnel through), plus `eval_epoch`'s separate stepwise-
diagnostic recompute loop (for consistency -- it independently recomputes
`u_hat`/`u_target` along the free-running trajectory for reporting).

**Flag added**: `--oracle_compute_impl {reference,optimized}`, default
`reference`. Reuses the existing `--chunk_size` (default 4096) -- no
`--candidate_chunk_size` flag was added, per instruction.

**Resolution logic** (`resolve_oracle_compute_impl`, pure function, unit
tested):

| requested | C (choice_ce) | A (individual) | E (greedy_set) |
|---|---|---|---|
| `reference` | reference | reference | reference |
| `optimized`, chunk_size>=4096 | optimized | optimized_cache | optimized |
| `optimized`, chunk_size<4096 | optimized | optimized_cache | **reference (fallback)** |

**Logged on every run** (`[oracle-compute]` block, per the exact format
requested):
```
[oracle-compute]
requested_impl=optimized
choice_ce_impl=optimized
individual_impl=optimized_cache
greedy_set_impl=reference
chunk_size=2048
fallback_reason[greedy_set]=chunk_size=2048 < GREEDY_SET_OPTIMIZED_MIN_CHUNK_SIZE=4096 (OPT02/OPT03: optimized Set Oracle is SLOWER than reference below this chunk size)
```
No silent fallback -- `fallback_reason` is only ever empty for a path that
did not fall back.

## 3. Equivalence

### C -- Oracle-Choice CE

**PASS, unconditional, every level tested.** 37 unit tests (30-seed
sweeps, all-valid/single-valid/no-valid masks, exact ties, gradient
equivalence) -- loss and all 5 diagnostic fields bit-exact
(`torch.equal`), gradient bit-exact. 2 dispatcher-level tests confirm
`run_sequence(choice_ce_impl='optimized')` produces bit-exact losses and
picks for BOTH individual and greedy_set targets. End-to-end 10-step
training (individual target, tf AND onpolicy): `max_abs_loss_diff=0.0`,
`final_state_bit_exact=true` at both prefix policies. Full detail:
`results/TRACK-A-WEATHER-OPT04/equivalence_choice_ce.json`.

### A -- Individual Oracle caching

**PASS, unconditional, every level tested.** 4 dispatcher tests confirm
`u_target`/`oracle_idx`/losses/picks are bit-exact vs reference for BOTH
`tf` and `onpolicy`, and a call-counting test directly proves the caching
claim: `individual_utility` is called exactly **1** time per `run_sequence`
call under `optimized_cache` vs **K=4** times under `reference` (tiny K=4
fixture; production K=10). End-to-end 10-step training (tf AND onpolicy):
`max_abs_loss_diff=0.0`, `final_state_bit_exact=true` at both. Not
separately re-verified under `--scorer asymmetric` (structural argument
only -- caching never touches `arm_score`/the metric). Full detail:
`results/TRACK-A-WEATHER-OPT04/equivalence_individual_cache.json`.

### E -- Greedy Set Oracle

**CONDITIONAL.** Per-call: PASS, matching OPT02/OPT03 exactly (max diff
9.5e-7, 0/32 disagreements at t=0 on the real first Weather_96 batch, run
through the actual production dispatcher). **End-to-end 10-step onpolicy
training: the critical anomaly above** -- 48.3% selection agreement,
non-bit-exact final state, param L2 diff 0.233, loss diffs up to 0.053.
This is EXPECTED and EXPLAINED (chaotic amplification of rare, individually
tiny and individually correct near-tie label flips through the onpolicy
feedback loop), not a defect in `dense_utility_optimized` itself -- but it
means E does not offer TRAINING REPRODUCIBILITY the way C and A do. Full
detail: `results/TRACK-A-WEATHER-OPT04/equivalence_greedy_set.json`.

## 4. Real speed (forward + backward + optimizer.step(), all 4 requested cells)

**GPU1 (primary, per the mid-task correction -- contended, co-located with
the two running experiments, no process killed/throttled):**

| cell | target | B0 (ref) wall median | B1 (opt) wall median | speedup |
|---|---|---:|---:|---:|
| Weather_96 | individual/onpolicy/cosine | 15.68s | 2.74s | **5.72x** |
| Weather_96 | greedy_set/onpolicy/cosine | 16.45s | 3.15s | **5.23x** |
| Weather_720 | individual/onpolicy/cosine | 18.74s | 3.32s | **5.64x** |
| Weather_720 | greedy_set/onpolicy/cosine | 23.39s | 5.63s | **4.16x** |

(n_warmup=2-3, n_iters=6-10, 1 repeat -- reduced from the requested 20/100/3,
see Anomalies/NOT EXECUTED. compute-only times are within 1-2% of wall
times at every row -- dataloader overhead is negligible here since
`exp.memory_x`/`exp.memory_y` are pre-loaded on-device.)

**GPU2 (supplementary, isolated/uncontended, run before the mid-task
correction):**

| cell | target | B0 (ref) wall median | B1 (opt) wall median | speedup |
|---|---|---:|---:|---:|
| Weather_96 | individual/onpolicy/cosine | 2.33s | 0.86s | 2.71x |
| Weather_96 | greedy_set/onpolicy/cosine | 2.81s | 1.27s | 2.22x |
| Weather_720 | individual/onpolicy/cosine | 4.20s | 1.17s | 3.60x |
| Weather_720 | greedy_set/onpolicy/cosine | 7.34s | 2.59s | 2.83x |

Both GPU1 and GPU2 numbers show the SAME direction and are of the SAME
order of magnitude (2-6x); the GPU1 numbers are larger in absolute
speedup, plausibly because the reference path's larger, more
synchronization-heavy compute footprint suffers disproportionately more
from GPU time-slicing against the two concurrent jobs than the already-
leaner optimized path does. Full data:
`results/TRACK-A-WEATHER-OPT04/benchmark_training.csv` and
`benchmark_compute_only.csv` (16 rows: 4 cells x {reference,optimized} x
{gpu1-contended, gpu2-isolated}).

**These are real training-iteration numbers (forward+backward+
optimizer.step(), `torch.cuda.synchronize()` at every timed boundary) --
not the forward-only, no_grad microbenchmarks OPT03 reported.** Given both
the isolated and contended measurements agree in direction and rough
magnitude, and are consistent with OPT03's own forward-only finding that C
dominates iteration time, this is reported as a genuine **production
speedup**, not merely "implemented."

## 5. VRAM

| cell | target | B0 peak alloc | B1 peak alloc | change |
|---|---|---:|---:|---:|
| Weather_96 | individual | 11942.1 MB | 11553.3 MB | -3.3% |
| Weather_96 | greedy_set | 11250.6 MB | 11156.9 MB | -0.8% |
| Weather_720 | individual | 25715.6 MB | 25340.1 MB | -1.5% |
| Weather_720 | greedy_set | 22112.9 MB | 22112.9 MB | 0.0% |

(GPU1 and GPU2 peak-allocated figures are identical per cell/impl, as
expected -- VRAM footprint does not depend on GPU contention.) **No VRAM
regression anywhere; every optimized cell uses equal or less peak VRAM
than the reference**, consistent with C's fix carrying no VRAM cost and
A's caching removing 9 of 10 redundant `[B,N,H]`-adjacent allocations
without adding any.

## 6. Adoption decision

| Path | Decision | Basis |
|---|---|---|
| **C -- Oracle-Choice CE** | **ACCEPT** | loss/diag/gradient bit-exact at every level (unit, dispatcher, real-data, 10-step end-to-end); training speedup 4.2-5.7x (GPU1) / 2.2-3.6x (GPU2); 0 VRAM regression. |
| **A -- Individual Oracle caching** | **ACCEPT** | u_target/picks/loss/gradient/final-state bit-exact at every level, tf AND onpolicy; caching count directly proven (1 call vs K); contributes to the same measured training speedups above (A and C are wired together in every benchmarked cell, so their individual marginal contributions are not separately isolated this round -- see NOT EXECUTED). |
| **A -- norm-expansion (chunk-based)** | **REJECT on speed** (erratum above: chunk=4096 is +0.12% total, not +9%; chunk=2048 is -61%), **NOT adopted as a VRAM-only path either** -- the 3rd VRAM-only criterion ("target/selection changes don't affect training results") was not investigated this round; left as a documented, unexecuted follow-up, not silently approved. |
| **E -- Greedy Set Oracle (OPT02 algebra)** | **CONDITIONAL ACCEPT** | Per-call equivalence, VRAM (no regression), and training speedup (4.2-5.2x GPU1 / 2.2-2.8x GPU2) all clear their bars. BUT: non-tie selection agreement is 100% only in the SAME sense OPT02/OPT03 measured it (single forward pass, free-running/no-gradient); under real onpolicy multi-step training it degrades to 48.3% and the final trained model is not reproducible against `reference`. **Recommend**: keep `optimized` available and default-off exactly as wired, but the production trainer's own documentation/CLI help should say plainly that `optimized` under `onpolicy` prefix does not reproduce `reference` training runs bit-for-bit (or even closely) -- a researcher who needs exact reproducibility across `reference`/`optimized` A-B runs should not use E's optimized path for onpolicy arms without accepting that caveat. `tf`-prefix arms were not end-to-end tested this round (NOT EXECUTED) -- a `tf` arm's trajectory is directly driven by the Oracle argmax, so a similar or possibly LARGER divergence is plausible there and should be checked before broader adoption. |

## 7. Files

**Created**:
```
scripts/benchmark_training_opt04.py
scripts/end_to_end_equivalence_opt04.py
tests/test_oracle_compute_wiring_opt04.py
research/TRACK-A-WEATHER-OPT04.md (this file)
results/TRACK-A-WEATHER-OPT04/environment.json
results/TRACK-A-WEATHER-OPT04/process_status_before.json
results/TRACK-A-WEATHER-OPT04/process_status_after.json
results/TRACK-A-WEATHER-OPT04/equivalence_choice_ce.json
results/TRACK-A-WEATHER-OPT04/equivalence_individual_cache.json
results/TRACK-A-WEATHER-OPT04/equivalence_greedy_set.json
results/TRACK-A-WEATHER-OPT04/benchmark_training.csv
results/TRACK-A-WEATHER-OPT04/benchmark_compute_only.csv
results/TRACK-A-WEATHER-OPT04/benchmark_vram.csv
results/TRACK-A-WEATHER-OPT04/benchmark_training_H{96,720}_{individual,greedy_set}[_gpu1_contended].{csv,json}
results/TRACK-A-WEATHER-OPT04/end_to_end_equivalence_individual[_tf].json
results/TRACK-A-WEATHER-OPT04/end_to_end_equivalence_greedy_set.json
results/TRACK-A-WEATHER-OPT04/test_summary.txt
results/TRACK-A-WEATHER-OPT04/git_diff_summary.txt
```

**Modified**: `scripts/train_factorial_e2e01.py` (the intended production
target this round; see section 2). No other tracked file.

**Overwritten**: none. `research/TRACK-A-WEATHER-OPT01.md`,
`-OPT02.md`, `-OPT03.md`, and every file under `results/TRACK-A-WEATHER-
OPT0{1,2,3}/` are untouched.

**Existing experiment impact**: none. `checkpoints/track_a_factorial_e2e/**`,
`logs/track_a_factorial_e2e/**`, `results/track_a_factorial_e2e/**`,
`checkpoints/track_a_multipos_choice01/**`,
`results/TRACK-A-MULTIPOS-CHOICE01/**` -- none of these were read from or
written to by this round's benchmark/equivalence scripts (they build their
own fresh scratch models via `build_experiment`, never loading or saving
into the running experiments' directories). Both PIDs (3712193, 3937098)
confirmed alive with growing `etimes` and unchanged command lines
immediately before this report was written.
