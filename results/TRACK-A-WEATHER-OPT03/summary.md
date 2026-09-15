# TRACK-A-WEATHER-OPT03 -- results summary

Full report: `research/TRACK-A-WEATHER-OPT03.md`.

## Scope

User-approved narrowing: only A (Individual Oracle), C (Oracle-Choice CE),
E (Greedy Set Oracle) -- these are the only three Oracle paths the running
Weather H96/H720 factorial (`scripts/train_factorial_e2e01.py`) actually
calls. B (Future-MSE teacher/KL) and D (Dense marginal utility) exist only
in unrelated scripts and were out of scope.

## Headline finding

Profiling on real Weather data found the dominant cost is **not** the
Oracle target computation the spec expected -- it is the shared
**Oracle-Choice CE loss (path C)**, at **85-89% of iteration time**, caused
by an unused diagnostic (`std_per_row`) computed via a per-batch-row Python
loop with a GPU->CPU sync on every row, every call. It is never read by
the returned `diag` dict -- pure dead code.

## Decisions

| Path | Optimization | Decision |
|---|---|---|
| C -- Oracle-Choice CE | Remove the dead `std_per_row` computation | **ACCEPT, unconditional.** 7.0x/6.7x path speedup (H96/H720), 4.7x/3.0x total iteration speedup, 0 VRAM change, loss+diag bit-exact, gradient-identical. |
| A -- Individual Oracle | Norm-expansion identity, chunked | **ACCEPT only at chunk_size>=4096** (+9% speed, -43% VRAM); REJECT at chunk=2048 (61% slower). Caching across K steps (not chunk-dependent) is a further, unconditionally-safe win, proposed for the production call site. |
| E -- Greedy Set Oracle | (OPT02's code, cited not rerun) | Carried over from OPT02: ACCEPT only at chunk_size>=~4096. |

## Correctness

- 87/87 new unit tests pass; full suite 836/838 (2 pre-existing failures,
  unchanged, no new regressions).
- Real Weather H96 (500 q) / H720 (300 q): A's few disagreements (5/5000,
  1/3000) are all reference-itself float32 ties among near-duplicate
  candidates, 0 real mismatches, 100% non-tie agreement. C's loss and every
  diagnostic field are bit-exact (0.0 diff) across all real trajectories
  checked.

## Not yet done

No production flag wired (`--oracle_compute_impl` design proposed only).
No `REVIEW_FOR_CHATGPT.md` append or git push performed unless separately
requested. Both concurrently running experiments (Weather H96 factorial,
MULTIPOS-CHOICE01 ETTh1_720) confirmed untouched throughout.
