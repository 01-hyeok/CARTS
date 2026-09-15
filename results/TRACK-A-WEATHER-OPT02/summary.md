# TRACK-A-WEATHER-OPT02 -- results summary

Full report: `research/TRACK-A-WEATHER-OPT02.md`. Raw data: `benchmark_H{96,720}[_supplementary_4096].csv`, `equivalence_H{96,720}.json`.

## Headline

| Goal | Status |
|---|---|
| Fix OPT01's +58-59% peak-VRAM regression | **FIXED** -- B2 within +0.04% to +0.3% of reference at every chunk size, both horizons |
| Preserve OPT01's speedup | **PARTIALLY** -- met only at chunk=4096 (outside the requested 128-2048 sweep): 1.41x H96 / 2.95x H720. Within the requested sweep, best case (chunk=2048) is 1.17x H96 / 2.49x H720, below the 1.4x/3.0x targets. Small chunks (128-512) are SLOWER than the reference at both horizons. |
| Correctness (equivalence) | **PASSED** -- 100% non-tie selection agreement both horizons; the one H96 disagreement is a bit-exact reference tie already documented in OPT01, aggregate-MSE impact 3.9e-7 |
| Test suite | **PASSED** -- 39/39 new+adapted tests, 749/751 full suite (2 pre-existing failures, no new regressions) |

## Bottom line

The VRAM problem is solved. The speed problem is only solved if the production `candidate_chunk_size` is set large (~4096, not the 128-2048 range this round was asked to sweep) -- this is reported plainly, not spun as a full pass. See `research/TRACK-A-WEATHER-OPT02.md` for the Amdahl-consistency caveat, the full per-chunk-size table, and the proposed (not-yet-applied) minimal production patch plan.

No production trainer was modified this round. Both concurrently running experiments (`TRACK-A-FACTORIAL-E2E01` Weather cells, `TRACK-A-MULTIPOS-CHOICE01`) were left untouched throughout and are confirmed still running as of report completion.
