# PHASE A -- Stage2 Correctness Audit

## A1. Weather H96 "Base MSE = 0.19527" -- ROOT CAUSE FOUND, REPORT ERROR CONFIRMED

**This was a report-writing error in `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s own analysis, NOT an evaluation
or training bug.** The true Base Forecaster test MSE for Weather H96 is **0.1693644558551501**
(`results/TRACK-R-FINAL-METHOD-GENERALIZATION01/Weather/H96/seed0/base/metrics.json`, confirmed exact via
independent recomputation below).

**Traced source of the wrong number**: `research/EXPERIMENT_LOG.md:4607`, a table titled
"Base Forecaster (B0) MSE | Base MAE | Raw Cosine (R0) MSE | Raw Cosine MAE":

```
| Weather | 96 | 0.16936 | **0.21181** | 0.19527 | 0.22632 |
```

`0.19527` is the **Raw Cosine (R0) baseline's MSE** -- a completely different, weaker baseline (plain
cosine-similarity retrieval with no learned components) -- not the Base Forecaster's MSE, which sits two
columns to the left in the same row (`0.16936`). When `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s Phase 2/4/Final
reports were written, this row was misread and the wrong column (`0.19527`) was used as "Base". This
propagated into `PHASE_2_SHARED_STAGE2_REPORT.md`, `PHASE_4_SEPARATE_STAGE2_REPORT.md`, and
`FINAL_RESEARCH_REPORT.md`'s Weather_96 tables and the claim "all 6 retrieval arms beat Base" -- see A1-impact
below.

**Verdict**: METRIC_INVALID for the specific "Weather_96 Base = 0.19527" value in the three files named
above; this does not affect any Stage1 RetMSE/AggMSE number (those never used this figure), and does not
affect any Stage2 MSE number itself (those were computed independently of the `Base` display column, as A4
below confirms) -- only the Base *comparison point* displayed in three result tables was wrong.

## A2. Base Checkpoint Verification

| Cell | Base checkpoint path | SHA256 | test_mse (metrics.json) | test_mse (independent recompute, A4) | Match |
|---|---|---|---:|---:|---|
| ETTh1_96 | `checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth` | (not re-hashed this round; same file used by all 6 arms, confirmed by A4's identical per-arm `base_mse` below) | 0.39242 (EXPERIMENT_LOG) | 0.392423 | YES |
| ETTh1_720 | `checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth` | " | 0.56035 | 0.560354 | YES |
| Weather_96 | `checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth` | `3ebb33761a0302ef756cce13470c3d3df8612f287e7f409ade4c3fa3f87aff55` | **0.16936** (corrected) | 0.169364 | YES |

All 6 arms within each cell load the identical base checkpoint file (same path passed to every
`train_r_stage2_lambda01.py` invocation for that cell; confirmed by the per-arm independent `base_mse`
being bit-identical across all 6 arms in each cell's verification JSON -- see A4).

(Weather_720's base checkpoint is confirmed identical between `track_r_final_method_generalization01` and
`track_v_multiquery_generalization01` paths -- both have SHA `43f36a584d812001eb81e4c855041394d90c5cd76cfa69c5f6c03855c6a7782d` for `final_sha256` -- per the prior session's check before launching Weather_720 training; Weather_720 Stage2 is not yet complete and will be independently verified by this same script once it finishes.)

## A3. Retrieval Cache Schema

Not re-audited from scratch this round (the cache builder `build_v5_head_loss_stage2_cache01.py` writes
`query_start_idx` + `relation_outputs` with the identical schema as the canonical
`build_v_meanmix_cache_pool01.py`, and A4 below independently confirms the cached `R` reproduces the
recorded Stage2 numbers through the *exact* `load_tensors` lookup-by-`query_start_idx` path -- a schema
mismatch would have made every one of the 18 arm x cell matches below fail, not silently pass). A dedicated
per-field schema dump is deferred to a later pass if Phase B/C turn up any inconsistency.

## A4. Independent Stage2 Recomputation -- ALL 18 (arm x cell) COMBINATIONS MATCH EXACTLY

Script: `scripts/verify_v5_head_loss_stage2_correctness01.py` (new, read-only, no training). For each
(cell, arm): rebuilds `B` (fresh `BaseForecastHead` forward pass + `X_last` offset) and `R` (from the cache,
via `train_r_stage2_lambda01.py`'s own unmodified `load_tensors`), reads the already-trained lambda from
the committed `metrics.json`, recomputes `fused = B + lambda*(R-B)`, and compares MSE/MAE to the recorded
values, plus verifies the `lambda=0 -> B` and `lambda=1 -> R` identities exactly (`torch.allclose`, atol=1e-6).

| Cell | Arm | Independent Base MSE (test) | Recorded Stage2 MSE | Independent Stage2 MSE | Match | lambda=0 identity | lambda=1 identity |
|---|---|---:|---:|---:|---|---|---|
| ETTh1_96 | S0 | 0.392423 | 0.369597 | 0.369597 | YES | YES | YES |
| ETTh1_96 | S1 | 0.392423 | 0.370652 | 0.370652 | YES | YES | YES |
| ETTh1_96 | S2 | 0.392423 | 0.371837 | 0.371837 | YES | YES | YES |
| ETTh1_96 | D0 | 0.392423 | 0.376534 | 0.376534 | YES | YES | YES |
| ETTh1_96 | D1 | 0.392423 | 0.373972 | 0.373972 | YES | YES | YES |
| ETTh1_96 | D2 | 0.392423 | 0.375010 | 0.375010 | YES | YES | YES |
| ETTh1_720 | S0 | 0.560354 | 0.514917 | 0.514917 | YES | YES | YES |
| ETTh1_720 | S1 | 0.560354 | 0.517572 | 0.517572 | YES | YES | YES |
| ETTh1_720 | S2 | 0.560354 | 0.514618 | 0.514618 | YES | YES | YES |
| ETTh1_720 | D0 | 0.560354 | 0.505956 | 0.505956 | YES | YES | YES |
| ETTh1_720 | D1 | 0.560354 | 0.512241 | 0.512241 | YES | YES | YES |
| ETTh1_720 | D2 | 0.560354 | 0.506632 | 0.506632 | YES | YES | YES |
| **Weather_96** | S0 | **0.169364** | 0.176649 | 0.176649 | YES | YES | YES |
| Weather_96 | S1 | 0.169364 | 0.179473 | 0.179473 | YES | YES | YES |
| Weather_96 | S2 | 0.169364 | **0.168375** | 0.168375 | YES | YES | YES |
| Weather_96 | D0 | 0.169364 | 0.181700 | 0.181700 | YES | YES | YES |
| Weather_96 | D1 | 0.169364 | 0.178282 | 0.178282 | YES | YES | YES |
| Weather_96 | D2 | 0.169364 | **0.168030** | 0.168030 | YES | YES | YES |

**18/18 match exactly (<1e-5 MSE/MAE tolerance), both identities hold exactly, for every arm in every
completed cell.** `IMPLEMENTATION_PASS` / `METRIC_VALID` for the Stage2 training and evaluation pipeline
itself. The only error found was the display-table column misread described in A1 -- the underlying
numbers were always computed correctly.

## A1-impact: corrected Weather_96 Base comparison

With the correct Base = **0.169364** (not 0.19527), the Weather_96 "Base vs retrieval arms" conclusion
**reverses for 4 of 6 arms**:

| Arm | Stage2 MSE | vs corrected Base (0.169364) |
|---|---:|---|
| S0 (L0, Shared) | 0.176649 | **WORSE** than Base (+4.3%) |
| S1 (L3a, Shared) | 0.179473 | **WORSE** than Base (+6.0%) |
| S2 (L3b, Shared) | 0.168375 | better than Base (-0.6%) |
| D0 (L0, Separate) | 0.181700 | **WORSE** than Base (+7.3%) |
| D1 (L3a, Separate) | 0.178282 | **WORSE** than Base (+5.2%) |
| D2 (L3b, Separate) | 0.168030 | better than Base (-0.8%) |

**The correction makes the L3b finding stronger, not weaker**: it is no longer "retrieval generically helps
on Weather_96, L3b helps a bit more" -- it is now "plain-L0/L3a retrieval (S0/S1/D0/D1) all LOSE to Base on
this cell, and only the Top-K-aligned aggregate loss (S2/D2) manages to even edge out Base, by a very thin
margin (<1%)." This is consistent with, not a reversal of, this project's long-standing repeated finding
that Weather is hard for Base to beat -- Weather_96 is not an exception after all; L3b is just the one loss
variant that gets close.

`TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s `PHASE_2_SHARED_STAGE2_REPORT.md`, `PHASE_4_SEPARATE_STAGE2_REPORT.md`,
and `FINAL_RESEARCH_REPORT.md` have been given a correction notice (not silently edited/deleted) pointing
here; see each file's new "CORRECTION" block.
