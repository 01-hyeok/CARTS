# TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 -- Final Report

4 cells (ETTh1 H96/H720, Weather H96/H720) x 4 arms (V1/V5/PH1/PH5),
seed0, Shared-P100, sequential, GPU1. All 16 Stage-1 trainings (fixed
10 epochs, no early stopping) + post-hoc checkpoint selection + cache
build + Stage-2 completed without retraining failures (one
path-construction bug in the orchestrator was caught and fixed after
V1/ETTh1_96's training had already finished -- no retraining was lost,
see `research/EXPERIMENT_LOG.md`). MLP-V1/V5 are read-only reused from
`TRACK-V-SHARED-TOP100-SIGNFIX01`, never retrained. See `AUDIT.md` for
the pre-registered implementation audit and smoke-test results.

## Table 1 -- Final Forecast (Stage-2 Test MSE)

| Dataset | H | MLP-V1 | MLP-V5 | T-V1 | T-V5 | T-PH1 | T-PH5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.37019 | 0.36960 | 0.37295 | 0.37141 | 0.37180 | 0.37364 |
| ETTh1 | 720 | 0.51822 | 0.51492 | 0.50920 | **0.50701** | 0.51443 | 0.51309 |
| Weather | 96 | 0.17750 | 0.17665 | 0.17543 | 0.17624 | 0.17439 | **0.17410** |
| Weather | 720 | **0.37760** | 0.38553 | 0.37865 | 0.37995 | 0.37338 | 0.38847 |

Bold = best arm for that cell. No single arm wins on every cell;
canonical MLP-V1 is in fact the best arm on Weather_720.

## Table 2 -- Retrieval Quality (Stage-1 test, restricted-P100 oracle)

| Dataset | H | Model | RetMSE@10 | AggMSE@10 | Recall@10 | NDCG@10 |
|---|---:|---|---:|---:|---:|---:|
| ETTh1 | 96 | V1 | 0.6722 | 0.4189 | 0.2474 | 0.8717 |
| ETTh1 | 96 | V5 | 0.6846 | 0.4181 | 0.2414 | 0.8674 |
| ETTh1 | 96 | PH1 | 0.6879 | 0.4253 | 0.2362 | 0.8647 |
| ETTh1 | 96 | PH5 | 0.6756 | 0.4262 | 0.2458 | 0.8714 |
| ETTh1 | 720 | V1 | 0.9050 | 0.6003 | 0.3181 | 0.8999 |
| ETTh1 | 720 | V5 | 0.9220 | 0.5994 | 0.3151 | 0.8878 |
| ETTh1 | 720 | PH1 | 0.9445 | 0.6287 | 0.3164 | 0.8933 |
| ETTh1 | 720 | PH5 | 0.9400 | 0.6254 | 0.3173 | 0.8940 |
| Weather | 96 | V1 | 0.4480 | 0.2602 | 0.2634 | 0.8287 |
| Weather | 96 | V5 | 0.4430 | 0.2590 | 0.2690 | 0.8304 |
| Weather | 96 | PH1 | 0.3935 | 0.2352 | 0.2652 | 0.8319 |
| Weather | 96 | PH5 | 0.3953 | 0.2347 | 0.2657 | 0.8318 |
| Weather | 720 | V1 | 0.7293 | 0.5813 | 0.2765 | 0.8419 |
| Weather | 720 | V5 | 0.7391 | 0.5902 | 0.2723 | 0.8414 |
| Weather | 720 | PH1 | 0.7129 | 0.5637 | 0.2657 | 0.8379 |
| Weather | 720 | PH5 | 0.7853 | 0.6242 | 0.2654 | 0.8331 |

## Table 3 -- Head Complementarity (V5/PH5 only; V1/PH1 have 1 head, no pairs)

| Dataset | H | Model | Pairwise Top-10 overlap | Union size (/50) | Oracle Recall 1-head | Oracle Recall 5-head | 5-head gain |
|---|---:|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | V5 | 0.3454 | 28.10 | 0.2227 | 0.4919 | +120.9% |
| ETTh1 | 96 | PH5 | **1.0000** | **10.00** | 0.2481 | 0.2482 | **+0.04%** |
| ETTh1 | 720 | V5 | 0.5337 | 22.12 | 0.3058 | 0.5299 | +73.3% |
| ETTh1 | 720 | PH5 | **1.0000** | **10.00** | 0.3167 | 0.3167 | **+0.0%** |
| Weather | 96 | V5 | 0.2864 | 32.37 | 0.2444 | 0.5265 | +115.4% |
| Weather | 96 | PH5 | **0.9999** | **10.00** | 0.2657 | 0.2657 | **+0.0%** |
| Weather | 720 | V5 | 0.4447 | 24.88 | 0.2334 | 0.4939 | +111.6% |
| Weather | 720 | PH5 | **0.9960** | **10.09** | 0.2653 | 0.2671 | **+0.68%** |

T-PH5's heads are essentially one head duplicated five times, in
EVERY cell -- far more collapsed than canonical V5's own (and T-V5's
own, which was not separately tabled here but follows the same pattern
as canonical V5, consistent with Table 2's AggMSE numbers) static
-linear heads, which show substantial, consistent complementary value
(+73% to +121% Union Oracle Recall gain from 1 to 5 heads).

## Table 4 -- Patch-Head Attention Diagnostics (PH1/PH5)

| Dataset | H | Model | Attn entropy (norm.) | Head attn cosine | Query-to-query diversity |
|---|---:|---|---:|---:|---:|
| ETTh1 | 96 | PH5 | 0.9982 | 1.0000 | 0.9946 |
| ETTh1 | 720 | PH5 | 0.9985 | 1.0000 | 0.9959 |
| Weather | 96 | PH5 | 0.9994 | 1.0000 | 0.9986 |
| Weather | 720 | PH5 | 0.9979 | 0.9997 | 0.9978 |

Attention is close to uniform over the 6 patches (entropy ~0.998 of
the max possible) AND nearly identical across different heads (cosine
~1.0) AND nearly identical across different queries (diversity
~0.995-0.999) in every single cell -- the mechanism is not
differentiating by head, nor by query, despite having real patch
-token diversity available to it (patch tokens themselves are NOT
degenerate -- see the predecessor track's diagnostics on real patch
content; this is a property of how the attention is used, not of the
tokens).

## Comparison-by-comparison read

**Comparison 1 (backbone effect, MLP-V5 vs T-V5)**: Transformer backbone
beats MLP in 3/4 cells (ETTh1_720 -1.54%, Weather_96 -0.23%,
Weather_720 -1.45%), loses in 1/4 (ETTh1_96 +0.49%). A real but
modest and not fully consistent effect.

**Comparison 2 (patch-token utilization, T-V5 vs T-PH5)**: T-PH5 is
WORSE than T-V5 in 3/4 cells (ETTh1_96 +0.60%, ETTh1_720 +1.20%,
Weather_720 +2.24%), better in 1/4 (Weather_96 -1.21%). Patch-aware
multi-head construction does not reliably beat pooled-only multi-head
under the same backbone.

**Comparison 3 (multi-head effect, T-PH1 vs T-PH5)**: Mixed and mostly
inside noise (ETTh1_96 +0.49%, ETTh1_720 -0.26%, Weather_96 -0.17%),
except Weather_720 where PH5 is clearly worse (+4.04%). No cell shows
a convincing multi-head gain; Table 3 shows why -- the 5 heads are
not behaving differently from each other.

**Comparison 4 (patch-aware effect without multi-head, T-V1 vs T-PH1)**:
T-PH1 beats T-V1 in 3/4 cells (ETTh1_96 -0.31%, Weather_96 -0.59%,
Weather_720 -1.39%), loses in 1/4 (ETTh1_720 +1.03%). The most
consistently positive comparison in this track, but NOT the one the
pre-registered GO/STOP gate (Section 25 of the spec) is keyed to.

## Pre-registered expansion gate (spec Section 25) -- checked directly

| Condition | Required | Actual | Pass? |
|---|---|---|---|
| T-PH5 beats T-V5 on AggMSE in >=3/4 cells | >=3/4 | **1/4** (Weather_96 only) | **FAIL** |
| Stage-2 MSE also improves or clear positive trend, >=3/4 cells | >=3/4 | 1/4 (Weather_96 only, -1.2%; ETTh1_720 PH5 beats MLP-V5 but not T-V5) | **FAIL** |
| T-PH5 beats T-PH1 (additional multi-head gain) | consistent | mixed, 2/4 trivially (<0.3%), 1/4 clearly worse | **FAIL** |
| Head overlap decreases AND Union Oracle Recall increases | both | neither moves meaningfully in any cell (Table 3) | **FAIL** |

**All four gate conditions fail.** Per the pre-registered rule (spec
Section 25): do NOT expand to seed1/2, H192/H336, full-candidate, or
any temperature/diversity-loss tuning.

## Verdict

Closest to **CASE 5** (`T-PH1 ≈ T-PH5, overlap ≈ 1` -> "실제
Transformer patch tokens를 사용해도 multi-head specialization이 발생하지
않음") for the head-collapse finding, which is the cleanest and most
uniform result in this track: in all 4 cells, independent of dataset
or horizon, PH5's 5 heads collapse to pairwise Top-10 overlap
0.996-1.000 and contribute 0.0-0.7% additional Union Oracle Recall --
dramatically less complementary than canonical V5's own static-linear
heads (+73% to +121% in the same metric, same cells). This is a
STRONGER, more uniform collapse than the predecessor
masked-MLP-token track found (which itself already showed severe
collapse).

The backbone effect (Comparison 1) and the single-head patch-aware
effect (Comparison 4) are each real in 3/4 cells but NOT part of what
this track's GO gate was conditioned on, and neither is large or
uniform enough on its own to recommend as a standalone next
experiment without further scoping -- reported honestly as secondary,
suggestive observations, not a second finding to act on.

**Verdict: Head-construction (multi-head over patch tokens) direction
STOPS here.** No seed1/2, no H192/H336, no full-candidate, no
temperature or architecture tuning follows from this track per the
pre-registered rule. Whether the backbone-only or patch-aware-single
-head observations are worth a dedicated, separately-scoped follow-up
is a decision for the user / ChatGPT PI review, not resolved here.

## Known deviation and its impact

Section 9's "shared Transformer init per cell across all 4 arms" was
verified via GPU smoke test (per-cell `init_hash` identical across
V1/V5/PH1/PH5, see `AUDIT.md`) and holds for the actual grid runs too
(each arm's own `config.json`, e.g. `ETTh1_96/V1/config.json`'s
`init_hash` matches `ETTh1_96/V5/config.json`'s, etc. -- same
`build_transformer_model` call, same `init_seed`, deterministic
construction). T-V5 and T-PH5 have byte-identical trainable parameter
counts in every cell (PH-arms add zero parameters, asserted at
startup in every run, not just the smoke test).

## Reproducibility

- Code: `scripts/train_transformer_patch_query_heads01.py`,
  `scripts/eval_transformer_patch_query_meanmix_selection01.py`,
  `scripts/build_transformer_patch_query_cache01.py`,
  `scripts/run_transformer_patch_query_heads01.sh`,
  `tests/test_transformer_patch_query_heads01.py`.
- Raw artifacts: `results/TRACK-TRANSFORMER-PATCH-QUERY-HEADS01/<cell>/<arm>/`.
- MLP-V1/V5 baselines reused unmodified from
  `results/TRACK-V-SHARED-TOP100-SIGNFIX01/<ds>/H<h>/seed0/pool_top100/<cell>/{V1,V5}/`.
