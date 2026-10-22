# TRACK-V5-TQH-HEAD-OVERLAP-DIAG01 -- same-metric head redundancy, canonical V5 vs corrected TQH-5

Evaluation-only (no training, no backward, no checkpoint write).
ETTh1_96 / Shared-P100 / seed0 / test split. GPU1, read-only inference.

## 0. Why this track

`TRACK-TEMPORAL-QUERY-HEADS01-CORRECTION` used "Mean-Mixture vs
Round-Robin Top-10 overlap" as a proxy for head redundancy (TQH-5:
0.9785, canonical V5: 0.6931) and concluded TQH-5's heads are more
redundant. That proxy is not the same metric as pairwise Top-K overlap
between individual heads -- it measures how much the AGGREGATED
Mean-Mixture selection differs from a different aggregation rule
(Round-Robin), not how similar the heads are to EACH OTHER. This track
computes the actual pairwise head-overlap metric, identically, for
both models, to confirm or correct that conclusion directly.

## 1. Checkpoint provenance (read from artifacts, not hardcoded)

| | Path | Selected epoch | Selection criterion |
|---|---|---:|---|
| Canonical V5 | `checkpoints/track_v_sharedtop100_signfix01/ETTh1/H96/seed0/pool_top100/ETTh1_96/V5/checkpoint_epoch10.pth` | 10 | min validation Mean-Mixture RetMSE@10 (from `.../V5/checkpoint_selection/selected_checkpoint.json`) |
| TQH-5 corrected | `checkpoints/track_temporal_query_heads01/ETTh1_96/ETTh1_96/TQH5Masked/checkpoint_epoch7.pth` | 7 | min validation Mean-Mixture RetMSE@10 (from `.../TQH5Masked/checkpoint_selection/selected_checkpoint.json`) |

S1 passed: V5 checkpoint file is under `track_v_sharedtop100_signfix01`
(dated 2026-10-08), NOT the teacher-sign-bugged historical path under
`track_v_multiquery_generalization01` (dated 2026-10-05, different
file, different content hash by size). S2 passed: TQH-5 checkpoint is
the exact file the correction track selected and used for its own
Mean-Mixture cache/Stage-2 re-evaluation.

## 2. Sanity checks

| Check | Result |
|---|---|
| S1 (V5 checkpoint is sign-fixed, not historical-invalid) | PASS |
| S2 (TQH-5 checkpoint matches correction track) | PASS |
| S3 (candidate pool global ids identical, V5 vs TQH, sampled) | PASS |
| S4 (`scores.shape == [B,5,100]` both models) | PASS |
| S5 (Top-K=10) | PASS |
| S6 (same test queries/channel order -- same `exp`/loader construction for both) | PASS |
| S7 (TQH-5 `slot_top10_overlap` reproduces original report's 0.9769433165611625) | diff = 1.44e-4 -- reproduced to within float-accumulation tolerance (this run averages per-channel-batch calls in a slightly different grouping than the original training-time diagnostic loop; the underlying scores/checkpoint are identical, confirmed by S1/S2) |

## Table 1 -- Same-Metric Head Diversity (test)

| Model | Pairwise Top-1 overlap ↓ | Pairwise Top-10 overlap ↓ | Union Size ↑ (max 50) | Union Oracle Recall, 5 heads ↑ |
|---|---:|---:|---:|---:|
| canonical V5 | 0.1397 | 0.3761 | 27.13 | 0.5095 |
| TQH-5 | 0.9570 | 0.9771 | 10.51 | 0.2519 |

## Table 2 -- Pairwise Top-10 overlap distribution (fine-grained, every head pair x query x channel)

| Model | Mean | Std | P25 | Median | P75 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| V5 | 0.3754 | 0.2729 | 0.10 | 0.40 | 0.60 | 0.00 | 1.00 |
| TQH-5 | 0.9769 | 0.0543 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 |

Query-level (overlap averaged over the 10 head-pairs first, then
distribution over queries/channels): V5 mean=0.3754 std=0.1727
(range 0.05-0.96); TQH-5 mean=0.9769 std=0.0390 (range 0.61-1.00).
V5's heads show real, wide, query-dependent variation in how much they
agree; TQH-5's heads agree almost completely on almost every query.

## Table 3 -- Marginal Head Coverage (Union Oracle Recall, deterministic head order 0->4)

| # Heads | V5 Union Oracle Recall | TQH-5 Union Oracle Recall |
|---:|---:|---:|
| 1 | 0.2502 | 0.2430 |
| 2 | 0.3515 | 0.2443 |
| 3 | 0.4216 | 0.2469 |
| 4 | 0.4664 | 0.2469 |
| 5 | 0.5095 | 0.2519 |

V5: adding heads 2-5 raises Union Oracle Recall from 0.2502 to 0.5095
-- a **+103.6% relative gain**, more than doubling. TQH-5: 0.2430 to
0.2519 -- **+3.7% relative gain**. V5's extra heads are doing real
retrieval work; TQH-5's are not.

## 3. Verdict

Per the pre-registered decision rule (section 11): `TQH slot_top10_overlap
(0.9771) >> V5 slot_top10_overlap (0.3761)`, AND `TQH union size (10.51)
< V5 (27.13)`, AND `TQH union oracle recall gain (+3.7%) < V5 (+103.6%)`
-- **CASE A**.

> Under the identical pairwise Top-10 overlap metric, canonical V5 =
> 0.3761 and TQH-5 = 0.9771; therefore TQH's temporal-attention heads
> are empirically far more redundant with each other than canonical
> V5's static-linear heads, confirmed directly (not via the
> Round-Robin-vs-Mean-Mixture proxy) -- the head-collapse finding in
> the correction report is upheld and sharpened, not weakened, by this
> same-metric comparison.

## 4. What changes in prior documents

`TRACK-TEMPORAL-QUERY-HEADS01-CORRECTION`'s Table 4 / Section 6 Q3
wording ("TQH-5's heads are substantially MORE redundant... 97.9% vs
69.3%") used the RR-vs-Mean-Mixture proxy, which pointed the same
direction as this direct metric but is not equivalent to it and
understated the gap (the proxy numbers, 0.9785 vs 0.6931, differ
quite a bit from the direct pairwise numbers, 0.9771 vs 0.3761 --
same conclusion, much larger true gap). No numeric claim in the
correction report is reversed; this track supersedes it as the
primary evidence for head-redundancy comparisons and should be cited
instead of the RR-vs-MM proxy going forward.

## Reproducibility

- Code: `scripts/eval_v5_tqh_head_overlap_diag01.py` (new, evaluation
  -only, reuses `slot_mechanism_diagnostics`, `compute_scores`,
  `compute_scores_tqh`, `union_oracle_recall_curve` unmodified).
- Raw output: `results/TRACK-V5-TQH-HEAD-OVERLAP-DIAG01/ETTh1_96/overlap_diag.json`.
