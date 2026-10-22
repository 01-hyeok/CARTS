# TRACK-TEMPORAL-QUERY-HEADS01-CORRECTION -- ETTh1_96, corrected baseline + Mean-Mixture evaluation

Supersedes the V1/V5 comparison in `REPORT_ETTh1_96.md` (now marked
SUPERSEDED at the top of that file; not deleted). Read-only
re-evaluation -- **no TQH retraining was done**; all 10 epoch
checkpoints for TQH-1 and TQH-5 already existed and were audited as
complete before reuse (Step C).

## 0. What was wrong

`REPORT_ETTh1_96.md` compared TQH-1/TQH-5 against V1/V5 numbers pulled
from `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/`.
That exact subtree carries
`INVALIDATED_BY_TEACHER_SIGN_BUG.md` in each arm's directory (V0/V1/V2/V5
all present) -- missed when the original report was written. The bug:
`scripts/train_v_sharedtop100_01.py` called
`normalized_teacher_prob(-d_pool, ...)` at its training-loss and
diagnostic sites. `d_pool` is already a correctly-signed, lower-is
-better MSE distance, and `normalized_teacher_prob(d, ...)` already
internally z-scores and negates (`softmax(-zscore(d)/tau)`). Passing
`-d_pool` double-negates: the teacher ends up favoring the
HIGHEST-MSE candidates. Reproduced directly in this session:

```
d = [0.1, 1.0, 10.0]
normalized_teacher_prob(d, ...).argmax()   -> 0  (correct: lowest MSE)
normalized_teacher_prob(-d, ...).argmax()  -> 2  (bug: highest MSE)
```

This bug affected every P100/Shared-Top-100 Stage-1 cell trained via
that script -- including the V1/V5 checkpoints the original TQH report
used as baseline. Those V1/V5 numbers (RetMSE@10=1.62/1.62, Stage2 MSE
0.394/0.394) are real outputs of a retriever trained against an
inverted teacher, not a fair baseline.

Separately, V5's canonical evaluation protocol uses **Mean-Mixture**
Top-K selection (`p_mix = mean_h softmax(scores_h/tau_s)`, then Top-K
on `p_mix`), not Round-Robin -- established in
`TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01` after finding the two give
materially different numbers for multi-head arms. `TQH-5`'s own cache
builder (`build_tqh_retrieval_cache01.py`) used Round-Robin, so even
setting the baseline-provenance bug aside, TQH-5 was not evaluated
under the same inference rule as canonical V5.

## 1. Step A -- canonical baseline provenance (ETTh1_96, seed0)

| Field | Value |
|---|---|
| Stage-1 trainer | `scripts/train_v_sharedtop100_01.py`, teacher-sign FIXED (`normalized_teacher_prob(-d_pool,...)` -> `normalized_teacher_prob(d_pool,...)`) |
| Stage-1 checkpoints | `checkpoints/track_v_sharedtop100_signfix01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/checkpoint_epoch{N}.pth`, fixed 10 epochs, no early stopping |
| Candidate pool | `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/shared_candidate_pool` (bug-independent -- `precompute_candidate_pool01.py` never calls `normalized_teacher_prob`; SAME file TQH itself was trained/evaluated against) |
| Stage-1 checkpoint-selection criterion | V1: single-distribution val RetMSE@10 (== Mean-Mixture for S<=1, proven); selected epoch 9. V5: post-hoc Mean-Mixture val RetMSE@10 across all 10 epochs (`eval_v5_meanmix_checkpoint_selection_pool01.py`), test split never touched; selected epoch 10. |
| Stage-1 inference rule | Mean-Mixture Top-K (`utils/mean_mixture_selection.py::mean_mixture_topk_selection`) for V5; ordinary Top-K for V1 (S=1) |
| Stage-2 mode | "Original CARTS Trainable Global Lambda", `Y = B + sigmoid(a)*(R-B)`, via `scripts/eval_r_stage2_lambda_signfix01.py` (reuses `train_r_stage2_lambda01.py`'s formula unmodified) on the sign-fixed, Mean-Mixture cache |
| tau_s / tau_t | 0.1 / 0.1 |
| seq_len / pred_len | 96 / 96 |
| seed | 0 |
| Base Forecaster | `checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth` -- SAME file used by the (invalid) original TQH report and by this correction |
| Result files | `results/TRACK-V-SHARED-TOP100-SIGNFIX01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/{cache/stage1_metrics.json, stage2_lambda_original/metrics.json}` |

Canonical V1/V5, ETTh1_96, seed0, test split:

| Arm | RetMSE@10 | AggMSE@10 | Recall@10 | NDCG@10 | Stage2 MSE | Stage2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 0.66255 | 0.41271 | 0.25857 | 0.87447 | 0.37019 | 0.39521 |
| V5 | 0.66386 | 0.40981 | 0.25752 | 0.87411 | 0.36960 | 0.39547 |

## 2. Step B -- TQH's own teacher-sign check

`scripts/train_temporal_query_heads01.py` calls
`normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)` at both sites
(lines ~304, ~309) -- the CORRECT call, never `-d_pool`. Confirmed by
direct grep and by re-running the same synthetic sanity check used for
the V5 bug report:

```
normalized_teacher_prob([0.1,1.0,10.0]).argmax()  -> 0  (correct, matches TQH's own call convention)
```

**TQH's own training was never affected by the teacher-sign bug.**

## 3. Step C -- TQH checkpoint completeness audit

Both `checkpoints/track_temporal_query_heads01/ETTh1_96/ETTh1_96/{TQH1Masked,TQH5Masked}/`
contain `checkpoint_epoch1.pth` through `checkpoint_epoch10.pth`,
`config.json`, `batch_order_hashes.csv`, `checkpoint_fingerprints.json`
-- complete. `init_hash` in both configs
(`0f044e3ce726fc5c16bb839e0651b1304349eab03896ec1e7b971f49934be7b1`)
matches the canonical sign-fixed V5 run's own recorded `init_hash`
byte-for-byte. **No retraining was needed or done.**

## 4. Step E -- Mean-Mixture checkpoint re-selection

New script `scripts/eval_tqh_meanmix_checkpoint_selection01.py`
(sibling of the V5 canonical script, swaps in `compute_scores_tqh`),
run read-only over all 10 saved epochs, validation split only:

| Arm | Old RR best epoch | New Mean-Mixture best epoch | Changed? |
|---|---:|---:|---|
| TQH-1 | 7 | 7 | No (RR==Mean-Mixture at S=1, proven; epoch-8 values differ only at the 6th decimal -- float noise) |
| TQH-5 | 7 | 7 | No |

TQH-5's selected epoch did not change, but the per-epoch RetMSE@10
values under Mean-Mixture vs Round-Robin differ only in the 3rd-4th
decimal at every epoch (e.g. epoch 7: 0.992074 RR vs 0.991612 Mean) --
a MUCH smaller RR-vs-Mean-Mixture gap than canonical V5 shows on this
same cell (see Table 3 below) -- itself informative, see Section 6.

## 5. Step F/G -- corrected retrieval cache + Stage-2 (Mean-Mixture, TQH-5; TQH-1 unchanged since RR==Mean-Mixture)

New scripts `scripts/build_tqh_meanmix_cache01.py` (TQH-5, using
`mean_mixture_topk_selection`) and reuse of the EXISTING
`build_tqh_retrieval_cache01.py` cache for TQH-1 (mathematically
identical to a Mean-Mixture cache at S=1). Both caches built against
the SAME Shared-P100 candidate-pool file as canonical V1/V5 (pool
identity guaranteed by file reuse, not re-derived). Stage-2: reused
`train_r_stage2_lambda01.py` unmodified, fed the new caches, same Base
Forecaster checkpoint as canonical V1/V5.

## Table 1 -- Forecast (Stage-2 Test MSE/MAE)

| Arm | Provenance | Stage2 MSE | MAE | Delta vs canonical V5 |
|---|---|---:|---:|---:|
| V1 sign-fixed | canonical | 0.37019 | 0.39521 | +0.16% |
| V5 sign-fixed | canonical | 0.36960 | 0.39547 | -- |
| TQH-1 corrected | this track | 0.37176 | 0.39553 | **+0.58%** |
| TQH-5 corrected (Mean-Mixture) | this track | 0.37176 | 0.39553 | **+0.58%** |

## Table 2 -- Retrieval (test, restricted-P100 oracle)

| Arm | RetMSE@10 | AggMSE@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| V1 sign-fixed | 0.66255 | 0.41271 | 0.25857 | 0.87447 |
| V5 sign-fixed | 0.66386 | 0.40981 | 0.25752 | 0.87411 |
| TQH-1 | 0.67616 | 0.41583 | 0.24065 | 0.86866 |
| TQH-5 (Mean-Mixture) | 0.67634 | 0.41585 | 0.24039 | 0.86858 |

TQH-1/TQH-5 are now slightly WORSE than canonical V1/V5 on every
single metric in this table (RetMSE@10 +1.9% vs V5, Recall@10 -6.7pp,
NDCG@10 -0.6pp) -- the complete reverse of what the (invalid) original
report showed.

## Table 3 -- TQH-5 Round-Robin vs Mean-Mixture (correction delta)

| Metric | Old Round-Robin | Corrected Mean-Mixture | Delta |
|---|---:|---:|---:|
| selected epoch | 7 | 7 | unchanged |
| RetMSE@10 (test) | 0.676527 | 0.676336 | -0.03% |
| AggMSE@10 (test) | 0.415806 | 0.415849 | +0.01% |
| Recall@10 (test) | 0.24024 | 0.24039 | +0.06% |
| Stage2 MSE (test) | 0.371730 | 0.371760 | +0.01% |
| Top-10 overlap, Mean-Mixture vs Round-Robin | -- | **0.9785** | (canonical V5's own overlap: 0.6931, see Section 6) |

The Round-Robin-vs-Mean-Mixture correction itself changes essentially
nothing for TQH-5 -- all deltas are inside noise. This is NOT because
the inference-rule choice doesn't matter in general (canonical V5
shows a much larger RR/Mean-Mixture gap on the same cell); it is
because TQH-5's 5 heads are so close to identical that which selection
rule is applied over them barely matters.

## Table 4 -- Head Collapse (TQH-5 corrected, test)

| Metric | TQH-5 corrected | For comparison: canonical V5's own heads |
|---|---:|---:|
| Top-10 overlap, Mean-Mixture vs Round-Robin | 0.9785 | 0.6931 |
| Pairwise attention cosine (from the earlier uncorrected run, unaffected by this correction) | 0.9988 | n/a (V5 has no attention mechanism to measure) |
| Union Oracle Recall, 1 head -> 5 heads | 0.2430 -> 0.2519 | n/a |

**TQH-5's heads are substantially MORE redundant with each other than
canonical V5's own static-linear heads are** (97.9% RR/Mean-Mixture
overlap vs V5's 69.3%). The architecture intended to produce more
complementary retrieval heads in fact produces less complementary ones
than the baseline it was meant to improve on.

## 6. Answers to the four required questions

**Q1. Does TQH-1/TQH-5, re-evaluated under the canonical protocol,
still beat sign-fixed V1/V5?**
No. TQH-1 and TQH-5 both score worse than BOTH canonical V1 (0.37019)
and V5 (0.36960) on Stage-2 test MSE (0.37176, +0.42% vs V1, +0.58% vs
V5), and worse on every Stage-1 retrieval metric (Table 2). The
original report's "TQH beats V5 by 5.7%" was entirely an artifact of
comparing against the teacher-sign-bugged V1/V5 baseline, whose own
retrieval was corrupted (bugged V5 RetMSE@10=1.6219 vs its own
corrected value 0.6639 -- the bug alone cost V5 roughly 2.4x in
RetMSE@10, which is why TQH looked dramatically better than it than it
actually is relative to a healthy V5).

**Q2. How much does the Mean-Mixture correction change TQH-5's own
numbers?**
Negligibly (Table 3, all deltas <0.1%). The reason is itself a
finding: TQH-5's heads are so collapsed (Table 4) that Round-Robin and
Mean-Mixture select almost the same Top-10 set regardless.

**Q3. Is the head-collapse diagnosis the same after correction?**
Yes, and the correction adds a sharper, directly V5-comparable number:
TQH-5's heads overlap with each other (via the RR-vs-Mean-Mixture
proxy) at 0.9785, versus 0.6931 for canonical V5's own heads on the
identical cell. Collapse is not just present, it is worse than the
baseline.

**Q4. Does the original `CASE A: TQH-1 approx= TQH-5 > V5` verdict
survive?**
**No -- it must be and is corrected.** Under the canonical baseline
and canonical inference rule, the result is the NEW CASE A defined for
this correction track:

> TQH-1 approx= TQH-5 < canonical V1/V5

## 7. Verdict: CASE A (corrected definition)

> Temporal Query construction shows NO improvement over canonical V1/V5
> on this cell; multi-head contribution is also absent (TQH-1 approx=
> TQH-5, and TQH-5's heads are more collapsed than V5's own). Per the
> pre-registered decision rule: **TQH direction STOP** for this
> cell/seed, pending the user's/ChatGPT PI's read on whether any
> further investigation of this architecture is warranted at all.

This is a straightforward negative result, reported as such. The
large positive numbers in the original `REPORT_ETTh1_96.md` were a
baseline-provenance error, not a finding about TQH.

## 8. What this does NOT establish

- One cell, one seed. No claim is made about other horizons/datasets --
  none have been run for TQH, and per standing instruction none will
  be until this correction is reviewed.
- This correction does not explain why TQH-1/TQH-5 underperform
  canonical V1/V5 by this specific margin (~0.6%) -- only that they do.
- The random-masking control, patch-length sweep, and any architecture
  change remain explicitly out of scope until the user decides whether
  this result changes the research direction.

## Reproducibility

- New code (read-only re-evaluation, nothing retrained):
  `scripts/eval_tqh_meanmix_checkpoint_selection01.py`,
  `scripts/build_tqh_meanmix_cache01.py`.
- Raw artifacts:
  `results/TRACK-TEMPORAL-QUERY-HEADS01-CORRECTION/ETTh1_96/{TQH1Masked,TQH5Masked}/`.
- Canonical baseline reused unmodified from
  `results/TRACK-V-SHARED-TOP100-SIGNFIX01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/`.
- Bug documentation reused unmodified from
  `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/ETTh1_96/V5/INVALIDATED_BY_TEACHER_SIGN_BUG.md`.
