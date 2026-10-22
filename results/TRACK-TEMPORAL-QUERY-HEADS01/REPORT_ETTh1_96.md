> ## SUPERSEDED / INVALID BASELINE COMPARISON (2026-10-22)
>
> The V1/V5 numbers below were pulled from
> `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/`,
> a subtree each arm's own directory marks with
> `INVALIDATED_BY_TEACHER_SIGN_BUG.md` -- missed when this report was
> written. That bug (`normalized_teacher_prob(-d_pool,...)` double
> -negating an already-correctly-signed distance) corrupted those V1/V5
> checkpoints' own retrieval quality, making TQH look far better than
> it actually is by comparison. TQH-5 was also evaluated with
> Round-Robin Top-K instead of V5's canonical Mean-Mixture rule.
>
> **The CASE A verdict below ("TQH-1 approx= TQH-5 > V5") does NOT
> survive correction.** Under the canonical sign-fixed V1/V5 baseline
> and Mean-Mixture inference rule, TQH-1/TQH-5 are instead slightly
> WORSE than V1/V5 on every metric, and TQH-5's heads are MORE
> collapsed relative to Round-Robin than V5's own heads are. See
> `REPORT_CORRECTED_ETTh1_96.md` for the corrected analysis and the
> new verdict. This file is kept for audit trail, not deleted.

# TRACK-TEMPORAL-QUERY-HEADS01 -- ETTh1_96 single-cell diagnostic run

Status: ONE cell (ETTh1_96), ONE seed (seed0), FULL training/checkpoint
-selection protocol (not a short pilot -- `train_epochs<=10`, `patience=5`,
early stopping enabled, same as V1/V5's own runs). This is a diagnostic
cell meant to answer the CASE A-D question before committing to the
full ETTh1/Weather x H96/192/336/720 grid -- **not yet a generalization
claim.**

## 0. Why this cell only, so far

The first smoke test showed the proposed 5-head mechanism's heads were
indistinguishable from each other at step 0 even with `/sqrt(D)`
removed from the attention logits (user-approved fix, see commit
history). A 3-epoch forced pilot on this cell showed retrieval quality
improving sharply over V5 while heads stayed collapsed -- sharply
enough, and consistently enough across epochs, that a dedicated
TQH-1-vs-TQH-5-vs-V1-vs-V5 comparison on ONE cell, under the FULL
training protocol, was run before touching any other cell, per
explicit user instruction.

## 1. What changed vs V5 (recap)

Only the query-head construction:
- V5: `q_m = normalize(W_m z_q)` -- 5 independent static linear
  projections of the pooled query embedding.
- TQH-n (n=1 or 5): `g_m = normalize(W_m z_q)` (V5's OWN `SlotHeads`,
  same W/init/std) attends over a per-query temporal token bank built
  by re-running the SAME shared MLP relation-encoder on P
  non-overlapping time-masked copies of the exact same relation input,
  minus a zero-input baseline (`z_0`), then `q_m = normalize(sum_p
  softmax_p(g_m . t_p) * t_p)` -- **no 1/sqrt(D) scaling** (removed
  after diagnosing near-uniform attention; both vectors are already
  L2-unit-norm, so the standard Transformer variance correction didn't
  apply here -- see `models/TemporalQueryHeads.py` docstring). No
  fixed or learnable temperature, no diversity loss, candidate branch/
  loss/teacher/Stage-2/checkpoint-selection all byte-for-byte identical
  to V5.

Parameter-count fairness: TQH-1 (16,384) / TQH-5 (81,920) `SlotHeads`
params are IDENTICAL to V1/V5's own `SlotHeads` param counts for the
same `n_slots` -- asserted at startup, zero new trainable parameters
anywhere in the head-generation mechanism (attention is a pure
function of `g`/`t_p`).

## 2. Table 1 -- Stage-2 Forecast (ETTh1_96, test)

| Arm | Test MSE | Test MAE | Delta MSE vs V5 |
|---|---:|---:|---:|
| V1 (existing) | 0.39421 | 0.40596 | +0.01% |
| V5 (existing) | 0.39426 | 0.40604 | -- |
| TQH-1 | 0.37176 | 0.39553 | **-5.70%** |
| TQH-5 | 0.37173 | 0.39553 | **-5.71%** |

## 3. Table 2 -- Retrieval Quality (test, restricted-P100 oracle)

| Arm | RetMSE@10 | AggMSE@10 | Recall@10(P100) | NDCG@10 |
|---|---:|---:|---:|---:|
| V1 | 1.6154 | 1.0114 | 0.0557 | 0.5816 |
| V5 | 1.6219 | 1.0137 | 0.0551 | 0.5813 |
| TQH-1 | 0.6762 | 0.4158 | 0.2406 | 0.8687 |
| TQH-5 | 0.6765 | 0.4158 | 0.2402 | 0.8686 |

## 4. Table 3 -- Head Complementarity (TQH-5 only; TQH-1 has 1 head, no pairs)

| Metric (test, best_retmse ckpt, epoch 7) | Value |
|---|---:|
| Pairwise Top-10 overlap (slot_top10_overlap) | 0.9769 |
| Pairwise head Top-1 overlap (slot_top1_overlap) | 0.9574 |
| Union size (max 50 = 5 heads x 10) | 10.51 |
| Union Oracle Recall, 1 head | 0.2430 |
| Union Oracle Recall, 2 heads | 0.2443 |
| Union Oracle Recall, 3 heads | 0.2469 |
| Union Oracle Recall, 4 heads | 0.2469 |
| Union Oracle Recall, 5 heads | 0.2519 |

Adding heads 2-5 moves Union Oracle Recall from 0.2430 (1 head) to
0.2519 (5 heads) -- a **+3.7% relative gain**, essentially the entire
possible union-size headroom (10.5/50) delivering almost nothing.

## 5. Table 4 -- Attention Diagnostics (TQH-5, test, best_retmse ckpt)

| Metric | Value (1.0 = fully uniform / fully collapsed) |
|---|---:|
| Normalized attention entropy | 0.9915 |
| Mean max attention probability | 0.2099 (uniform over 6 patches = 0.1667) |
| Pairwise head attention cosine | 0.9988 |
| Query-to-query attention diversity | 0.9794 |

Trajectory across all 10 training epochs (not just final): pairwise
head cosine stayed in [0.9983, 1.0000], slot_top10_overlap moved only
from 0.9977 (epoch 1) to 0.9769 (epoch 10) -- a very slow, very small
drift, never remotely approaching real differentiation.

## 6. Answers to the three required questions

**Q1. Do TQH-1 and TQH-5 actually differ in retrieval/Stage-2 performance?**
No, not meaningfully. RetMSE@10: 0.6762 vs 0.6765 (+0.04%). AggMSE@10:
0.4158 vs 0.4158 (~0%). Stage-2 test MSE: 0.37176 vs 0.37173 (-0.01%,
TQH-5 is actually *very slightly* better, opposite sign from the
Stage-1 numbers -- both differences are well inside run-to-run noise
for a single seed). The 4 extra heads contribute nothing measurable.

**Q2. Are TQH-5's 5 heads still collapsed after full (not pilot) training?**
Yes. See Tables 3-4: pairwise cosine 0.9988, query diversity 0.9794,
Top-10 overlap 0.9769, Union Oracle Recall gain from heads 2-5 combined
is +3.7% relative. This holds at epoch 10 just as it held at epoch 1 --
full training did not resolve the collapse, it only drifted very
slightly over 10 epochs.

**Q3. Is TQH-1 (single head) also better than V5, pointing to the query
-construction mechanism itself rather than multi-head diversity?**
Yes, clearly. TQH-1 alone reaches RetMSE@10=0.6762 (vs V5's 1.6219,
-58.3%), AggMSE@10=0.4158 (vs 1.0137, -59.0%), and Stage-2 test
MSE=0.37176 (vs V5's 0.39426, -5.70%) -- using the IDENTICAL masked
-temporal-token/attention construction as TQH-5, but with only one
head. It alone accounts for essentially all of the improvement over V5.

## 7. Verdict: **CASE A**

> TQH-1 approx= TQH-5 > V5. Multi-head contribution is NOT supported
> by this cell's data.

Per the pre-registered decision rule, this points the hypothesis away
from "complementary multi-head retrieval" and toward "temporal-aware
single-query representation" as the actual source of improvement --
masking out most of the time axis and re-running the SAME MLP encoder,
then baseline-subtracting and attention-pooling the resulting patch
embeddings, appears to produce a materially better retrieval key than
a single static linear projection of the pooled embedding, independent
of how many heads attend over that token bank.

## 8. What this does NOT yet establish

- **One cell, one seed.** No replication across ETTh1 H192/336/720 or
  any Weather cell yet. The Stage-1 retrieval gain is large (-58%
  RetMSE@10) but the end-to-end Stage-2 gain is far more modest (-5.7%
  MSE) because Base dominates the learned fusion (lambda~=0.42) -- this
  balance could shift differently on other horizons/datasets.
- **Mechanism not yet explained.** Why the masked-token/baseline
  -subtracted/attention-pooled construction outperforms V5's static
  linear projection is not understood -- candidate explanations (the
  `z_0` subtraction acting as an implicit per-query centering/
  normalization V5 lacks; the patch-averaging acting as an incidental
  denoising ensemble) are untested hypotheses, not findings.
- **Not yet ruled out: a confound in TQH's single-query path itself**
  rather than something attributable to "temporal masking" per se --
  e.g., whether the SAME gain would appear from a degenerate variant
  that attends over P *random* (non-temporally-structured) maskings of
  the input rather than genuinely temporally contiguous patches. This
  track has NOT run that control.

## 9. Anomalies / surprises worth flagging

- TQH-1 is marginally *better* than TQH-5 on 2 of 4 Stage-1 metrics and
  on Stage-2 MSE -- the "extra" heads are not just useless, they are
  very slightly net-negative here (within noise, but consistently so
  across both the cache-build and Stage-2 re-runs). Not claimed as a
  real effect given the margin, but notable that more heads never
  helped even slightly on this cell.
- V1 and V5's own existing numbers are themselves nearly identical to
  each other (Stage-2 test MSE 0.39421 vs 0.39426) -- consistent with
  this project's own prior finding that V5's extra static heads add
  little over V1 on ETTh1 (see `REVIEW_FOR_CHATGPT.md` V-family
  history); this is additional context for how large the TQH gain is
  by comparison, not something newly discovered here.

## 10. Recommended next step

Per CASE A's pre-registered action: do NOT pursue temperature-tuning
or diversity losses to force head differentiation (the user's standing
instruction for this stage). Before expanding to the full grid, the
random-masking control (point 8, last bullet) would most directly test
whether "temporal structure" specifically matters, or whether any
masked-averaging-of-the-same-encoder construction would do. This
decision is left to the user / ChatGPT PI review, not decided here.

## Reproducibility

- Code: `models/TemporalQueryHeads.py`, `scripts/train_temporal_query_heads01.py`,
  `scripts/build_tqh_retrieval_cache01.py`.
- Raw artifacts: `results/TRACK-TEMPORAL-QUERY-HEADS01/ETTh1_96/ETTh1_96/{TQH1Masked,TQH5Masked}/`
  (`final_test_metrics_best_retmse.json`, `val_metrics.csv`, `config.json`,
  `stage2/metrics.json`); checkpoints under
  `checkpoints/track_temporal_query_heads01/ETTh1_96/ETTh1_96/`.
- V1/V5 baselines reused UNMODIFIED from
  `results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/ETTh1_96/{V1,V5}/`.
- Init-hash parity (TQH vs V5's own recorded `init_hash`
  `0f044e3ce726fc5c...`) confirmed byte-identical at startup for both
  TQH-1 and TQH-5.
