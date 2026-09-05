# CURRENT_EXPERIMENT.md

Status: **DESIGN ONLY — NOT APPROVED, NOT STARTED. No code written, no GPU used.**

EXP-3 (soft_set_mse) closed with verdict STOP (see `EXPERIMENT_LOG.md` →
EXP-3-CLOSURE, `REVIEW_FOR_CHATGPT.md`). This is the candidate next design,
written per instruction "design only" — Claude does not choose to start this;
that is the user's and the reviewer's decision.

---

# Research Question

Why does the current past-only retriever fail to find Individual-Oracle-level
candidates at all (the primary, larger bottleneck), independent of the
set-composition question EXP-3 tested? Concretely: does separating **coarse
retrieval** (cheap, full-memory, cosine-based) from **fine reranking**
(expensive, small shortlist, learned) let a reranker improve real Top-10
retrieval without the full-memory softmax diffusion that sank EXP-3?

# Hypothesis

The full-memory soft objective in EXP-3 failed partly because a softmax over
tens of thousands of candidates has no floor on how diffuse it can get (Weather
N_eff hit 8870/700 out of ~36,700). Restricting the learned objective to a
fixed, small coarse shortlist (e.g. the current retriever's own Top-100)
removes that degree of freedom by construction — the reranker cannot diffuse
across more candidates than the shortlist contains.

# Motivation

- EXP-1/EXP-2 (Oracle Intervention): a Weighted-Set Oracle beats an Individual
  Oracle on A_weighted at every horizon on both datasets (17.7-43.2%
  improvement) — real headroom exists in candidate *combination*.
- EXP-3 closure (this document's sibling): training a full-memory soft
  aggregate loss to reach that headroom made Stage-2 worse at all 4
  representative cells, confounded by soft/hard mismatch (ETTh1) and
  N_eff/collapse (Weather).
- Before trying another full-memory objective, check whether the *support*
  itself (P100 vs FULL) is the larger bottleneck. EXP-1's own P100 vs FULL
  comparison (already in `EXPERIMENT_LOG.md`, EXP-1 section) showed Individual
  Oracle A_weighted improves substantially from P100 to FULL support at ETTh1
  H96 (0.1442 P100 → lower under FULL — recheck exact figures before running),
  meaning the *current* retriever's Top-100 already excludes candidates the
  oracle would want. A reranker confined to that Top-100 inherits that ceiling
  — this experiment does not fix the coarse-retrieval bottleneck, only
  isolates whether a reranker can capture set-composition value **within**
  whatever the coarse stage already offers, cleanly separated from diffusion.

# Baseline

S0 (WCE, plain cosine retriever), the same checkpoints already trained in
EXP-3, at ETTh1 H96/H720 and Weather H96/H720. No retraining of Stage-1.

# Experimental Change (the only component that changes)

Add a reranker stage between coarse retrieval and Stage-2:

```
Full memory --[frozen S0 retriever, cosine]--> Top-100 (fixed shortlist)
Top-100     --[NEW: learned reranker]--------> Top-10 (fed to Stage-2)
```

The reranker's objective is a set-level utility (softmax aggregate error),
identical in form to `soft_set_mse`, but the softmax runs **only over the
100-candidate shortlist**, not the full bank — this is the one thing EXP-3 did
not test and the one change this experiment isolates.

Everything else stays exactly as EXP-3/EXP-1 held it: Stage-1 S0 encoder
frozen, Stage-2 architecture/gate/fusion/`top_k`/`tau_topk` unchanged,
`relation_top_n=1` (self-only, not mixed with the cross-channel question).

# Controlled Variables

- Coarse retriever: frozen S0 (WCE, cosine) checkpoint, unchanged.
- Coarse shortlist size: 100 (same P100 definition already used and verified
  in EXP-1/EXP-2's oracle intervention).
- Stage-2 architecture, gate, fusion, `top_k=10`, `tau_topk=0.1`: unchanged.
- Seed, split, candidate masking: unchanged.
- Checkpoint selection: validation, not test (same discipline as EXP-3).

# Dataset

ETTh1 and Weather, same as EXP-3.

# Prediction Horizons

H96 and H720 only, matching EXP-3's representative-cell precedent — do not
expand to H192/H336 until this shows a clear signal at the cheaper horizons.

# Loss

Reranker-only softmax aggregate loss over the fixed 100-shortlist:

```
p_i = softmax(reranker_score_i / tau)     i in the 100-candidate shortlist only
Y_ret_soft = sum_i p_i * Y_i
L = MSE(Y_ret_soft, Y_q)
```

`tau` requires its own Step-0 calibration **on the 100-candidate support**,
not reused from EXP-3's full-memory calibration (support size changed, so the
temperature-to-N_eff mapping changed) — this is the same calibration procedure
already implemented and acceptance-tested (`tau_calibration_diag`), pointed at
the 100-candidate case.

# Metrics

Same as EXP-3, plus the shortlist-vs-support distinction made explicit:

- `HardAggregateMSE@10` computed with the Top-10 the reranker actually selects
  from the 100-shortlist (never from the full bank at eval time).
- `N_eff` (should now be bounded above by 100 by construction — report it to
  confirm this, not to newly discover a diffusion problem).
- `effective_rank` / mean pairwise cosine (collapse check, same as EXP-3).
- Stage-2 Final MSE, downstream, on validation for arm selection and test for
  the reported number (same discipline as EXP-3/EXP-1).
- Recall@10 reported, not used as the success criterion (established finding:
  it does not track Stage-2).

# Success Criterion

`HardAggregateMSE@10` (hard, Top-10, 100-shortlist-constrained) improves over
S0's own Top-10 **and** that improvement reaches Stage-2 Final MSE beyond the
≈0.01 MSE seed-noise floor, at both H96 and H720, on at least one dataset,
without `N_eff` approaching the shortlist size (100) or `effective_rank`
dropping further than S0's own baseline.

# Failure Criterion

Any of: Stage-2 does not improve beyond noise; `N_eff` saturates near 100
(diffusing across the whole shortlist, the small-scale analogue of EXP-3's
failure); `effective_rank` collapses further than the frozen S0 baseline
(would indicate the reranker itself is degenerating, since the encoder is
frozen and cannot itself collapse — a nonobvious result worth flagging rather
than average over, since it is not supposed to be able to happen with a frozen
encoder); or the result is confined to one dataset only in a way that cannot
be distinguished from that dataset's already-known bottleneck differences
(e.g. Weather's baseline collapse) rather than the reranking idea itself.

# Expected Compute

Reranker is a small module (comparable to the existing `PairwiseScorer`) scored
over only 100 candidates per query — orders of magnitude cheaper per step than
EXP-3's full-memory softmax over ~8,000-36,700 candidates. Rough estimate:
Stage-1-equivalent reranker training at ETTh1 scale (~minutes based on this
project's existing Stage-1 timing), Weather scale proportionally longer given
its larger channel count, but still far below EXP-3's Weather Stage-1 run
times (~40min/run) since the candidate axis shrinks by 2-3 orders of
magnitude. Full estimate to be refined at implementation time, before any GPU
run.

# Decision Criterion

If the success criterion is met on both datasets at both horizons: proceed to
the full H192/H336 horizons and consider this the retrieval-quality-improving
direction to develop further. If it succeeds on one dataset only: treat as
dataset-dependent and investigate that dataset's specific bottleneck (Weather's
known representation collapse) before generalizing. If it fails on both: the
bottleneck is more likely in the coarse stage (the P100 shortlist itself
excluding good candidates, per EXP-1's P100-vs-FULL gap) than in what a
reranker can do once handed the shortlist — the next question becomes
improving the coarse retriever, not the reranker.
