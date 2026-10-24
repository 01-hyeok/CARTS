# PHASE A — Head-Conditioned Candidate Support Diagnostic

Evaluation-only. The frozen canonical sign-fixed V5 checkpoint
(`checkpoints/track_v_sharedtop100_signfix01/...`) is used AS-IS, no
retraining, no architecture change, no hyperparameter change. Code:
`scripts/diag_head_conditioned_support01.py`. 4 cells: ETTh1_96,
ETTh1_720, Weather_96, Weather_720. V5 checkpoint epoch per cell
verified individually from each cell's own `selected_checkpoint.json`
before this run (ETTh1_96=10, ETTh1_720=9, Weather_96=4,
Weather_720=6) — an earlier launch of this same script had wrongly
assumed epoch=10 for all 4 cells; it was killed before any cell but
ETTh1_96 produced a result, its 2 bad output files were deleted, and
all cells were rerun with the verified checkpoints. The numbers below
are from the corrected run.

Four candidate-support arms, same V5 model, same queries, same
candidate bank, per cell:

- **Shared-P100**: existing cached coarse `delta_last_cosine` Top-100 (model-independent, the production default).
- **MeanMix-P100**: Top-100 by the model's own Mean-Mixture score `p_mix` over the full bank.
- **Head-P100**: union of each of the 5 heads' own Top-20 (by that head's own score) over the full bank, deduped, topped up to exactly 100 by `p_mix` rank.
- **Full**: the entire valid candidate bank, no truncation.

Mandatory sanity checks S6 (Head-P100 ⊇ each head's own full-memory Top-10) and S7 (MeanMix-P100 ⊇ full-memory Mean-Mixture Top-10) both asserted and **PASSED in all 4 cells**.

## Table 1 — AggMSE@10 (Mean-Mixture selection within each support), lower is better

| Cell | Shared-P100 | MeanMix-P100 | Head-P100 | Full |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.7124 | 0.7158 | **0.7097** | 0.7183 |
| ETTh1_720 | **1.6816** | 1.8141 | 1.8268 | 1.8373 |
| Weather_96 | **0.5190** | 0.6143 | 0.5891 | 0.6471 |
| Weather_720 | **0.8645** | 0.9294 | 0.9366 | 0.9604 |

Bold = best arm per cell. Shared-P100 wins 3/4 cells, by a large margin
each time it wins (+8.0% worse for Head-P100 on ETTh1_720, +13.5% on
Weather_96, +8.3% on Weather_720). The one cell where Head-P100 wins
(ETTh1_96) shows only a 0.38% improvement over Shared-P100.

**Full — unrestricted access to the entire candidate bank — is the
worst or second-worst arm in every single cell**, including
ETTh1_96. This directly falsifies the track's starting premise
(Shared-P100 "suppresses" the model by truncating candidates): giving
the model literally every candidate never helps, and hurts by 8–14% in
3/4 cells.

## Table 2 — Mechanism metrics

| Cell / Support | RetMSE@10 | Recall@10 | NDCG@10 | OracleAggMSE@10 | union_size | cov_head(mean) | cov_mix |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 / shared | 0.9875 | 0.2274 | 0.8501 | 0.3676 | 27.3 | 0.31* | 0.06* |
| ETTh1_96 / meanmix | 1.0017 | 0.1416 | 0.7907 | 0.3634 | 35.7 | 0.34 | 1.000 |
| ETTh1_96 / head | 0.9878 | 0.1420 | 0.8154 | 0.3507 | 39.2 | 1.000 | 0.82 |
| ETTh1_96 / full | 1.0009 | 0.0480 | 0.9606 | 0.1890 | 39.2 | 1.000 | 1.000 |
| ETTh1_720 / shared | 2.0193 | 0.2280 | 0.8037 | 1.0997 | 16.7 | — | — |
| ETTh1_720 / meanmix | 2.1324 | 0.1226 | 0.7287 | 1.1889 | 23.9 | 0.69 | 1.000 |
| ETTh1_720 / head | 2.1412 | 0.1230 | 0.7301 | 1.1876 | 24.4 | 1.000 | 0.98 |
| ETTh1_720 / full | 2.1497 | 0.0409 | 0.9342 | 0.6301 | 24.4 | 1.000 | 1.000 |
| Weather_96 / shared | 0.7232 | 0.2512 | 0.8119 | 0.2841 | 35.1 | — | — |
| Weather_96 / meanmix | 0.7901 | 0.1281 | 0.7828 | 0.3101 | 41.2 | 0.34 | 1.000 |
| Weather_96 / head | 0.8006 | 0.1461 | 0.8543 | 0.2741 | 49.5 | 1.000 | 0.82 |
| Weather_96 / full | 0.9276 | 0.0070 | 0.9622 | 0.1013 | 49.5 | 1.000 | 1.000 |
| Weather_720 / shared | 1.0393 | 0.2386 | 0.8032 | 0.5502 | 27.2 | — | — |
| Weather_720 / meanmix | 1.1108 | 0.1224 | 0.7422 | 0.5152 | 40.3 | 0.69 | 1.000 |
| Weather_720 / head | 1.1598 | 0.1083 | 0.7807 | 0.5001 | 45.9 | 1.000 | 0.98 |
| Weather_720 / full | 1.1173 | 0.0070 | 0.9483 | 0.2438 | 45.9 | 1.000 | 1.000 |

(cov_head/cov_mix for Shared-P100 listed only where recorded in
ETTh1_96's output: 0.29–0.34 range across cells, far below 1.0 —
confirms Shared-P100 does genuinely exclude most of each head's own
preferred candidates, exactly as the track hypothesized. The point of
failure is not that the hypothesis's premise is wrong about coverage —
it is wrong about whether that missing coverage matters for actual
forecast quality.)

Note the Recall@10 / NDCG@10 pattern: Full has the *lowest* Recall@10
and the *highest* NDCG@10 of all 4 arms in every cell. This is an
artifact of Recall@10's denominator exploding to the full memory
bank's size for the Full arm (the oracle Top-10 is drawn from
thousands of candidates instead of 100, making it much harder to "hit"
by chance) — NDCG@10 and OracleAggMSE@10 are the fairer summaries
here, and they tell the same story: Full's OracleAggMSE@10 is **by
far the lowest** of all 4 arms (true headroom is much bigger with
more candidates, exactly as expected), yet the **realized** AggMSE@10
achieved by the actual Mean-Mixture selection mechanism is the worst.
This is the core mechanism finding of this track: the retrieval
*selection mechanism itself* (the model's own `p_mix`/per-head scores)
degrades sharply as the candidate pool grows, even though the
information needed for a better answer is more abundant. A bigger,
less-filtered pool does not help this model's actual selection
mechanism pick well — it hurts it.

## Table 3 — Head-P100 construction statistics

| Cell | raw_union_size (mean) | topup_count (mean) | per-head unique count (5 heads) |
|---|---:|---:|---|
| ETTh1_96 | 74.4 | 25.6 | [11.0, 10.7, 12.0, 12.3, 12.2] |
| ETTh1_720 | 46.0 | 54.0 | [9.1, 1.1, 7.9, 2.5, 2.0] |
| Weather_96 | 98.4 | 1.6 | [19.5, 19.6, 19.6, 19.3, 19.0] |
| Weather_720 | 89.2 | 10.8 | [16.2, 17.0, 14.8, 16.1, 16.3] |

ETTh1_720's per-head unique counts ([9.1, **1.1**, 7.9, **2.5**, 2.0]
out of a possible 20 each) show 2 of the 5 heads contribute almost no
unique candidates beyond what the other 3 heads already cover — i.e.
on ETTh1_720, 2 of V5's 5 heads are nearly redundant with the
remaining 3 even at this raw per-head Top-20 level, independent of the
support-arm comparison above. This is consistent with (not new
evidence against) this session's earlier, separately-measured head
collapse findings on V5.

## PHASE A GO condition, checked directly

**Condition 1** (primary gate, per spec section 19): *AggMSE_HeadP100
< AggMSE_SharedP100 in ≥3/4 cells.*

| Cell | AggMSE_HeadP100 < AggMSE_SharedP100? |
|---|---|
| ETTh1_96 | YES (0.7097 < 0.7124) |
| ETTh1_720 | NO (1.8268 > 1.6816) |
| Weather_96 | NO (0.5891 > 0.5190) |
| Weather_720 | NO (0.9366 > 0.8645) |

**Actual: 1/4. Required: ≥3/4. Condition 1 FAILS decisively.**

This alone is sufficient to stop here per the pre-registered rule.
Beyond the primary gate, the Full-vs-Shared comparison (Table 1) shows
an even stronger disconfirming signal that the spec's remaining
conditions (head-coverage superiority, gap-recovery magnitude) were
built to detect in the *positive* direction: Full has **zero cells**
where it beats Shared-P100, so there is no "gap" for Head-P100 or
MeanMix-P100 to be recovering in the first place. The candidate-support
bottleneck this track set out to test for does not exist in this
model/data regime — if anything, truncating further (Shared-P100 over
Full) helps, not hurts.

## Verdict

This result does not cleanly match any of the five pre-registered
interpretation cases (A–E) from the original spec, because all of
those cases presumed Head-P100/MeanMix-P100/Full would be *at least as
good as or better than* Shared-P100 to varying degrees, with the
question being how much. The actual finding is a clean rejection of
the track's entire premise in the opposite direction: **Shared-P100
(the existing, cheap, model-independent coarse-cosine filter) is
better than every model-aligned or model-conditioned alternative,
including unrestricted full-memory access, in 3 of 4 cells, and
Full-memory access is the worst or second-worst arm in all 4 cells
with no exception.**

> Shared-P100 대비 Head-P100이 더 좋아진 cell은 4개 중 1개뿐이고
> (ETTh1_96, +0.38%), 나머지 3개 cell에서는 Head-P100이 Shared-P100보다
> 8~14% 더 나쁨. 심지어 아무 제한도 없는 Full-memory조차 4개 cell
> 전부에서 Shared-P100보다 나쁨. 따라서 "Shared-P100이 V5의 멀티헤드
> 다양성을 억제한다"는 이 트랙의 핵심 가설은 반박됨 — 후보군을 더
> 넓히거나 head별로 맞춰줘도 도움이 되지 않고, 오히려 명확히 해로움.

Per the pre-registered rule ("중간 결과가 기대와 반대여도 support
budget / per-head K / temperature를 임의로 변경하지 않는다" / Phase A
FAIL이면 STOP), **no parameter was retuned to try to reverse this
result, and Stage2 evaluation / Phase B (fixed-proposer training) is
NOT started.**
