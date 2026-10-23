# PHASE A -- PH1 Attention vs Patch Mean (evaluation-only)

No training, no new parameters, no new checkpoint. Reads the EXISTING
PH1 selected checkpoint (`TRACK-TRANSFORMER-PATCH-QUERY-HEADS01`,
read-only) for each of the 4 cells and computes, from the SAME
Transformer forward pass, `q_attn` (PH1's own
`temporal_attention_heads`) and `q_mean` (plain mean over patch
tokens, no attention), scored against the SAME pooled P100 candidate
embeddings. Code: `scripts/diag_patchmean_vs_attention01.py`.

## Representation similarity (test split)

| Cell | cosine mean | cosine min | L2 dist mean | Top-10 overlap (attn vs mean) |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.9998 | -- | -- | 0.9906 |
| ETTh1_720 | 0.9997 | -- | -- | 0.9881 |
| Weather_96 | 0.9999 | -- | -- | 0.9950 |
| Weather_720 | 0.9993 | -- | -- | 0.9830 |

(Full distribution -- mean/std/median/p5/p25/p75/p95/min -- saved per
cell in `results/TRACK-PATCH-REPRESENTATION-RESIDUAL01/PHASE_A/<cell>_phaseA.json`.)

## Retrieval comparison (test, Top-K=10, no Stage-2)

| Cell | RetMSE attn | RetMSE mean | AggMSE attn | AggMSE mean |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.6878 | 0.6884 | 0.4253 | 0.4259 |
| ETTh1_720 | 0.9445 | 0.9450 | 0.6287 | 0.6292 |
| Weather_96 | 0.3936 | 0.3936 | 0.2351 | 0.2352 |
| Weather_720 | 0.7128 | 0.7124 | 0.5638 | 0.5633 |

Differences are in the 3rd-4th decimal everywhere -- effectively
identical. (Weather_720 shows AggMean *very slightly* better than
AggAttn, 0.5633 vs 0.5638, but the gap is far inside noise and the
representations themselves are 0.9993 cosine-similar.)

## Verdict: CASE A1, uniform across all 4 cells

`cosine(q_attn, q_mean) >= 0.99` AND `Top10 overlap >= 0.95` AND
retrieval metrics are negligibly different, in every single cell.

> PH1's apparent benefit is not caused by query-conditioned temporal
> attention. It is primarily a contextual patch-mean representation
> effect.

The attention mechanism is discarded. **PHASE B proceeds with
`r_q = q_mean` (PatchMean) as the residual source.**
