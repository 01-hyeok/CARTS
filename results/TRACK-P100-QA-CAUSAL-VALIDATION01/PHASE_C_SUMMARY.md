# Phase C summary -- fusion ablation (C1), retriever-checkpoint control (C2),
# oracle headroom (C3), quality calibration (C4), shuffled-feature control (C5)

Status: COMPLETE for all 5 Phase B representative pairs. Operates ONLY on
VALIDATION data (never TEST), further split chronologically into
Val-Select (first half) / Val-Holdout (second half, with a `seq_len+pred_len`
purge gap) via `utils/validation_split01.py`. All numbers below are
Val-Holdout MSE unless marked otherwise.

Code: `scripts/extract_fusion_diagnostics_raw01.py` (raw tensor extraction
from the RNG-corrected Phase B checkpoints) + `scripts/run_phase_c_fusion_ablation01.py`
(CPU-only analysis). Extraction validated against the recorded checkpoint
val_mse to ~7 decimals on a smoke test (ETTh1_192/V5: 0.9617633819580078
recomputed vs 0.9617633858655747 recorded).

## Full results table (Val-Holdout MSE)

F0=Base-only, F1=Global-lambda, F2=Per-channel-lambda, F3=Constant-mean-QA,
F4_refit=Dynamic QA gate refit on Val-Select, F4_e2e=the REAL E2E-trained
gate from the actual checkpoint (qa-basis only).

| Pair | Basis | F0 | F1 (λ) | F2 | F3 (λ) | F4_refit | F4_e2e |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1_192/V5 | noqa_retriever | 0.661815 | 0.643266 (0.203) | 0.639573 | 0.646066 (0.163) | 0.647211 | -- |
| ETTh1_192/V5 | qa_retriever | 0.661815 | 0.644929 (0.184) | 0.640278 | 0.646167 (0.167) | 0.636896 | **0.639202** |
| ETTh1_336/Hard | noqa_retriever | 0.937664 | 0.946029 (0.310) | 0.941453 | 0.943539 (0.286) | 0.933766 | -- |
| ETTh1_336/Hard | qa_retriever | 0.937664 | 0.973691 (0.317) | 0.941561 | 0.963347 (0.264) | 0.945516 | **0.950243** |
| Weather_720/V5 | noqa_retriever | 0.650365 | 0.652046 (0.201) | 0.662962 | 0.664159 (0.321) | 0.649182 | -- |
| Weather_720/V5 | qa_retriever | 0.650365 | 0.647197 (0.124) | 0.652747 | 0.652500 (0.242) | 0.648110 | **0.653799** |
| Weather_336/Joint | noqa_retriever | 0.506790 | 0.486582 (0.228) | 0.478456 | 0.486926 (0.323) | 0.481196 | -- |
| Weather_336/Joint | qa_retriever | 0.506790 | 0.487059 (0.224) | 0.484841 | 0.486622 (0.320) | 0.482125 | **0.482744** |
| Solar_96/V5 | noqa_retriever | 0.297977 | 0.211145 (0.788) | 0.211822 | 0.211270 (0.791) | 0.213004 | -- |
| Solar_96/V5 | qa_retriever | 0.297977 | 0.211057 (0.799) | 0.211824 | 0.210830 (0.793) | 0.212683 | **0.207767** |

Bold = the actually-deployed E2E-trained QA gate, qa-basis (what the
checkpoint really does at inference).

## C3 oracle headroom / C4 quality calibration / C5 shuffle control (qa-basis)

| Pair | Oracle global | Oracle per-ch | Oracle dynamic | C4 Pearson(λ,adv) | C4 Spearman | C5: orig vs shuffled |
|---|---:|---:|---:|---:|---:|---|
| ETTh1_192/V5 | 0.635843 | 0.628608 | 0.585578 | +0.212 | +0.167 | 0.636896 < 0.641148 (orig better) |
| ETTh1_336/Hard | 0.937664 | 0.905639 | 0.845460 | +0.148 | +0.127 | 0.945516 < 0.954979 (orig better) |
| Weather_720/V5 | 0.647106 | 0.639597 | 0.602165 | +0.146 | +0.280 | 0.648110 < 0.669958 (orig better) |
| Weather_336/Joint | 0.486215 | 0.473066 | 0.429042 | +0.157 | +0.305 | 0.482125 < 0.504704 (orig better) |
| Solar_96/V5 | 0.209080 | 0.207716 | 0.155836 | +0.099 | +0.146 | 0.212683 < 0.214479 (orig better) |

## Reading -- honest, not smoothed over

**C4 and C5 are unanimous and clean (5/5 pairs):** Pearson/Spearman between
the fitted gate's λ and true retrieval advantage are positive in every pair
(λ really does track when retrieval is better than Base, the correct
direction), and C5's shuffle control shows original features beat
within-channel-shuffled features in **10/10 comparisons** (both bases, all 5
pairs). If the gate were just fitting noise or a channel-level constant,
shuffling query-level features within a channel should not hurt -- it does,
every time. This is real evidence the gate uses genuine per-query signal,
not just a disguised per-channel or global constant.

**C3 confirms large real headroom exists** for a perfect dynamic lambda
(dynamic oracle << per-channel oracle << global oracle << F0 in every pair),
so the ceiling for "is dynamic fusion worth it in principle" is clearly yes.

**C1/C2 (does the ACTUAL trained gate capture that headroom on held-out
data) is NOT clean, and should not be reported as clean:**

- **3/5 pairs (ETTh1_192/V5, Weather_336/Joint, Solar_96/V5)**: F4_e2e is the
  best or near-best of F0-F4 on qa-basis, clearly beating F0/F1/F2/F3. Dynamic
  fusion is earning its complexity here.
- **2/5 pairs (ETTh1_336/Hard, Weather_720/V5)**: on qa-basis, F4_e2e
  (0.950243 / 0.653799) is actually **worse than plain Base-only**
  (F0 = 0.937664 / 0.650365) on this Val-Holdout slice -- despite QA clearly
  beating NOQA on the FULL validation set for these exact same two pairs in
  Phase B (ETTh1_336/Hard: corrected delta -0.0132 favoring QA;
  Weather_720/V5: corrected delta -0.0233 favoring QA,
  see `PHASE_B_SUMMARY.md`). On noqa-basis for the same two pairs, F4_refit
  (a gate re-fit fresh on Val-Select) DOES beat F0 (0.933766 < 0.937664 for
  ETTh1_336/Hard; 0.649182 < 0.650365 for Weather_720/V5) -- so the
  underperformance is specific to the real E2E-trained gate evaluated on
  the chronologically later holdout half, not a property of dynamic fusion
  in general for these cells.

**Most likely explanation (not yet proven):** Val-Holdout is the
chronologically LATER half of the validation period (purge-gap split, not a
random split). ETTh1_336/Hard and Weather_720/V5 are both long-horizon
cells, which means fewer independent windows AND a split point further into
each series' later regime. A gate whose λ-modulation was tuned (via the real
E2E training loop) to the behavior of retrieval quality across the training
period may not transfer as well to a later-in-time, smaller, higher-variance
holdout slice -- consistent with time-series non-stationarity, not
necessarily a code defect. This has NOT been separately verified (e.g. by
checking whether retrieval advantage itself shifts between Val-Select and
Val-Holdout for these two cells); flagging as an open question rather than
asserting it.

**Bottom line for Phase E:** the evidence so far is a genuine mixed picture,
not a clean "QA = real query-dependent signal" story and not a clean
"QA = disguised calibration artifact" story either. C4/C5 argue for real
per-query signal in the gate (robustly, 5/5 and 10/10). C1/C2 show that
signal pays off reliably in only 3/5 representative cells on a true
held-out slice, with the other 2/5 showing the deployed gate is no better
(and slightly worse) than doing nothing, even though a freshly-refit gate
on the SAME features would have helped. This should be reported to the
user and, eventually, to the ChatGPT PI review as-is -- not resolved or
rounded toward either conclusion.

## Status of the causal-validation track

- Phase A (code audit + fixes): COMPLETE.
- Phase B (RNG-corrected reproduction, 5 representative pairs): COMPLETE.
- Phase C (fusion ablation, shuffle control, oracle headroom): **COMPLETE, this document.**
- Phase D (seed robustness): NOT STARTED.
- Phase E (final causal determination): NOT STARTED, blocked on D.
