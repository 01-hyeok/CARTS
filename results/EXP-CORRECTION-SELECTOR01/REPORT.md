# EXP-CORRECTION-SELECTOR01 (Track B3) — Report

Frozen Stage-2 checkpoint (`checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_.../checkpoint.pth`,
the SAME checkpoint used throughout Track B1/B2) provides the encoder
embeddings, `base_forecast`, and the reference retrieval score
(`learned_ref`) used to weight the correction aggregate. B0/Stage-2 stay
fully frozen throughout — the only new trainable modules are a
`SetConditioner`+`EmptySetToken`+`UtilityHead` selector (49,794 params,
identical architecture/size to Track A's own T1/OPC1 selectors) and, in
the second phase, a fresh per-channel `RetrievalGate`. Ran on GPU 1 in
parallel with, independently of, Track A's `EXP-ONPOLICY-CHOICE-
GENERALIZATION01` — touches none of Track A's checkpoints or files.

## Research question

Track B1 found the Correction Set Oracle beats the Future Set Oracle on
4/7 channels and the cross-channel mean (a true-future upper bound).
Track B2 found that reusing the EXISTING future-oriented Top-K for a
correction-value fusion does NOT realize that headroom (F0=0.37312 beats
both B2-C0=0.39056 and B2-C1=0.38337 on 6/7 channels). **Does a selector
trained DIRECTLY toward the correction objective — Choice CE against
`u_i^(t) = -MSE(B_q + C(S_{t-1}+{i}), Y_q)`, recomputed from the model's
own on-policy prefix at every step, mirroring `EXP-ONPOLICY-CHOICE01`'s
(OPC1) training pattern exactly — recover that headroom in actual
retrieval and downstream forecasting?**

## Method

`scripts/train_correction_selector01.py`. Scorer input is UNCHANGED from
every other arm in this project: the frozen encoder's own L2-normalised
`_branch_embedding`/`_branch_memory` output. Residuals `r_i = Y_i - B_i`
(candidate's own past only, via `base_forecast`) NEVER enter the scorer —
used only to build the Choice-CE target (via `dense_utility`, called with
residuals instead of futures — exact reuse, no new math) and as the
retrieved correction VALUE. On-policy prefix construction, no-grad
argmax, full-memory (no shortlist) scoring: identical to
`train_onpolicy_choice01.py`'s own pattern.

Two evaluation arms after training, both using the trained selector's OWN
on-policy Top-K (never the existing future-oriented retrieval):
- **CorrSelector-Fixed** (γ=1, no training): `Y_final = B_q + C_ret`.
- **CorrSelector-Gate**: a FRESH `RetrievalGate` (does NOT load B2's own
  gate checkpoint — this selector's Top-K distribution differs), trained
  per channel on TRAIN, selected on VAL, evaluated once on TEST.

## Sanity checks

`tests/test_exp_correction_selector01.py`, 18/18 PASSED (residual
exactness; choice target == `dense_utility` correction argmax at every
step, recomputed from the CURRENT on-policy prefix; no duplicate picks;
on-policy state uses the model's own pick, never a separate oracle
sequence; no gradient through argmax; gradient reaches only the
trainable modules; scorer never receives `r_i`/`r_q`; full-memory, no
shortlist; runtime equivalence check holds on synthetic data AND is
caught by a negative control; gate positive control recovers a known
γ=0.5; etc.). Full project suite: 581 passed, same 2 pre-existing
failures, no regression.

**Runtime equivalence check** (`MSE(B_q+C,Y_q) == MSE(C,r_q)`, the
mandatory sanity item): asserted after every training epoch on both TRAIN
and VAL, and again during inference on TRAIN/VAL/TEST for every channel.
Max observed error across the entire run: **1.91e-06**, far under the
1e-3 threshold, every time.

## Training

15 epochs (the configured cap), CE loss falling monotonically every
single epoch (train 5.551→4.728, val 5.538→4.886) — **still improving
when the run was cut off at 15 epochs**, no early-stop triggered
(patience=5 was never exhausted). `val_top1_acc` 0.047→0.054,
`val_top10_acc` 0.322→0.360, `val_pred_rank_mean` 242→223 — real but slow
learning of the correction-utility ranking. `wall_clock=18021s` (~5h,
under heavy multi-tenant GPU-1 contention), `peak_gpu_mem=425MiB`.
**Limitation**: since the loss was still decreasing at the cutoff, the
selector below is plausibly undertrained relative to its ceiling — the
comparisons that follow are a lower bound on what this approach could
achieve with more epochs, not a fully-converged result.

## Results

### Aggregate (mean across all 7 channels, test split)

| Arm | MSE | vs. reference |
|---|---:|---:|
| F0 (Track B2, reference) | 0.37312 | — |
| **CorrSelector-Gate** | **0.38299** | vs B2-C1=0.38337: **−0.00038** (slightly better) |
| B2-C1 (Track B2, reference) | 0.38337 | — |
| B2-C0 (Track B2, reference) | 0.39056 | — |
| **CorrSelector-Fixed** | **0.39352** | vs B2-C0=0.39056: **+0.00296** (worse) |

### Per-channel (test split)

| Channel | B0 | F0 | B2-C0 | B2-C1 | CorrSelector-Fixed | CorrSelector-Gate | Fixed<C0? | Gate<C1? | Gate<F0? |
|---|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|
| 0 | 1.44957 | 0.7487 | 0.7557 | 0.7549 | 0.75476 | 0.75366 | ✓ | ✓ | ✗ |
| 1 | 0.32449 | 0.2191 | 0.2416 | 0.2311 | 0.24356 | 0.23090 | ✗ | ✓ | ✗ |
| 2 | 1.53713 | 0.7717 | 0.7870 | 0.7868 | 0.78740 | 0.78628 | ✗ | ✓ | ✗ |
| 3 | 0.27011 | 0.1763 | 0.1923 | 0.1922 | 0.19515 | 0.18882 | ✗ | ✓ | ✗ |
| 4 | 0.73039 | 0.5005 | 0.5224 | 0.5198 | 0.52389 | 0.52149 | ✗ | ✗ | ✗ |
| 5 | 0.15848 | 0.1333 | 0.1542 | 0.1384 | 0.16351 | 0.14137 | ✗ | ✗ | ✗ |
| 6 | 0.05696 | 0.0622 | 0.0806 | 0.0603 | 0.08640 | 0.05840 | ✗ | ✓ | **✓** |

`CorrSelector-Fixed < B2-C0` on **1/7 channels** (channel 0 only, barely).
`CorrSelector-Gate < B2-C1` on **5/7 channels** (0, 1, 2, 3, 6). `CorrSelector-
Gate < F0` on **1/7 channels** (channel 6 only — the same exception B2's own
C1 showed, the lowest-B0-MSE, easiest-to-forecast channel).

### Split by Track B1's Oracle headroom (does gain correlate with headroom?)

| Channel | B1 headroom (FutureOracle−CorrOracle MSE, +=Correction wins) | Gate gain vs B2-C1 |
|---|---:|---:|
| 0 | +0.0291 | +0.00124 |
| 2 | +0.0221 | +0.00052 |
| 4 | +0.0039 | **−0.00169** |
| 1 | +0.0008 | +0.00020 |
| 6 | 0.0000 | +0.00190 |
| 3 | −0.0008 | +0.00338 |
| 5 | −0.0014 | −0.00297 |

**No clean monotonic relationship**: channel 3 (Future Oracle actually won
in B1) shows the LARGEST gate improvement, while channel 4 (real,
moderate Correction Oracle headroom in B1) shows a gate REGRESSION. Larger
B1 headroom does not predict larger realized gain here.

## Answering the 8 required questions

1. **Did the correction-aware selector beat B2's future-oriented selector
   on aggregate quality?** Mixed and marginal. Fixed: no (0.39352 >
   0.39056, worse). Gate: yes but only barely (0.38299 < 0.38337, a
   0.10% relative improvement).
2. **Was fixed-correction MSE < 0.39056?** No, on the aggregate
   (0.39352). Only 1/7 individual channels (channel 0) improved.
3. **Did the learned gate add value over fixed?** Yes, consistently —
   Gate beat Fixed on the aggregate and on every one of the 7 channels
   individually (same pattern B2 found for its own C1 vs C0), confirming
   the gate mechanism itself works; this is not what changed between B2
   and this experiment.
4. **Did the final gated model beat F0=0.37312?** No, on the aggregate
   (0.38299 vs 0.37312) and on 6/7 channels — only channel 6 (already
   the channel where retrieval contributes least) is the exception, the
   same one B2 itself found for C1.
5. **Did channels with larger B1 Oracle headroom show larger learned-
   selector gains?** No — see the table above. The largest-headroom
   channels (0, 2) showed small positive gains, but channel 4 (moderate
   headroom) regressed, and channel 3 (negative headroom — Future Oracle
   actually won in B1) showed the single largest gain. No usable
   correlation.
6. **Did selector regret/rank/NDCG move in the same direction as final
   MSE?** Partially measurable: `val_top1_acc`/`top10_acc`/`pred_rank_mean`
   all improved monotonically through training (see Training section)
   while final MSE also should improve with better checkpoints — but this
   experiment did NOT implement the full B1-style rank/NDCG/overlap-with-
   Oracle diagnostic suite for the trained selector (a scope gap, noted
   under Limitations below), so a direct rank-vs-MSE correlation claim
   cannot be made with the same rigor as Track A's diagnostics.
7. **If it failed, is selection-mismatch or information-insufficiency
   more likely?** Information-insufficiency (H3) is more consistent with
   the evidence than selection-mismatch (H1). H1 predicted
   `CorrSelector-Fixed < B2-C0` on the aggregate; that did NOT happen (1/7
   channels only). The selector's own training diagnostics
   (`val_top1_acc` capped near 5%, `pred_rank_mean` ~223 out of thousands
   of candidates even after 15 epochs of still-improving training) show
   it is only weakly able to predict which candidate has the best
   correction utility from past-only observable features — consistent
   with H3's claim that useful residuals are hard to predict from
   available information, not merely mis-targeted.
8. **Do results justify a future value-aware selector experiment?** The
   evidence leans toward "maybe, but not with confidence yet" — this
   experiment's own H3 reading (information-insufficiency) is exactly the
   condition under which a value-aware selector (adding `r_i`-derived
   features to the scorer input) would plausibly help, since the current
   selector cannot see the one thing (candidate residual pattern) B1's
   headroom analysis implies matters. However, this experiment's selector
   was still improving at the training cutoff (15 epochs) — before adding
   scorer capacity, a cheaper next step would be to confirm this result
   is not simply undertrained by re-running with a longer budget. Per the
   spec's STOP rule, neither is started automatically here.

## Conclusion

**Outcome B3-C**: `CorrSelector-Fixed >= B2-C0` on the cross-channel
aggregate (0.39352 vs 0.39056) — the selection-mismatch hypothesis (H1)
is NOT well supported. The learned gate mechanism itself works (Gate
consistently beats Fixed, replicating B2's own C0-vs-C1 finding), and
produces a small aggregate improvement over B2's own gate (C1) — but this
does not close the gap to F0, and does not correlate with Track B1's own
Oracle-headroom measurements per-channel. Per the STOP rule, no
value-aware selector, residual scorer feature, new scorer/loss, encoder
change, or additional-horizon/dataset run is started automatically here.

## Limitations (explicit, not implemented this round)

- No full B1-style rank/NDCG/overlap-with-Oracle diagnostic suite was
  computed for the TRAINED selector's own on-policy retrieval (only the
  training-time Choice-CE diagnostics and the downstream fixed/gate MSE
  were measured) — required question 6 above is answered only partially
  as a result.
- Training was capped at 15 epochs and had not converged (loss still
  falling every epoch) — the comparisons above are a lower bound on this
  approach's ceiling, not a converged result.
- Single seed (0), ETTh1 H96 only, matching the spec's own scope.

---

Source files: `summary.json`, `channel{0..6}_gate_history.json`,
`command.txt` (below), `sanity checks: tests/test_exp_correction_selector01.py`,
training/inference log: `logs/exp_correction_selector01/run.log`.
