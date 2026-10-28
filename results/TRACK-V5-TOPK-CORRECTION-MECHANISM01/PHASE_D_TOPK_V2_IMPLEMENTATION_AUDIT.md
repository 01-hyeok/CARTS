# PHASE D -- Differentiable Top-K v2 Implementation Audit

## D1/D2. The v1 bug and the v2 fix

`utils/differentiable_topk01.py` (v1, PRESERVED UNMODIFIED) sets `threshold = s_(K)` (the K-th largest
score itself, detached). For the K-th-ranked candidate, `s_i - threshold == 0` identically, so
`sigmoid(0) = 0.5` **regardless of `tau`** -- v1's own docstring claims exact convergence to uniform `1/K`
weights, but the K-th candidate is permanently stuck at half the correct weight relative to the other
`K-1` selected candidates, even as `tau -> 0`. Directly demonstrated by this phase's new
`test_v2_fixes_v1_boundary_bug_directly`: with `tau=1e-5` and a strictly-ordered, non-tied score vector,
v1's K-th candidate carries `|weight - 1/K| > 0.05` while v2's carries `|weight - 1/K| < 1e-3`.

`utils/differentiable_topk02.py` (v2, NEW) fixes this with `threshold = (s_(K) + s_(K+1)) / 2` -- the
midpoint between the K-th and (K+1)-th largest scores. Whenever `s_(K) != s_(K+1)` (a real gap), every
top-K candidate's `s_i - threshold > 0` and every other candidate's `< 0`, so `tau -> 0` saturates to the
TRUE uniform `1/K` limit.

## Tie-breaking (not addressed by v1's docstring at all)

A pure midpoint fix does not resolve an EXACT tie at the `K`/`K+1` boundary -- the same "stuck at 0.5"
failure recurs for every tied candidate. v2 breaks ties the same way the repo's own hard selector
(`models.RelationStage1.stable_topk_indices`) already does (lower original index wins), via an
infinitesimal per-index perturbation (`TIE_EPS = 1e-9`) applied consistently to BOTH the threshold-ranking
step and the actual sigmoid numerator.

**Two implementation pitfalls found and fixed during this phase, both only visible once real synthetic
tests with exact ties were run (confirming the spec's own warning that v1's loose `5e-2` test tolerance was
insufficient to catch issues like this):**

1. **Premature float32 rounding.** An earlier version cast the averaged threshold back to the input's dtype
   immediately after computing it. When the input is float32 (the real training dtype), that cast rounds
   away the entire `TIE_EPS`-scale separation (float32's ULP near O(1) values, ~1.2e-7, is ~100x larger than
   `TIE_EPS * N` for any realistic `N`), silently resurrecting the exact bug v2 exists to fix, for any exact
   tie. Fixed by keeping the ENTIRE forward computation in float64 end-to-end, casting only the final
   output weights back to the original dtype (safe, since those are plain probabilities with no
   tie-breaking information left to lose).
2. **Perturbing only the threshold, not the numerator.** The first fix for (1) still applied `TIE_EPS` only
   to the values used to RANK and select `s_(K)`/`s_(K+1)`, while the actual per-candidate sigmoid
   numerator still used the original (unperturbed) score. For two genuinely tied candidates, this meant
   their final sigmoid ARGUMENTS were computed from identical original scores minus the same shared
   threshold -- i.e., bit-identical outputs regardless of `tau`, no matter how the tie-break decided which
   one was "labeled" `s_(K)` vs `s_(K+1)`. Fixed by applying the SAME perturbation to the numerator too
   (`d(perturbed_i)/d(score_i) == 1`, a fixed additive constant per index, so gradients are completely
   unaffected and the shift is `<= 1e-9 * N`, utterly negligible for any non-tied comparison).

**Threshold gradient: detached**, same policy as v1, for the same reason -- `s_(K)`/`s_(K+1)` come from a
hard order-statistic selection whose derivative has a discontinuity exactly at every rank-swap; detaching
keeps the surrogate gradient smooth everywhere, with the only gradient path running through
`d(sigmoid)/d(score)` at a fixed threshold.

**Degenerate case** (`n_valid <= K`): threshold set to `-inf`, giving every valid candidate equal weight
`1/n_valid` -- the only sensible behavior when there are not even K candidates to choose among.

## D3. The 15 required synthetic tests (+ 1 bonus) -- all pass, strict tolerances

`tests/test_differentiable_topk02.py`, 16/16 pass:

1. K=1 -- exact single-argmax selection in the hard limit.
2. K=5 -- uniform `1/5` weight convergence.
3. K=10 -- uniform `1/10` weight convergence (the repo's actual production `K`).
4. K=N-1 -- the single excluded candidate correctly gets ~0 weight.
5. All scores identical -- a GLOBAL tie across every candidate; tie-break (lower index wins) verified to
   match `stable_topk_indices` exactly; requires `tau=1e-11` to see saturation (the tie-break gap here is
   only `TIE_EPS=1e-9`, so `tau` must shrink several orders of magnitude below that to saturate -- this is
   expected and documented, not a bug: convergence for an EXACT tie is asymptotic in `tau`, same honest
   caveat as near-tied distinct scores, just against a much smaller effective gap).
6. K-th/(K+1)-th exactly tied (boundary-only tie) -- monotonic error decrease verified across a
   `tau: 1e-2 -> 1e-12` sweep, strict `atol=1e-2` at the smallest `tau`.
7. Very small (but nonzero) real score gap at the boundary -- converges cleanly at `tau=1e-7`, `atol=1e-2`.
8. Invalid candidates -- exactly zero weight verified (`torch.equal`, not `allclose`), even when given an
   artificially huge score.
9. Extreme score magnitudes (`1e8`, `-1e8`, `1e-8`, ...) -- no NaN/Inf at any of 3 tested `tau` values.
10. Convergence as `tau` shrinks -- monotonic error decrease over a 6-point sweep, strict `atol=1e-4` at
    the end.
11. PyTorch autograd gradient exists, finite, nonzero.
12. Finite-difference gradient matches autograd to `atol=1e-4` -- constructed so the probed index sits
    deep inside the Top-K region (far from any rank-boundary discontinuity a central difference could
    straddle), since the threshold genuinely is a non-smooth function of the OTHER scores at a rank swap
    even though it is detached from this probe's own gradient.
13. Gradient contains no NaN/Inf even with invalid candidates in the row.
14. Hard Top-K selection consistency with `stable_topk_indices` -- v2's `tau -> 0` limit selects the
    IDENTICAL index set the repo's actual hard inference uses.
15. Top-K uniform weight convergence repeated across 20 independent random seeds (not just one lucky draw).
16. (Bonus) Direct v1-vs-v2 comparison at the exact boundary bug case -- see D1 above.

## D4. Real-data validation (all 4 primary cells, existing S0/S2 checkpoints, primary `tau_topk=0.05`)

Script: `scripts/analyze_topk_v1_v2_real_data01.py` (new, read-only + isolated gradient probe, no
checkpoints written). For each cell: 3 test batches, channel 0, real trained Mean-Mixture `log(p_mix)`
scores fed into both v1 and v2 at the training-default `tau=0.05` (chosen per spec D5 -- NOT swept/tuned
after seeing results).

| Cell | Mean Top-10 margin | Mean K/(K+1) gap | Min K/(K+1) gap (any query) | v1 mass in hard-Top10 | v2 mass in hard-Top10 | Pred L2 diff | Grad cosine (v1,v2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.4477 | 0.027960 | 0.0000484 | 0.8693 | 0.8522 | 0.0904 | 0.9880 |
| ETTh1_720 | 1.1794 | 0.072652 | 0.0021074 | 0.9541 | 0.9354 | 0.4834 | 0.9170 |
| Weather_96 | 0.0636 | 0.005449 | 0.0000343 | 0.3912 | 0.3850 | 0.0094 | 0.9976 |
| Weather_720 | 0.1214 | 0.010487 | 0.0001669 | 0.5978 | 0.5888 | 0.0948 | 0.9981 |

**Key finding: the minimum observed K/(K+1) gap in every cell is 2-4 orders of magnitude SMALLER than the
training `tau=0.05`.** This means v1's exact bug (the K-th candidate stuck near 0.5-of-correct-weight) is
not a rare, theoretical edge case on this data -- for a meaningful fraction of real queries, the actual
K/(K+1) score gap is small enough relative to `tau` that the K-th candidate's weight is genuinely ambiguous,
and v1's specific mishandling of that ambiguity (anchoring the threshold to `s_(K)` itself rather than the
midpoint) measurably differs from v2's.

**v1 consistently assigns slightly MORE normalized weight to the hard-selected Top-10 set than v2 does**
(by roughly 1-2 percentage points in every cell). This is a real, structural (not numerical-noise)
difference: v2's threshold is always `<= ` v1's (the midpoint is never above `s_(K)`), which gives
candidates ranked just OUTSIDE the Top-10 (rank `K+1`, `K+2`, ...) a less-negative sigmoid argument under v2
than under v1 -- after renormalization, this very slightly dilutes v2's mass concentration inside the exact
hard-selected set relative to v1. **This is not evidence that v2 is "worse"** -- it reflects v2 honestly
distributing some weight to near-boundary candidates whose true rank is genuinely close, which is the
entire point of a differentiable relaxation (giving SOME gradient signal to near-miss candidates); v1's
higher concentration is partly an artifact of its threshold happening to sit exactly at `s_(K)`, not a sign
of better selectivity.

**Gradient cosine between v1 and v2 is high (0.92-0.998) but not 1.0 in any cell** -- the two operators
produce correlated but not identical training signals. ETTh1_720 shows the largest divergence (0.917),
consistent with it also having the largest absolute prediction difference (0.483, on a horizon-720
multi-step forecast where prediction vectors have naturally larger norms).

## D5. Temperature

All D4 numbers above use `tau_topk=0.05`, matching v1's own training default, unchanged -- no coefficient
or temperature sweep was used to select this value after seeing results, per spec section D5's explicit
instruction.

## Verdict: v2 is ready for Phase E

`IMPLEMENTATION_PASS` for v2: the fix is mathematically justified, passes all 15 required tests plus the
direct bug-reproduction demo, and the gradient path is confirmed finite/correct via both autograd
consistency and independent finite-difference checks. The real-data audit (D4) shows v1 and v2 are
meaningfully but not drastically different at the actual training temperature -- different enough
(1-2pp mass difference, grad cosine as low as 0.92) to be worth testing empirically in Phase E, not so
different that a dramatically different Phase E outcome should be assumed in advance.
