# EXP-CORRECTION-ORACLE-DIAG01 — Report (Track B, diagnostic only, no training)

ETTh1, H96, seed 0, top_k=10, channel 0 (self-only). **No training of any
kind** — reuses the existing B0 Stage-2 checkpoint (frozen `base_head` +
retrieval encoder) verbatim. Ran independently of, and in parallel with,
Track A (`EXP-ENCODER-ANCHOR01` → `EXP-ORACLE-CHOICE01` →
`EXP-TEACHER-FORCING-DIAG01` → `EXP-ONPOLICY-PREFIX01`); touches none of
Track A's code, checkpoints, or results.

**Research question:** is retrieving candidates whose realized future
resembles the query's future (the existing Future Set Oracle) the right
target, given that Stage-2's actual fusion is `Y_final = B_q + gamma *
Y_ret` (a residual/correction, not a from-scratch reconstruction) — or
should retrieval instead target candidates whose OWN frozen-B0 forecast
ERROR resembles the query's own error (a Correction Set Oracle)? And is
the Correction target more learnable from a past-only score?

**Scope limitation (documented, affects only the absolute MSE values
below, not the relative comparison or the rank/margin/NDCG diagnostics)**:
this diagnostic evaluates channel 0 only. Its downstream MSE numbers are
this channel's own error, computed directly (not through the full
multi-channel Stage-2 forward pass) — **not comparable in absolute terms**
to Track A's 0.37312-scale numbers, which average over all 7 channels.
Only the relative {B0, Future Oracle, Correction Oracle} comparison within
this diagnostic (same scope for all three) is used for the conclusions
below.

---

## Primary comparison table

| Metric | B0 | Future Set Oracle | Correction Set Oracle |
|---|---:|---:|---:|
| Final MSE ↓ (channel-0 scope) | 1.4496 | 0.2109 | **0.1818** |
| Gain vs B0 ↑ | — | 1.2387 | **1.2678** |
| Top1–Top2 margin (abs) ↑ | - | 0.00838 | 0.00573 |
| near-tie @1% ↓ | - | 14.5%\* | 20.0%\* |
| Oracle mean rank ↓ | - | 757.8 | **632.1** |
| Oracle median rank ↓ | - | see per-step CSV | see per-step CSV |
| NDCG@10 ↑ | - | 0.524 | 0.513 |
| ε-optimal@1% ↑ | - | 1.0%\* | 0.5%\* |

\*shown at t=2 (a representative early step; near-tie/ε-optimal both
degrade sharply with t for both oracles — see `utility_margin_diagnostics.csv`/
`epsilon_optimal_metrics.csv` for the full per-step trajectory, t=1..10).

## Sanity checks (before the GPU run)

`tests/test_exp_correction_oracle01.py`, 9/9 PASSED (all spec-numbered
items: residual exactness, `MSE(B_q+C,Y_q)==MSE(C,r_q)` equivalence, empty
correction reduces to B0 exactly, K=1 brute-force match, small-N
exhaustive stepwise match, no invalid/duplicate selection, full-memory
count invariant to chunking, no B0 gradient, candidate residual
independent of the query). Full list: `sanity_checks.txt`. The
`MSE(B_q+C,Y_q)==MSE(C,r_q)` equivalence was ALSO asserted at runtime on
every real batch (not just the unit test) — held exactly (atol 1e-4)
throughout the actual evaluation run. An additional online cross-check —
`argmax` of a `dense_utility`-derived per-candidate utility must match
`select_greedy_weighted_set`'s own independently-computed picks at every
step — matched with 0% mismatch at t=1–4 and ≤4.5% at t=10 (floating-
point/near-tie boundary noise at the flattest part of the utility
landscape, not a correctness concern).

Two implementation bugs were caught and fixed before trusting results:
(1) a dict-indexing bug where per-step diagnostic rows were looked up on
the wrong (outer, step-indexed) dictionary; (2) Future Oracle and
Correction Oracle pick DIFFERENT candidates at each step in general, so
sharing a single "already selected" mask between them (as an early draft
did) incorrectly treated one oracle's own picks as unavailable to the
other — fixed by giving each oracle its own independent selected-mask
trajectory, matching how `select_greedy_weighted_set` itself already
treats them independently.

## Results

**Downstream headroom**: the Correction Set Oracle achieves a lower final
MSE than the Future Set Oracle within this diagnostic's channel-0 scope
(0.1818 vs. 0.2109, a further ~13.8% relative reduction beyond Future
Oracle's own already-large gain over B0). Both oracles are, as expected
for diagnostic upper bounds using the true query future, dramatically
better than B0 alone (B0=1.4496).

**Oracle target geometry / separability**: mixed. Correction's mean Oracle
rank under the SAME past-only reference score is better (632.1 vs.
757.8 — lower is better), but its NDCG@10 (0.513 vs. 0.524), Top-1–Top-2
margin (0.0057 vs. 0.0084), and Top-10/Top-50 containment are all
slightly worse than Future's. Near-tie fraction is HIGHER for Correction
at every step measured (e.g. 20.0% vs. 14.5% at t=2) — the Correction
target's true top-1/top-2 utility values are closer together than
Future's, a genuine (if modest) LEARNABILITY headwind.

**ε-optimality**: very low for BOTH oracles at every threshold tested
(0.1%/1%/5%) and every step — the past-only reference score's own top pick
is rarely within even a 5% relative gap of either oracle's true best pick.
This argues against a "near-ties make the reference score's imperfect
choice practically fine" reading for either value space — the reference
score does not reliably land near-optimal for either target.

**Oracle trajectory overlap**: Future and Correction Oracles select
notably different candidate sets — set overlap@k stays in the 22–28%
range across all 10 steps (`oracle_overlap.csv`), confirming these are
genuinely different retrieval targets, not two labels for the same
underlying selection.

## Answering the spec's five required questions

1. **Did the Correction Oracle increase downstream headroom?** Yes, modestly
   (1.2678 vs. 1.2387 gain over B0, channel-0 scope) — a real but not
   dramatic improvement.
2. **Does it select a genuinely different set from the Future Oracle?**
   Yes — set overlap stays around 22–28% throughout, far from identical.
3. **Is Correction utility more separated than Future utility?** No —
   near-tie fraction is consistently HIGHER for Correction at every step,
   and its Top-1–Top-2 margin is smaller. Correction utility is LESS
   separated, not more.
4. **Does the Correction target look more learnable from a past-only
   score?** Mixed, leaning negative: mean rank is somewhat better, but
   NDCG, containment, and margin are all slightly worse, and near-tie
   fraction is worse. Not a clear "more learnable" signal.
5. **Is there enough evidence to justify a learned Correction Selector
   next?** Not on this diagnostic's evidence alone — the downstream MSE
   gain is real but modest, and the accompanying learnability signals are
   mixed-to-negative, not the "same-or-better + lower final MSE" pattern
   the spec's Outcome B-A would require.

## Outcome classification

**Outcome B-B — Oracle MSE improves, but learnability signals are mixed
to worse, not clearly better.** Per the spec's own Outcome B-B
interpretation: *"Correction target은 downstream 목적에는 좋지만 현재
past-only representation만으로는 더 어려운 target일 수 있다"* — this
matches the evidence closely: real (if modest) downstream headroom gain,
alongside a consistently higher near-tie fraction and smaller margin. This
is NOT strong enough evidence, on its own, for Outcome B-A's clean
"proceed to `EXP-CORRECTION-SELECTOR01`" recommendation, nor does it match
B-C (no headroom improvement) or B-D (oracles nearly identical — ruled out
by the 22–28% overlap and the real MSE gap).

Per the spec's own guidance for this outcome: this is read as **weak,
mixed evidence that a future value-aware selector (adding `v_i = g(r_i)`
as an observable candidate feature, not just changing the retrieval
TARGET) might be worth considering** — explicitly NOT justified by this
diagnostic alone to start automatically, and explicitly NOT falsified by
`EXP-STRONG-SCORER-DIAG01`'s earlier negative scorer-capacity result (per
the spec's own instruction, since that experiment gave the scorer more
CAPACITY over the SAME observable information, not new observable
information).

---

Source files: `summary.json`, `future_oracle_metrics.csv`,
`correction_oracle_metrics.csv`, `utility_margin_diagnostics.csv`,
`epsilon_optimal_metrics.csv`, `oracle_overlap.csv`, `per_step_metrics.csv`,
`sanity_checks.txt`, `command.txt`, `env.txt`, `checkpoint_fingerprints.txt`,
`working_tree.diff`, `logs/`.

**STOP per spec.** No selector, encoder, residual encoder, or Stage-2 gate
was trained. `EXP-CORRECTION-SELECTOR01`'s execution is not decided here.
