# EXP-ONPOLICY-CHOICE01 — Report

Fills the last cell of Track A's 2×2 table (prefix source × loss). C0/D1/T1
reused verbatim (not retrained). **OPC1 (this experiment, the only new
training) = T1's on-policy prefix + D1's Oracle-Choice CE loss, combined.**
Frozen B0 encoder throughout. Ran in parallel with, independently of,
Track B1/B2.

**Research question:** does the model, learning to directly predict the
greedy Oracle best action at states it actually visits during inference
(on-policy), beat T1 alone and finally cross B0?

---

## Sanity checks (before the GPU run)

`tests/test_exp_onpolicy_choice01.py`, 11/11 PASSED — t=0 state matches
T1's own construction exactly; model-selected candidates (never the
Oracle's) enter the prefix; recomputed on-policy-state utility matches a
direct re-derivation; no duplicate/invalid selection; full-memory
invariant to chunking; the CE target equals the current-state utility
`argmax` exactly; no gradient through the `argmax` trajectory; CE gradient
flows normally to `SetConditioner`/`UtilityHead`; the encoder has no role
in this module at all (frozen, no forward call). Code-review confirmed the
diff vs. `train_onpolicy_prefix01.py` touches only the loss-computation
call site.

## Primary result table

| Arm | Stage-2 MSE | HardAggregate@10 | gap_recovery | t2 selected rank median | t2 continuation regret |
|---|---:|---:|---:|---:|---:|
| B0 | 0.37312 | — | — | — | — |
| C0 (Oracle-prefix + R2) | 0.39526 | 0.5514 | -0.2626 | 1934.0 | 0.3498 |
| D1 (Oracle-prefix + Choice CE) | 0.38916 | 0.5480 | -0.2317 | 88.0 | 0.1964 |
| T1 (On-policy + R2) | 0.37455 | 0.4096 | -0.0100 | 135.5 | 0.2303 |
| **OPC1 (On-policy + Choice CE)** | **0.37340** | **0.4110** | **-0.0091** | **47.0** | **0.1422** |

**OPC1 beats T1 on Stage-2, gap_recovery, t2 selected rank, AND t2
continuation regret — the best combined-metric result of the entire
session.** HardAggregate is very close to T1's (0.4110 vs 0.4096,
essentially tied, a 0.3% relative difference). t1/t2 metrics are also the
best or near-best of every arm tried (t1 oracle rank median: OPC1=44,
better than C0=99, D1=47, T1=144; t2 Spearman: OPC1=0.480, better than
T1=0.409, D1=0.329, C0=0.140; t2 hurt_frac: OPC1=0.200, worse only than
D1's 0.122).

## Training

`best_epoch=4`, early-stopped at epoch 9, `wall_clock=2085.1s` (~35 min),
`peak_gpu_mem=579MiB`, no OOM. `val_top1_acc` reached 5.2% by the final
epoch — notably higher than D1's own oracle-prefix teacher-forced top-1
accuracy (~1.3% at t=1, ~0.36% mean across steps) — the on-policy state's
utility landscape appears to give a SHARPER (less near-tied) choice signal
than the fixed Oracle trajectory's own states did. Full per-epoch
trajectory: `per_step_metrics.csv`, `utility_margin_diagnostics.csv`.

## Answering the 5 required questions

1. **Is OPC1's Stage-2 better than T1's?** Yes — 0.37340 < 0.37455.
2. **Does OPC1 beat B0?** No, not quite — 0.37340 > 0.37312, by only
   0.00028 (0.075% relative). The closest any arm has come to B0 this
   session.
3. **Are D1's and T1's effects combinable?** Partially yes. C0→OPC1
   improved Stage-2 by 0.02186; T1's own gain over C0 was 0.02071 and D1's
   own gain over C0 was 0.00610 — the SUM of the two individual gains
   (0.02681) exceeds OPC1's actual combined gain (0.02186), so the
   combination is positive and real but **sub-additive**, not fully
   additive: combining both fixes helps, but not by the naive sum of their
   separate effects.
4. **Do HardAggregate/regret agree with the Stage-2 direction?** Mostly
   yes — gap_recovery and t2 continuation regret both improve alongside
   Stage-2. HardAggregate is essentially flat vs. T1 (0.4110 vs 0.4096,
   negligibly worse) rather than clearly improving — the one metric that
   does not unambiguously move in the same direction as Stage-2.
5. **If it "failed" (didn't beat B0), is there near-tie/hard-choice-
   instability evidence?** `val_top1_acc` (5.2%) and `val_margin`
   (~0.0197, stable across epochs, not degrading) do not show the kind of
   instability signature D1's own near-tie diagnostic flagged at deeper
   steps — the training signal looks stable throughout. The 0.00028 gap to
   B0 is small enough that this reads as "very close, not clearly blocked
   by instability" rather than a failure requiring a different explanation.

## Case classification

**Case B**: `0.37312 <= MSE_OPC1 (0.37340) < 0.37455`. Per the
pre-registered decision rule: *"T1보다 개선했지만 B0는 못 넘은 것.
positive but insufficient."* This is NOT Case A (a first true downstream
success beating B0) — OPC1 falls 0.00028 short. It is also clearly not
Case C (OPC1 is unambiguously better than T1 on the primary metric, by a
real if modest margin, with t1/t2 improvements far larger than Stage-2's
own small delta would suggest).

## Conclusion

**H1 (loss mismatch and state-distribution mismatch are at least partially
independent, combinable positive interventions) is supported, with a
sub-additivity caveat.** OPC1 is the best combined-metric result this
session has produced and comes closer to B0 than any other arm — but the
combination's Stage-2 gain is smaller than the sum of D1's and T1's
individual gains, indicating diminishing (not negative) returns from
stacking both fixes. **H2 (hard Choice CE is brittle on the on-policy
utility landscape) is NOT supported** — training was stable, `val_top1_acc`
was higher than D1's own oracle-prefix accuracy, and no near-tie-driven
degradation was observed.

Per the pre-registered STOP rule, no further experiment (soft Oracle,
temperature sweep, encoder unfreeze, Track B combination, scheduled
sampling, DAgger, additional loss sweep) is started automatically.

---

Source files: `stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`per_step_metrics.csv`, `utility_margin_diagnostics.csv`, `metrics.csv`,
`train_summary.json`, `checkpoint_fingerprints.txt`, `working_tree.diff`,
`sanity_summary.json`, `command.txt`, `config.json`.
