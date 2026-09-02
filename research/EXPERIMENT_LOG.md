# EXPERIMENT_LOG.md

Append-only, factual record of completed experiments. Newest entries go at the
bottom. Keep this file primarily factual; interpretation belongs to the reviewer.

Pre-existing results produced before this log was created are recorded in
`RESULTS_SUMMARY.md` (aggregate tables, 2026-08-04) and under `metrics/` and
`logs/`. They are **not** retro-fitted into this log, to avoid fabricating
history.

---

## Template

Copy this block for each completed experiment.

```markdown
## <EXP-ID, e.g. EXP-0001-short-slug>

### Date
YYYY-MM-DD

### Research Question

### Configuration
Full command line(s), or the driver script path plus any overrides.

### Changed Variable

### Controlled Variables

### Dataset

### Prediction Horizon

### Seed

### Important Hyperparameters

### Result Files
Paths under `results/<EXP-ID>/`, plus the `metrics/` and `logs/` paths.

### Results
Exact numbers, with the baseline number alongside each.

### Sanity Checks
What was run before the expensive job, and the outcome.

### Implementation Notes
What changed in the code; files touched.

### Status
completed | failed | aborted
```

---

<!-- Append experiment entries below this line. -->

## EXP-C01 — 2026-09 retrieval-transfer diagnostic campaign

### Date
2026-08-30 → 2026-09-02 (artifact timestamps span 08-29 22:20 → 09-02 04:05)

### Research Question
Why does improving Stage-1 retrieval (Recall@10) not improve Stage-2 forecasting?

### Configuration
Multiple probes, each in its own log directory. Common setting where stated:
ETTh1, `features=M`, `seq_len = pred_len`, `top_k=10`, `relation_top_n=3`,
`d_model=128 / n_heads=4 / e_layers=2 / d_ff=256 / label_len=0`.
Frozen-encoder probes use a healthy WCE-trained Stage-1 encoder and a fixed
per-query candidate pool P100.

### Changed Variable
Varies per sub-probe — see the per-phase table below. No single controlled
variable; this is a diagnostic campaign, not a single comparison.

### Controlled Variables
Dataset, split protocol, `seq_len = pred_len`, `top_k`, `relation_top_n`, model
size. Oracle-selection probes (Phase A, set oracle) hold the candidate pool fixed
so that only the *selection rule* varies.

### Dataset
ETTh1 (primary); Weather (Stage-2 wiring-fixed verification only).

### Prediction Horizon
96 / 192 / 336 / 720 (Phase C imitation: 96 and 720 only).

### Seed
Not recorded per-probe in the logs. **UNKNOWN** — `--seed` default is 0; this was
not confirmed for these runs.

### Important Hyperparameters
- Phase 0 calibrated temperatures: τ = 0.015 (H96), 0.015 (H192), 0.01 (H336),
  0.02 (H720).
- Imitation probes: fixed pool P100, 3 epochs, LR decayed to 2.5e-4 by epoch 3.
- Candidate memory bank at H96: shape `(7, 1, 8449, 128)`, valid candidate pool
  8449.

### Result Files

| Probe | Directory | Key files |
|---|---|---|
| Score geometry | `logs/score_geometry/` | `ETTh1_pred{96,720}.log` |
| Boundary rank arms | `logs/rank_arms/` | `ETTh1_pred{96,720}_wce_rank.log` |
| Persistent pair mining | `logs/persistent_probe/` | `Dynamic.log`, `Persistent.log` |
| Collapse onset | `logs/collapse_diag/` | `WCE.log`, `Dynamic.log`, `Persistent.log` |
| Frozen encoder + rank isolation | `logs/frozen_scorer/` | `A_cosine.log`, `B_asym_rank.log` |
| Global anchor KL | `logs/global_anchor/` | `beta0p1.log`, `beta1p0.log` |
| Swap / conflict | `logs/swap_conflict/` | `swap_rows.csv`, `fingerprints.txt` |
| Oracle headroom + redundancy | `logs/headroom_redundancy/` | 40 × `ETTh1_pred<H>_<metric>_<loss>.log` |
| Retrieval-off counterfactual | `logs/utilization_diag/` | 40 × same scheme |
| Set-utility oracle (raw) | `logs/set_oracle/` | `set_only_candidates_pred{96,192,336,720}.csv` |
| **Phase 0** — τ calibration | `logs/tau_calibration/` | `pred{96,192,336,720}.log` |
| **Phase A** — pool × K sweep | `logs/set_oracle/` | `pool_k_sweep_pred{96,192,336,720}.csv` |
| **Phase C** — imitation gate | `logs/imitation/` | `pred{96,720}_{individual,set}.log` |
| Stage-2 final MSE | `logs/stage2_learned_score_corrected_selection/` | `ETTh1/pred<H>/<arm>_{stage2,e2e}.log` |
| Weather Stage-2 | `logs/weather_stage2/`, `logs/weather_top3/` | `weather/pred<H>/` |

Phases B, D, E, F of the stated 7-phase plan: **UNKNOWN** — not named, no
directory identified.

### Results

Full numeric tables are in `research/RESEARCH_CONTEXT.md` →
*Established Experimental Findings*, section B (B1–B17), with per-block
provenance marking. Headline factual results:

1. Stage-1 R@10 is improvable ~2–4× by asymmetric / pair-MLP scorers, most at
   H336/H720 (0.0489 → 0.1311 at H336).
2. Weather H96 post-fix: R@10 0.0583 → 0.0044 (−92.5%) while Stage-2 MSE
   0.1925 → 0.1794 (−6.8%). Directions disagree.
3. Retrieval gain is nonzero and horizon-dependent (ETTh1 H96 ≈40.05%,
   H720 ≈6.18%), so "Stage-2 ignores retrieval" is rejected.
4. Spearman vs Stage-2 MSE: Recall@10 +0.032 (ETTh1) / −0.350 (Weather);
   HardAggregateMSE@10 +0.810 / +0.935.
5. Rank-only encoder training collapses the representation
   (effective rank 16.38 → 1.07 within one epoch).
6. With the encoder frozen, rank training improves local pair ordering
   (PairAcc100 0.53185 → 0.54605) while global R@10 falls 0.05775 → 0.02046.
7. Global KL anchor raises Top-10 retention 0.142 → 0.716 with β, but
   `cos(g_rank, g_global) < 0` and only ≈3.7% of anchor gradient reaches Top-10.
8. Swap analysis: added candidates were *better* individually
   (0.59782 vs removed 0.73979), yet candidate variance fell up to 74%.
9. `I = A + V` verified exactly (residual ≤ 3e-9 across all `pool_k_sweep` rows).
10. Set Oracle vs Individual Oracle aggregate gain, recomputed here from raw
    CSVs: **25.9% / 18.9% / 12.3% / 4.0%** at H96/192/336/720, with
    `A_set < A_ind AND I_set ≥ I_ind` holding for **100.0%** of query units.
11. Good+Diverse Oracle loses to Individual Oracle at every horizon
    (−6.8% / −1.8% / −6.8% / −3.8%).
12. Phase 0 τ calibration complete; per-horizon τ as listed above.
13. Phase C: every imitation arm sits at random/uniform on validation
    (TeacherSetRecall@10 0.0847–0.1363 vs random 0.10; imitation loss
    4.589–4.621 vs ln(100) ≈ 4.605) — for the **Individual** target as well as
    the Set target.
14. Greedy Set Oracle restart overlap 0.878 (H96) / 0.902 (H720) → target-noise
    explanation rejected.

### Sanity Checks
- `pytest tests/` at the time of logging: `396 passed, 2 failed`; both failures
  pre-existing at HEAD `c306def` (see *Known Repository Issues*).
- `I = A + V` identity verified numerically on every `pool_k_sweep` row.
- Phase C reference baselines pinned in advance: random Top-10 overlap in P100 =
  0.10, uniform CE = ln(100) ≈ 4.605.
- Greedy Set Oracle restart stability measured before treating it as a target.

### Implementation Notes
Implemented by the uncommitted working-tree changes inventoried in
`RESEARCH_CONTEXT.md` → *Uncommitted Work In Progress* (boundary rank loss,
rank-failure diagnostics, collapse/score geometry, frozen-encoder scorer, global
anchor, set-level loss, set-utility oracle selectors, retrieval-off
counterfactual, Stage-2 selection/redundancy reporting).

**A wiring bug was found and fixed during this campaign:** the configured scorer
was not propagated into Stage-2's actual Top-K selection path, so some runs
selected via the cosine path regardless of the trained scorer. Weather was
re-verified post-fix; the pre-fix **ETTh1 Stage-2 sweep is invalidated** and
needs a rerun. See *Invalidated Results — Do Not Cite*.

### Verification Discrepancies
Recomputation from the artifacts did not reproduce three user-supplied figures.
Same conclusions; different numbers. Flagged for the reviewer:

| Quantity | User-supplied | Recomputed / logged | Source |
|---|---|---|---|
| H336 set-oracle aggregate gain | 11.8% | **12.3%** | `set_only_candidates_pred336.csv` |
| H720 set-oracle aggregate gain | 3.6% | **4.0%** | `set_only_candidates_pred720.csv` |
| Joint-condition fraction | 99.9 / 100 / 97.8 / 95.5% | **100.0%** at all four | recomputed, strict `<` and `≥` |
| H336 N_eff at τ=0.015 | ≈32 | **93.6** (N_eff ≈ 35.9 occurs at τ=0.01) | `tau_calibration/pred336.log` |
| H336 N_eff / Mass@10 at τ=0.1 | ≈1632 / ≈0.065 | **2444.9 / 0.0202** (τ=0.07 gives 1625.1 / 0.0385) | same |
| H96 cosine I / A / V | 0.6612 / 0.3573 / 0.3038 | **0.594556 / 0.318727 / 0.275829** | `pool_k_sweep_pred96.csv`, pool=100 K=10 |

The chosen τ values (0.015 / 0.015 / 0.01 / 0.02) match the logs exactly.
The Phase C imitation numbers match the logs exactly.

### Status
completed (diagnostic campaign); **ETTh1 Stage-2 sweep requires rerun**

