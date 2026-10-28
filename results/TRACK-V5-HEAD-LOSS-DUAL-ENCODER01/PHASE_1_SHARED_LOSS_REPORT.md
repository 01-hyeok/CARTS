# PHASE 1 -- Shared Encoder Loss Comparison (S0/S1/S2)

**Status: ETTh1_96 complete. ETTh1_720/Weather_96/Weather_720 pending (this track is still in progress; see FINAL note).** seed0, `lambda_A=1.0` (fixed, per the gradient probe in PHASE 0 -- no coefficient sweep), 10 epochs fixed (no early stopping), GPU1 only, strictly sequential.

Canonical `S0`-baseline parity (vs `TRACK-V5-HEAD-OBJECTIVE-AUDIT01`'s own Mean-Mixture test numbers for ETTh1_96: retMSE@10=0.66386, aggMSE@10=0.40981): S0 reproduces **0.663855 / 0.409808** -- matches to 4+ significant figures, confirming this track's new training script is a correct re-implementation, not a new or different training procedure.

## ETTh1_96 (test split, primary checkpoint = min val RetMSE@10)

| Arm | Loss | RetMSE@10 | Hard AggMSE@10 | D | C | Recall@10 | NDCG@10 | Soft AggMSE | Best epoch | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | L0 | 0.663855 | 0.409808 | 0.066386 | 0.343422 | n/a* | n/a* | 0.399121 | 10 | 158.5 |
| S1 | L3a | 0.669296 | 0.411558 | 0.066930 | 0.344628 | n/a* | n/a* | 0.398719 | 8 | 166.9 |
| S2 | L3b | 0.672601 | 0.412088 | 0.067260 | 0.344828 | n/a* | n/a* | 0.403409 | 7 | 171.9 |

`*` Recall@10/NDCG@10 were computed during training-time validation diagnostics but not re-extracted into this summary table from the raw per-epoch CSVs this round (available in `results/.../phase1/ETTh1_96/<arm>/val_metrics.csv`, not fabricated here).

## RQ1/RQ2 answer (ETTh1_96 only so far)

**No** -- at this fixed `lambda_A=1.0`, seed0: adding the Full-P100 Soft Aggregate loss (S1) or the Top-K-aligned Aggregate loss (S2) makes Hard AggMSE@10 WORSE than plain L0 (S0), not better (0.409808 -> 0.411558 -> 0.412088, monotonically worse as the aggregate term is added and as the aggregate relaxation gets tighter/more top-k-aligned). L3b (S2) is not better than L3a (S1) at matching Hard AggMSE on this cell -- if anything slightly worse, though both are close. **D increases monotonically alongside the aggregate loss (0.066386 -> 0.066930 -> 0.067260) while C stays roughly flat (0.343-0.345)** -- the degradation is coming from the individual-candidate-relevance term (D), not from a cross-interaction (C) regression; the aggregate loss is NOT succeeding at trading individual relevance for better set-level complementarity here, it is simply making both slightly worse together.

## Soft/Hard gap, this cell

| Arm | Soft AggMSE | Hard AggMSE | Gap |
|---|---:|---:|---:|
| S0 | 0.399121 | 0.409808 | soft better by 2.61% (identical to the `TRACK-V5-HEAD-OBJECTIVE-AUDIT01` PHASE G number for canonical V5, confirming consistency) |
| S1 | 0.398719 | 0.411558 | soft better by 3.12% |
| S2 | 0.403409 | 0.412088 | soft better by 2.11% |

Training against L3a (S1) very slightly improved its OWN soft AggMSE (0.398719 vs S0's 0.399121) but made the actual deployed Hard AggMSE worse -- the soft/hard gap did not close by training against an aggregate-aligned signal, on this cell. L3b (S2) shows the opposite pattern on this cell (worse soft AggMSE, but the smallest soft/hard percentage gap of the three) -- not a clean win for either relaxation.

## ETTh1_720 (test split, primary checkpoint = min val RetMSE@10)

| Arm | Loss | RetMSE@10 | Hard AggMSE@10 | D | C | Soft AggMSE | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | L0 | 0.942633 | 0.620570 | 0.094263 | 0.526307 | 0.579579 | 9 |
| S1 | L3a | 0.951641 | 0.626839 | 0.095164 | 0.531675 | 0.581932 | 9 |
| S2 | L3b | 0.948144 | **0.614959** | 0.094814 | 0.520145 | 0.562827 | 6 |

S0 reproduces canonical V5 ETTh1_720 (retMSE@10=0.94263, aggMSE@10=0.62057) essentially exactly (0.942633/0.620570).

**Reversal from ETTh1_96**: on this horizon, **S2 (L3b) actually BEATS S0** on Hard AggMSE (0.614959 vs 0.620570, a genuine ~1% improvement) -- the first case in this track where an aggregate-aligned loss helps within the Shared structure. S1 (L3a) is still worse than S0, as on ETTh1_96. So within Shared, the two aggregate losses no longer move together across horizons: L3a is consistently non-helpful (both cells), but L3b flips from slightly-harmful (H96) to genuinely-helpful (H720).

## Weather_96 (test split, primary checkpoint = min val RetMSE@10)

| Arm | Loss | RetMSE@10 | Hard AggMSE@10 | Best epoch |
|---|---|---:|---:|---:|
| S0 | L0 | 0.455618 | 0.264330 | 4 |
| S1 | L3a | 0.538645 | 0.293258 | 9 |
| S2 | L3b | **0.330478** | **0.204646** | 5 |

**A third, distinct pattern -- L3b wins decisively.** Unlike either ETTh1 cell (where L3a/L3b were close to L0, sometimes slightly worse, sometimes slightly better by ~1%), on Weather_96 **S2 (L3b) beats S0 by 27.5% on RetMSE@10 and 22.6% on Hard AggMSE@10** -- the largest effect size of any Loss comparison in this track so far, in either direction. S1 (L3a) is clearly the worst arm (RetMSE +18.2% vs S0), the opposite of S2's direction -- so the two aggregate losses no longer move together at all on this cell: L3a actively hurts, L3b strongly helps.

**(Weather_720 pending -- see FINAL_RESEARCH_REPORT.md's status section.)**
