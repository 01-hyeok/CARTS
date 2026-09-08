# EXP-CORRECTION-ORACLE-DIAG02 (Track B1) — Report

No training. Reuses the existing B0 Stage-2 checkpoint verbatim.
`diag_correction_oracle01.evaluate()` called once per channel (its own
9/9 sanity checks reused unmodified), aggregated across all 7 ETTh1
channels. Ran in parallel with, independently of, Track A
(`EXP-ONPOLICY-CHOICE01`) and Track B2 (`EXP-CORRECTION-STAGE2-SEMANTICS01`).

**Question**: does the channel-0-only Correction Set Oracle advantage
found in `EXP-CORRECTION-ORACLE-DIAG01` reproduce across the full ETTh1
multivariate setting?

## Per-channel results

| channel | B0 MSE | Future Oracle MSE | Correction Oracle MSE | Correction beats Future | Future rank | Correction rank | Future NDCG@10 | Correction NDCG@10 |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|
| 0 | 1.4496 | 0.2109 | **0.1818** | ✓ | 757.8 | 632.1 | 0.524 | 0.513 |
| 1 | 0.3245 | 0.0778 | **0.0770** | ✓ | 751.7 | 2172.7 | 0.572 | **0.660** |
| 2 | 1.5371 | 0.1969 | **0.1748** | ✓ | 990.6 | 703.9 | 0.519 | 0.516 |
| 3 | 0.2701 | **0.0657** | 0.0665 | ✗ | 839.3 | 3218.6 | 0.564 | **0.721** |
| 4 | 0.7304 | 0.1295 | **0.1256** | ✓ | 612.5 | 1013.3 | 0.610 | **0.666** |
| 5 | 0.1585 | **0.0372** | 0.0386 | ✗ | 2776.3 | 3427.6 | 0.924 | **0.931** |
| 6 | 0.0570 | **0.00324** | 0.00324 | ✗ (~tied) | 3820.3 | 5022.3 | 0.879 | **0.884** |

## Aggregate (mean across all 7 channels)

| Metric | Future Oracle | Correction Oracle |
|---|---:|---:|
| Gain vs B0 (mean) | 0.5437 | **0.5514** |
| Mean rank (mean) | 1506.9 | 2312.9 |
| NDCG@10 (mean) | 0.656 | **0.699** |
| Margin abs (mean) | 0.00404 | 0.00279 |
| Channels where Correction wins on MSE | — | **4 / 7** |

## Interpretation

**The Correction Set Oracle's downstream MSE advantage from the channel-0
diagnostic partially, not fully, reproduces across the full multivariate
setting**: Correction wins on 4 of 7 channels individually, and wins on
the cross-channel MEAN gain (0.5514 vs 0.5437) — but loses on 3 channels
(3, 5, 6), all three of which happen to be channels with relatively LOW
B0 MSE (0.057–0.270), i.e. channels B0 already forecasts well, where there
is less room for either oracle to improve much and the two are closer to
tied.

**A genuinely new finding, differing from the channel-0-only diagnostic**:
NDCG@10 now favors the Correction Oracle on the cross-channel mean (0.699
vs 0.656) and on 5 of 7 individual channels — the OPPOSITE direction from
`EXP-CORRECTION-ORACLE-DIAG01`'s own channel-0 result (where Future had
the higher NDCG). Mean rank and margin still favor Future Oracle on the
aggregate, consistent with the earlier channel-0 finding (Correction
target less separated / harder to rank by a past-only score) — but this
is no longer a uniformly negative learnability signal once NDCG is
considered across all channels; the two rank-quality metrics (NDCG vs.
mean-rank/margin) now DISAGREE with each other, not just with the exact-
choice metrics as they did in Track A's own diagnostics.

## Conclusion

Confirms the channel-0 diagnostic's headline finding (Correction Oracle
modestly beats Future Oracle on downstream MSE) generalizes on AVERAGE and
on a MAJORITY of channels, but is not universal — 3 of 7 channels favor
Future, concentrated among the easiest-to-forecast channels. Learnability
signals remain genuinely mixed (NDCG now favors Correction on average;
rank/margin still favor Future) rather than cleanly negative as the
channel-0-only result suggested. This nuances, without overturning,
`EXP-CORRECTION-ORACLE-DIAG01`'s own Outcome B-B classification.

---

Source files: `summary_aggregate.json`, `per_channel_metrics.csv`,
`summary_channel{0..6}.json`, `command.txt`, `config.json`,
`sanity_summary.json`, `env.txt`, `checkpoint_fingerprints.txt`,
`working_tree.diff`, `logs/`.
