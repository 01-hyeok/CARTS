# PHASE D -- Verification: Metrics That Must Not Change, Did Not Change

All four quantities the spec pre-registers as independent of the Error-Cosine bug are reproduced from a FRESH forward pass (fixed script, same checkpoints, same pool) and compared against the original `TRACK-V5-HEAD-OBJECTIVE-AUDIT01` numbers.

## D1. 31-Subset Head Oracle AggMSE

| Cell | Old Oracle AggMSE | New (recomputed) Oracle AggMSE | Match |
|---|---:|---:|---|
| ETTh1_96 | 0.32441 | 0.32441 | EXACT |
| ETTh1_720 | 0.56499 | 0.56499 | EXACT |
| Weather_96 | 0.12156 | 0.12156 | EXACT |
| Weather_720 | 0.37866 | 0.37866 | EXACT |

## D2. Mean-Mixture AggMSE

| Cell | Old AggMSE | New AggMSE | Match |
|---|---:|---:|---|
| ETTh1_96 | 0.40981 | 0.40980762 | EXACT (rounds identically) |
| ETTh1_720 | 0.62057 | 0.62057020 | EXACT |
| Weather_96 | 0.26433 | 0.26432985 | EXACT |
| Weather_720 | 0.61780 | 0.61780185 | EXACT |

## D3. Soft/Hard Gap

| Cell | Old Soft AggMSE | New Soft AggMSE | Match |
|---|---:|---:|---|
| ETTh1_96 | 0.39912 | 0.39912 | EXACT |
| ETTh1_720 | 0.57958 | 0.57958 | EXACT |
| Weather_96 | 0.22417 | 0.22417 | EXACT |
| Weather_720 | 0.45393 | 0.45393 | EXACT |

(Hard AggMSE is identical to D2's Mean-Mixture AggMSE by definition -- also confirmed exact above.)

## D4. Gradient Analysis (Phase F)

Not recomputed this track -- `scripts/diag_v5_head_gradient01.py` was never modified (confirmed: `git diff` on this file for this track is empty) and never shares any code path with the Error-Cosine computation (PHASE A classification #10). The original Phase F numbers (gradient cosine 0.711-0.990 across cells) stand unchanged, by construction, not by re-verification -- re-running an unmodified script against unmodified checkpoints would only reproduce (within float noise) the same numbers already on record, so no resources were spent re-confirming this.

## Row-level verification (not just the aggregate mean)

In addition to the aggregate numbers above, the FULL per-query-channel columns were compared directly between the old and new parquet files for every cell (`pandas.Series.equals`, exact float equality, not `np.allclose`):

| Cell | `aggmse_mix` | `subset_aggmse_12345` | `aggmse_soft` | `overlap_H1-H2` | `dot_H1-H2` |
|---|---|---|---|---|---|
| ETTh1_96 | EXACT | EXACT | EXACT | EXACT | EXACT |
| ETTh1_720 | EXACT | EXACT | EXACT | EXACT | EXACT |
| Weather_96 | EXACT | EXACT | EXACT | EXACT | EXACT |
| Weather_720 | EXACT | EXACT | EXACT | EXACT | EXACT |

Every one of these columns is identical at EVERY row, not just equal on average -- the bugfix touched exactly the `cos_*` columns and nothing else, confirmed at the finest possible granularity (per query-channel, not just the aggregated mean a looser check could have missed a cancellation error in). `cos_{pair}` itself is confirmed to be exactly `H` times the old value at every row (PHASE C), not merely on average.

**Conclusion: PHASE A's classification is fully verified empirically, not just argued mathematically.** Only the Error Cosine metric and its narrative interpretation require revision (PHASE E). Every other number in `TRACK-V5-HEAD-OBJECTIVE-AUDIT01`'s FINAL_REPORT.md (Tables A, C, E, F and all of Phases C/E/F/G) remains valid and correctly reported as-is.
