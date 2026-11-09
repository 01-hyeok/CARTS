# Full candidate query analysis 02

Owned diagnostic implementation; shared CARTS code is not modified. Existing checkpoint/cache files are read-only. No training, new seeds, or push.

## Execution

Activate `/data/pjh_workspace/ts-env/bin/activate` first. `controller.py` runs E96, E720, W96, W720 sequentially under a file lock. It uses `/usr/bin/python3` (legacy torch1.12 CUDA11.3) for `extract.py` because original retrieval reduction must reproduce exactly. Only GPU UUID `GPU-91418a73-202a-2a9b-57d5-2538b19e07ec` (physical1) is visible. Do not start a duplicate controller.

`analyze.py --cell CELL` and `finalize.py` run on CPU via ts-env Python with CUDA hidden. `finalize.py` requires all4 ANALYSIS_PASS markers, verifies all11 cached Stage1 metrics against the prior31-subset array, and generates the Korean report plus aggregate small tables.

Tests: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider experiments/full_head_query_analysis02/test_core.py`.

## Outputs and units

Root: `results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02/`.
Each cell has PROVENANCE, exact EXTRACTION_PASS, ANALYSIS_PASS, POST_VERIFICATION, and small CSVs. `local/` contains large query-level arrays; keep local, do not commit/upload.
- Query IDs are global history-start row indices in the original dataset.
- Candidate IDs index full memory; `candidate_start_ids.npy` maps them to global train-start indices.
- `[val,test]_top10_ids.npy`: query x channel x11 configurations x10, order in QUERY_ARRAY_SCHEMA.
- `[val,test]_loo_entering_ids.npy` / leaving IDs: query x channel x5 removed heads x10; -1 means an unused rank slot. Map nonnegative IDs through candidate_start_ids.
- Forecast metrics: query x channel x11 x4 (fixed MSE, fixed MAE, existing-refit MSE, existing-refit MAE).
- Strata features come only from historical inputs. Validation per-channel tertiles are frozen before test; no future-based selection.
- Global delta is configuration minus All5. Positive means All5 helps. Same candidate sets count as ties even when canonical float32 summation order leaves a tiny residual. Canonical forecasts/metrics remain untouched.
- Feature strata use equal-channel macro; standardized tables further hold a common channel set across all3 bins in both validation/test. Time-block resampling keeps channels paired, retains draws with every fixed channel present, and reports valid draw counts.
- Simple head-prediction averaging may use up to50 candidates. It is a counterfactual aggregation diagnostic, not an equal Top10-budget comparison.

Existing refit lambdas were selected by the previous validation protocol; none are refitted here. The new primary diagnostic fixes the old All5 lambda. Old ANALYSIS01 primary columns referred to per-configuration refit and remain separate.
