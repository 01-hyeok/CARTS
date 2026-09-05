# EXP-FRR01 model-discovery pilot -- factual notes

## Scope actually run

ETTh1 H96 only, 1 seed, 5 arms (R0/R1/R2/R12/R3) + B0 reused. This is the
model-discovery pilot only, not the full protocol (temporal legality gate,
exact arm definitions, sanity/leakage checks) -- see the chat transcript for
the full Model Discovery writeup this pilot was gated behind.

## A real architecture gap found and fixed mid-run

`RelationStage2.load_stage1_checkpoint` originally transplanted only the
encoder, `shared_cross_projection`, and `retrieval_metric` weights from a
Stage-1 checkpoint. It had no path for the new `query_cond_proj` /
`candidate_cond_proj` conditioning modules, so running Stage-2 as first
implemented would have kept only "an encoder shaped by conditioning during
training" and silently dropped R1/R2's actual live mechanism at retrieval
time -- not what R1/R2/R12/R3 are defined to be. Confirmed with the user
before proceeding (they chose "wire it properly" over the two faster/partial
options). Fixed by:
  - adding `stage2_query_base_conditioning` / `stage2_candidate_residual_conditioning`
    flags and the matching `query_cond_proj` / `candidate_cond_proj` modules
    to `RelationStage2`
  - extending `load_stage1_checkpoint`'s existing per-module loop (already
    used for `retrieval_metric`/`pairwise_scorer`) to also load these two
  - threading `query_residual` into `build_retrieval_cache` (the static-cache
    z_q site) and `memory_residual` into `build_memory_key_bank` (the
    candidate-bank z_k site) -- both needed because, historically, this
    codebase has a recorded bug pattern of wiring a mechanism into only one
    of the two z_q/forced-selection call sites and getting silently-identical
    results across arms; both sites were checked and wired here
  - `_e2e_extras` (renamed in effect, not in name) now populates
    `query_residual`/`memory_residual` whenever the new conditioning flags are
    active, independent of `stage2_e2e` (previously it returned `{}`
    immediately when `stage2_e2e=0`, which is what B0's protocol always uses)
  - the `else` (eval/no-grad) branch of `_run_loader`'s `self.model(...)` call
    was passing no `target_y`, but `_condition_query_embedding` needs
    `query_y` to recover `base = Y_q - R_q`; added `target_y=batch_y` there
    too. Not a leakage change: `base` is a deterministic function of `X_q`
    alone recovered from two already-precomputed quantities, not new
    information Y_final's forecast head could see.

Verified with a 1-epoch smoke test of R3 (the arm exercising every new code
path at once) end to end before the real 5-arm x 10-epoch run.

## Sanity

`pytest tests/` before and after every code change: 444 passed, exactly the
2 pre-existing failures (`test_topk_coverage_reuses_target_indices_across_relations`,
`test_identity_retrieval_uses_raw_target_source_relation_without_encoder`),
no regression introduced.

## Temporal legality

`[legality]` line printed and asserted at Stage-1 run start for every arm
(see `exp/exp_stage1_relation.py::_residual_cache`): `memory_residual` row
count must equal the train split's query count (8449 for ETTh1), which it did
for every arm. The residual-teacher cache
(`cache/residual_teacher_frr01/ETTh1_pred96.pt`, sha256 in
`residual_cache_sha256.txt`) was rebuilt from the current campaign's own B0
Stage-2 checkpoint rather than reusing the older
`cache/residual_teacher/ETTh1_pred96.pt` (built from a different, older
KL/raft_concat checkpoint -- a provenance mismatch this run did not want to
carry, matching a past discrepancy the user caught in this same campaign).

## Checkpoint selection

Every Stage-1 checkpoint selected by validation `hard_aggregate_mse10` (never
test, never each arm's own training objective), matching every prior arm
selection in this campaign (EXP-3).

## Decision rule and result (see the chat report for the full table)

Pre-registered: only an arm beating B0 by >=0.01 Stage-2 Final MSE triggers a
3-seed confirmation run; Stage-1 proxy improvement (recall, hard_aggregate_mse10)
alone is not sufficient for GO. Actual deltas vs B0 (0.37312): R0 +0.00085,
R1 -0.00213, R2 +0.00711, R12 +0.00800, R3 +0.00641 -- none reach +/-0.01, so
no 3-seed confirmation was triggered and no arm meets GO.
