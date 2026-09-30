# TRACK-V-MULTIQUERY-GENERALIZATION01 -- AUDIT.md

## PART 0/16: GPU and pre-flight state

Standing rule: GPU1 only. Verified fully free (`ps`/`nvidia-smi`, 0%
util, 4 MiB used, no CARTS process) immediately before starting.
`CUDA_VISIBLE_DEVICES=1` throughout.

## Reuse mapping (why almost no new architecture code was needed)

TRACK-V's four arms map EXACTLY onto architectures this session has
already built, audited, and validated multiple times:

| Arm | Architecture | Script (reused UNMODIFIED except one additive CLI extension) |
|---|---|---|
| V0 (S=0) | TRUE Original KL, 0 projection params | `train_j_shared_encoder_drift01.py` (same script as TRACK-T2's T0 / TRACK-U's U0) |
| V1 (S=1) | one learnable query projection, key unprojected, full gradient | `train_t_pure_multislot01.py --num_slots 1` (= TRACK-T's own S1) |
| V2 (S=2) | two learnable query projections | `train_t_pure_multislot01.py --num_slots 2` (= TRACK-T's own S2) |
| V5 (S=5) | five learnable query projections | `train_t_pure_multislot01.py --num_slots 5` (NEW value) |

Only two purely-ADDITIVE CLI extensions were needed (nothing removed,
nothing changed for existing callers):
1. `train_t_pure_multislot01.py`: `VALID_NUM_SLOTS = (1, 2, 4, 10)` ->
   `(1, 2, 4, 5, 10)`.
2. `build_t_multislot_cache01.py`: `--num_slots` choices extended the
   same way.

`tests/test_t_pure_multislot_validation01.py` and
`tests/test_t2_projection_multislot_decomposition01.py` (30 tests, the
existing TRACK-T/TRACK-T2 suites) were re-run after both changes: still
30/30 pass, confirming zero behavior change for S in {1,2,4,10}.

## PART 4 cross-arm initialization fairness -- holds by construction

`SlotHeads.__init__` (from `train_k_multislot_predictive_retrieval01.py`,
reused unmodified since TRACK-T/K) generates each slot's `W_m` via its
OWN fixed seed (`1000 + m`), independent of `n_slots`. This means
`SlotHeads(n_slots=1).W[0] == SlotHeads(n_slots=2).W[0] ==
SlotHeads(n_slots=5).W[0]` (all seed 1000) and `SlotHeads(n_slots=2).W[1]
== SlotHeads(n_slots=5).W[1]` (both seed 1001) are bit-identical
WITHOUT any new code -- PART 4's fairness requirement was already
satisfied by TRACK-T's own architecture. Verified live (`torch.equal`,
not just `allclose`) in `tests/test_v_multiquery_generalization01.py`
items 2/2b, plus a sanity check (item 2c) that the 5 slots within V5 are
NOT all identical to each other (symmetry genuinely broken).

## Cache format extended (additive) for D/C bootstrap (PART 9)

Neither `build_t_multislot_cache01.py` nor
`build_t2_true_original_kl_cache01.py` previously saved PER-QUERY D/C
values (only the final scalar mean, over both queries and channels
combined) -- insufficient for query_start_idx-unit paired bootstrap on
D/C specifically (PART 9: "가능하면 D/C 각각도 bootstrap"). Both cache
builders were extended to additionally save `D_per_query`/
`C_per_query` (channel-averaged per query, aligned to
`query_start_idx`) into the cache dict -- purely additive, does not
change `relation_outputs`/`query_start_idx` or any existing scalar
metric. Verified consistent with the saved scalar `D`/`C` in a live
smoke test (`mean(D_per_query) == metrics['D']` to float precision) and
by unit test item 9 (`skipif`-guarded on real artifacts).

While already in this code, the same train/val-split-wasted-Spearman
inefficiency found and fixed in TRACK-U's cache builder was ALSO
present in `build_t2_true_original_kl_cache01.py` (used by V0) --
fixed the same way (Spearman only computed on the `test` split, whose
metrics are the only ones ever saved) before any real run.

## Missing base forecaster: Weather H720

TRACK-R never completed a Weather H720 setting (it crashed on Weather
H96's M2 arm and never proceeded to H720). No
`checkpoints/track_r_final_method_generalization01/Weather/H720/`
directory exists. Since Stage2 requires one frozen base forecaster
shared identically across all 4 arms, and PART 8 requires "동일한 base
forecast checkpoint," a fresh Weather H720 base forecaster was trained
via `train_r_base_forecaster01.py` (the SAME script, unmodified, that
TRACK-R itself used for every other setting), using the same reference
checkpoint naming convention (`..._S0_wce_...`) TRACK-R always used.
Saved under THIS track's own checkpoint tree
(`checkpoints/track_v_multiquery_generalization01/Weather/H720/seed0/base/`)
-- not written into TRACK-R's own tree, to respect that track's
ownership boundary. Result: `test_mse=0.319430, test_mae=0.343818,
best_epoch=7`. All 4 of this setting's V-arms share this one checkpoint.

## Pipeline integration smoke tests (pre-flight)

Ran real 1-epoch/3-batch V5 (S=5) and V0 training -> cache chains on
GPU1 before the full run; both completed without error, `Agg=D+C`
identity held, `D_per_query`/`C_per_query` verified present and
numerically consistent with the saved scalar metrics for both V0 and
V5's cache output.

## Unit test summary (PART 10.D, 18 items, 17 pass / 1 skip-until-artifacts)

`tests/test_v_multiquery_generalization01.py`: V5 accepted as a valid
`--num_slots` value; cross-arm `W_1`/`W_2` init bit-identical across
S=1/2/5; the 5 slots within V5 are mutually distinct; V0 score/loss
match the literal Original-KL reference; query AND candidate encoder
gradients (and, for S>=1, the projection gradient) are all nonzero for
every arm; teacher has zero S-dependence; K=10 with all-unique
selection for S in {1,2,5}; `Agg=D+C` holds for S in {1,2,5}; S=1's
round-robin selection reduces exactly to ordinary Top-10; cache
`D_per_query`/`C_per_query` consistent with saved scalar D/C
(skip-guarded until real artifacts exist). Full pytest suite unaffected
(1381 passed, same 2 pre-existing failures, before this track's own
real runs).

## STOP/VALIDITY checks

No trigger encountered pre-execution: V0 has zero projection
parameters (no SlotHeads/projection module anywhere in
`train_j_shared_encoder_drift01.py`); candidate-branch gradient
verified nonzero for every arm (no `.detach()` anywhere in
`compute_scores_full_grad`, already an established TRACK-T invariant);
the ONLY intended difference across V0-V5 is `num_slots`/projection
presence (teacher, candidate mask, memory, split, checkpoint criterion,
Stage2 base, and K=10 are all identical by construction -- shared code
paths, not per-arm branches); `Agg=D+C` holds; cross-arm init fairness
holds bit-exactly; batch order is deterministic via the same shared
`make_loader_generator`/`_get_data` path every arm already used in
TRACK-T/TRACK-T2/TRACK-U.
