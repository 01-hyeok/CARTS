# TRACK-A-PATCH-RETRIEVAL-EXPERT01 — INTERIM STATUS (INFRA ONLY, NOT STARTED)

**Status: Phase A has NOT been launched yet.** This document records
infrastructure built and verified so far -- no real training run (beyond a
throwaway smoke test) has produced a result. Nothing here is a research
finding. A real report will replace this once Phase A actually runs on
ETTh1/Weather/Solar.

Generated: 2026-09-22.

## 1. Scope and boundary

Separate experiment from `TRACK-A-HORIZON-RETRIEVAL-EXPERT01` (category C).
That experiment varies which FUTURE block is retrieved for; this one varies
the PAST input's patch granularity, with a single full-horizon [0:720]
relevance target throughout -- no future-block teacher. Uses its own code,
checkpoint, and result paths:
- `scripts/train_patch_retrieval_expert01.py`
- `scripts/calibrate_patch_retrieval_expert01_temperature.py`
- `results/TRACK-A-PATCH-RETRIEVAL-EXPERT01/`
- `checkpoints/track_a_patch_retrieval_expert01/`

Nothing under category C was read or written while building this.

## 2. Pre-execution code verification finding (spec-mandated check)

Per the experiment spec's explicit instruction to verify the candidate
mask, full-memory search, `delta_last` reconstruction, and Stage-2
cache/fusion paths against the actual code before running -- a real
mismatch was found and corrected before any real run:

**Finding**: every S0_wce reference checkpoint reused this session
(ETTh1/Weather/Solar, all horizons) was trained with
`relation_encoder_type='mlp'` -- a flat MLP over the raw 720-length input.
It has no `patch_len`/`stride` awareness at all. A first smoke test with
`patch_len=16` vs `patch_len=24` produced byte-identical train/val
metrics, which is how this was caught (not assumed from reading code alone
-- confirmed by the identical numbers, then traced to the `mlp` branch in
`models/RelationStage1.py::RelationEncoder.__init__`).

**Fix**: `build_experiment`'s override dict now explicitly forces
`relation_encoder_type='transformer'` (the branch that actually owns
`RelationPatchEmbedding`, i.e. patch_len/stride/num_patches) and
`relation_self_fill='zero'` (transformer forbids the host's default
`'linear'`; `'zero'` is applied identically to every patch-size arm, so it
stays a controlled constant, not a free variable). Verified directly
(no training) that patch_len in {16, 24, 48, 120} now produces distinct
`RelationPatchEmbedding.num_patches` (45 / 30 / 15 / 6) and distinct
parameter counts before any real run was launched. Recorded in each arm's
`config_fingerprint_*.json` under `relation_encoder_type_corrected_from_host`.

This also means every OTHER Track-A scratch-encoder experiment this
session (B, C) has been training a flat MLP, not a patch-aware Transformer
-- harmless for those experiments (they never varied patch granularity),
but material context for interpreting any future comparison against them.

## 3. What is verified so far

- Unit tests: `tests/test_patch_retrieval_expert01.py`, 8/8 passing --
  non-overlap patch-count arithmetic (16/24/48/120 against seq_len=720),
  `recall_at_k`/`ndcg_at_k` on known synthetic cases, and a dedicated
  **candidate-side gradient flow test** (guards against the session's
  earlier candidate-embedding-detached-by-accident class of bug).
- Full regression suite: `1064 passed, 2 failed` -- the 2 failures match
  the pre-existing documented baseline exactly (`research/RESEARCH_CONTEXT.md`
  Known Repository Issues); no regression introduced.
- Temperature calibration (`calibrate_patch_retrieval_expert01_temperature.py`)
  run for real on ETTh1_720 train queries (n=256, teacher only, no model):
  medians at tau_T={0.1, 0.05, 0.02} were {0.0613, 0.2018, 0.7021}; rule
  (closest Top-10 mass median to 0.5, ties toward larger tau_T) selected
  **tau_T=0.02**. Not yet run for Weather or Solar.
- Smoke training (2 epochs, `--limit_batches 3`, throwaway checkpoints/results
  deleted after): ETTh1_720 `patch_len=24` completed end-to-end successfully
  after the fix (val_model_top10_individual_mse and val_oracle_regret both
  finite and moving epoch-to-epoch, distinct from the earlier `patch_len=16`
  run's numbers). `patch_len=16` smoke hit `CUDA OutOfMemoryError` twice --
  diagnosed as GPU1 resource contention with two other concurrently-running
  Track-A jobs (Solar_96 TF-Oracle-Learnability01, Solar_720
  Horizon-Retrieval-Expert01), not a script defect; not yet retried.

## 4. Not yet done (everything downstream of infra)

- Phase A real runs (any dataset, any patch size) -- zero real training has
  happened. `patch_len=16` ETTh1 smoke retry is the next immediate step,
  pending GPU1 headroom.
- Temperature calibration for Weather, Solar.
- Phase A post-hoc query-level "best patch" Oracle diagnostic script
  (section 3's pass/fail gate) -- not written yet.
- Phase A pass/fail gate itself has therefore not been evaluated for any
  dataset.
- Phase B (shared-encoder multi-expert), Stage-2/ACF (sections 4-5 of the
  spec) -- not started; explicitly gated on Phase A passing its own
  criterion first, per spec.

No interpretation or next-direction recommendation is offered -- there is
no result yet to interpret.
