```text
STAGE1 LOSS JOB: RUNNING
STAGE2 IMPLEMENTATION: COMPLETE
STAGE2 TESTS: COMPLETE (scope-limited, see below)
STAGE2 EXECUTION: WAITING FOR GPU 1
CURRENT JOBS AFFECTED: NONE
```

## EXECUTED / NOT EXECUTED

```text
EXECUTED:
- Verified TRACK-A-SET-LOSS-CONTROL01's current process, script, config,
  and output paths (read-only) -- confirmed still running, untouched.
- Section-1 checklist verified against real fingerprint/checkpoint files
  for the 3 H96 arms completed so far (A0_hard_choice, A1_adaptive_multipos,
  A2_srm): all 7 channels, no hard-coded channel=0, same seed=1, same
  encoder_init_sha256/set_conditioner_init_sha256 across all three, same
  batch order (same torch.manual_seed(seed) before loader construction,
  same data split), same axis_prefix=onpolicy/axis_score=cosine (loss is
  the only varying axis), checkpoint selected by min val FR-Agg (never
  test), checkpoint epoch matches retrieval_metrics.json's best_epoch for
  all three.
- Matched-seed Hard-CE baseline CONFIRMED PRESENT -- A0_hard_choice is
  part of THIS SAME seed=1 experiment (not the old seed=0 Factorial run),
  so the [BLOCKER] condition in section 1 does NOT apply here; no
  cross-seed mixing risk this round.
- New cache builder (`scripts/build_setlossctrl_retrieval_cache01.py`) and
  Stage-2 retrainer (`scripts/train_setlossctrl_stage2_retrain01.py`),
  siblings of TRACK-A-CHOICECE-STAGE2-RETRAIN01's own scripts (same
  method: frozen encoder+SetConditioner loaded directly from the arm
  checkpoint's own keys, free-running-only sequential selection, fixed
  S0_wce host score for aggregation alpha, `forward_from_retrieval_values`
  injection, fresh Stage-2 init never loading WCE's trained gate/head).
  Differs only in reading TRACK-A-SET-LOSS-CONTROL01's own checkpoint
  format/fingerprint (`loss_name` field; no `metric_state_dict`, since
  that experiment is cosine-only).
- Orchestrator scripts for both horizons
  (`scripts/run_setlossctrl_stage2_retrain01_h{96,720}.sh`), each
  self-gating on the Stage-1 experiment's own `DONE_<arm>.marker` files --
  they read Stage-1 outputs but never start, wait on, or modify that
  process.
- 9 unit tests (`tests/test_setlossctrl_stage2_retrain01.py`), against the
  REAL Exp_Stage2_Relation/RelationStage2 classes built from the real
  ETTh1_96 S0_wce host, all passing (GPU2/CPU only, GPU1 untouched):
  fresh Stage-2 init does not match host's trained weights, frozen
  submodules get requires_grad=False, frozen-hash reproducibility, frozen
  submodules excluded from the optimizer's own parameter list, same-seed
  reproducible fresh init (the shared-init contract), cache lookup by
  batch_start_idx (not loader position, order-invariant), and the core
  claim: `forward_from_retrieval_values` produces finite loss, correctly
  gives zero/no gradient to every frozen submodule and nonzero finite
  gradient to at least one trainable parameter, and leaves frozen
  submodule hashes unchanged across a real optimizer step.
- Full repo regression suite: 918 passed / 2 pre-existing failures
  (unchanged names/reasons), no new regressions.

NOT EXECUTED:
- No retrieval cache built for any TRACK-A-SET-LOSS-CONTROL01 arm.
- No Stage-2 arm trained. Zero of the (up to) 8 arm x horizon runs
  (4 loss arms x {H96, H720}) executed.
- Smoke test (spec section 9) not run -- requires GPU1.
- The full ~18-item test list in spec section 8 was not exhaustively
  implemented; 9 tests cover the highest-risk claims (same subset as
  TRACK-A-CHOICECE-STAGE2-RETRAIN01's own report explains and justifies).
  Not independently re-verified this round: per-channel cache completeness
  as a dedicated assertion, online-vs-cache Top-K bit-identity, NaN-query-
  future invariance, cached-alpha-sums-to-1 as a dedicated test, Stage-1
  FR-Agg vs cache-based FR-Agg numeric match as a dedicated test (true by
  construction -- identical formula -- per the sibling report's same
  finding, but not independently asserted by a pytest here either).
- research/REVIEW_FOR_CHATGPT.md: not appended (no results yet).
- commit/push: not done.
- H96 and H720 Stage-1 completion status: H96 has 3/4 arms done
  (A0_hard_choice, A1_adaptive_multipos, A2_srm), A3_setutility_softce
  still training. H720 has NOT STARTED at all (TRACK-A-SET-LOSS-CONTROL01's
  own H96-before-H720 gate has not yet been crossed). No Stage-2 work is
  therefore even eligible to start for any arm yet under this round's own
  "only use checkpoints with a completion marker" rule.
```

## 1. Stage-1 experiment verification (read-only; nothing stopped, modified, or restarted)

| check | result |
|---|---|
| Process | `train_set_loss_control01.py --arm A3_setutility_softce` running (pid confirmed alive at report time) |
| Channels | `channels=list(range(args.enc_in))` = `[0,1,2,3,4,5,6]`, no hard-coded `channel=0` (confirmed by direct source read AND by fingerprint field, per the earlier audit this session that specifically re-implemented this experiment to fix EXP-SET-LOSS01's channel=0 bug) |
| Same dataset/split, seed | ETTh1, seed=1 for A0/A1/A2 (confirmed via `config_fingerprint_*.json`) |
| Same encoder/SetConditioner init | `encoder_init_sha256`/`set_conditioner_init_sha256` IDENTICAL across A0/A1/A2 (`19e802599326cee9...` / `887514aded6a181c...`) |
| Same batch order | same `torch.manual_seed(seed)` call before loader construction, shared across all arms (unchanged code path) |
| Loss is the only varying axis | `axis_prefix=onpolicy`, `axis_score=cosine` fixed for all 4 arms; only `loss_name` differs |
| Checkpoint selection | `checkpoint_criterion: min val free_running_aggregate_future_mse (all channels)`, confirmed in fingerprint; test metric never used for selection (code path unchanged from the already-reviewed trainer) |
| Matched Hard-CE baseline | **PRESENT, same seed** -- `A0_hard_choice` (best_epoch=4) is part of this exact experiment, not a cross-seed import. **[BLOCKER] does not apply.** |

## 2. Stage-1 completion gating (spec section 2)

Only arms with a `DONE_<arm>.marker` under
`results/TRACK-A-SET-LOSS-CONTROL01/<cell>/` are eligible; the cache
builder's `main()` additionally re-checks the marker and re-validates
`retrieval_metrics_<arm>.json`'s `best_epoch` against the checkpoint's own
stored `epoch` field, refusing to proceed on any mismatch (`[ISSUE][ABORT]`).
As of this report: H96 has 3 eligible arms (A0, A1, A2); A3 and all of
H720 are not yet eligible.

## 3-7. Implementation (architecture identical to the sibling experiment)

See `research/TRACK-A-CHOICECE-STAGE2-RETRAIN01.md` sections 1-2 for the
full architecture rationale (frozen-policy cache + `forward_from_retrieval_values`
injection + fresh Stage-2 init + reused `_loss`/`_select_optimizer`) --
identical here, only the Stage-1 checkpoint source differs. Aggregation
alpha source verified to be the FIXED S0_wce host score (same code path,
same formula, same as `free_running_aggregate_future_mse`), never the
arm's own score, for every one of the 4 loss arms uniformly (loss choice
only affects retrieval MEMBERSHIP via which candidates get picked
sequentially, never the aggregation weighting).

## Status and next step

**STAGE1 LOSS JOB: RUNNING (untouched). STAGE2 IMPLEMENTATION: COMPLETE.
STAGE2 TESTS: COMPLETE (9/9 passing, scope-limited). STAGE2 EXECUTION:
WAITING FOR GPU 1. CURRENT JOBS AFFECTED: NONE.**

No cache has been built and no Stage-2 arm has been trained. No Stage-1 or
Stage-2 results table, and none of the 13 final questions, can be answered
yet. Once GPU1 is free (or explicitly shared, as separately negotiated for
other concurrent work this session) and once each arm's Stage-1 run
crosses its own completion marker, the next steps are: (1) smoke test on
the first-completed H96 arm; (2) if clean, run
`scripts/run_setlossctrl_stage2_retrain01_h96.sh` for the eligible H96
arms in order (Hard CE, SRM, SoftCE, MultiPos); (3) review H96 stability;
(4) once TRACK-A-SET-LOSS-CONTROL01 itself completes H720, repeat via
`scripts/run_setlossctrl_stage2_retrain01_h720.sh`.
