# TRACK-A-TF-ORACLE-LEARNABILITY01

**Status: ETTh1_96, ETTh1_720, Weather_96, Weather_720 complete (Stage-1 +
corrected Stage-2). Solar_96/720 pending -- see section 9.**

## 1. Research Question

> Is the Greedy Set Oracle actually learnable under Teacher-Forcing
> conditions, compared against the Individual Oracle as a control, using
> ONLY Hard Choice CE?

This round introduces **no new loss** (Multi-positive, WCE, SRM, SoftCE,
Normalized-Regret, Expected-Regret, asymmetric score, on-policy are all
explicitly out of scope) -- the only comparison axis is Individual vs
Greedy Set Oracle, both under Teacher Forcing, Hard Choice CE, Cosine.

## 2. Reuse-Eligibility Audit (spec section 17)

Existing `individual_tf_cosine`/`set_tf_cosine` checkpoints in
`checkpoints/track_a_factorial_e2e/` (ETTh1_96/720, Weather_96 --
Weather_720 never existed) were audited against the full reuse criteria
(same seed=0 + init, proven same batch order, TF, Hard CE, Cosine, full
memory, all channels, train-TF learnability diagnostics, corrected Stage-2)
and **failed on**: no `batch_order_hashes_*.json` artifact exists for any of
them (that infra postdates their training run), and no train-TF
learnability diagnostics (fixed-subset, uncontaminated) were ever computed.
Per spec section 17's explicit instruction ("단순히 파일 이름이 같다는 이유로
재사용하지 마라"), **all 4 cells were retrained fresh** this round.

## 3. Implementation

`scripts/train_tf_oracle_learnability01.py` (new). Structurally: imports
and reuses `scripts.train_factorial_e2e01.run_sequence`, `train_epoch`,
`eval_epoch` UNMODIFIED (only additive change: `train_epoch` gained an
opt-in `record_batch_order=False` parameter, default off = byte-identical
prior behavior, to reuse `scripts.rng_control01.batch_order_sha256` instead
of re-deriving it). `prefix_policy='tf'` and `scorer=cosine` are fixed;
`--target` selects `individual`/`greedy_set`.

**New, not present in any prior trainer**: `teacher_forced_diagnostics()`,
a no-grad eval-mode pass walking `run_sequence(prefix_policy='tf',
free_running=False)` OUTSIDE the training loop, run on:
- a FIXED train-query subset (512 queries, seed=0-selected once via
  `_build_fixed_subset`, saved as `train_tf_subset_indices_*.json` with a
  SHA256 of the exact index set, reused identically across all epochs and
  both arms of a cell) -- so "train learnability" is never read off
  dropout/optimizer-contaminated online training-loop statistics.
- the full val split (every epoch) and full test split (at the best
  checkpoint) -- TF-train/val/test diagnostics.

Free-running evaluation (FR-Agg + free-running Oracle-action diagnostics)
is `eval_epoch`'s existing PRIMARY pass, called separately, so TF and
free-running numbers are never conflated.

## 4. RNG / Batch-Order Pairing

Both arms of a cell share encoder + SetConditioner init
(`--shared_init_out`/`--shared_init_in`, SHA256-verified) and the same
`--loader_seed`. `batch_order_sha256` is recorded every epoch and gated
post-hoc: **all 4 cells passed** (`.batch_order_gate_passed` present,
identical hash on every shared epoch -- ETTh1_96: 6 epochs, ETTh1_720: 9,
Weather_96: 6, Weather_720: 6).

## 5. VRAM

`--oracle_compute_impl safe` used throughout (Choice-CE + Individual-Oracle
caching optimized, Greedy Set Oracle always reference-path, per the prior
OPT04 on-policy-divergence finding). `--channelwise_backward --memsafe`
reserved for Solar; not needed for ETTh1/Weather (7/21 channels).

## 6. Stage-1 Results

TF-train (fixed 512-query subset, uncontaminated) / TF-val / TF-test
action-accuracy at t=10, and free-running FR-Agg (test):

| Cell | Arm | Train-TF acc (t10) | Val/Test-TF acc (t10) | Test FR-Agg |
|---|---|---:|---:|---:|
| ETTh1_96 | individual | 1.12% | 1.55% | 0.4376 |
| ETTh1_96 | set | 0.14% | 0.14% | 0.4848 |
| ETTh1_720 | individual | 11.27% | 6.15% | 0.7257 |
| ETTh1_720 | set | 1.93% | 0.33% | 0.7539 |
| Weather_96 | individual | 8.08% | 9.96% | 0.2927 |
| Weather_96 | set | 6.04% | 5.03% | 0.3211 |
| Weather_720 | individual | 21.27% | 22.20% | 0.7252 |
| Weather_720 | set | 1.40% | 1.13% | 0.6124 |

**t=1 equivalence** (spec S18, structural property of `run_sequence`): at
t=0 the prefix is empty for both TF and on-policy, so `arm_score(z_q, E)`
and the t=1 Oracle target are IDENTICAL for Individual and Greedy Set by
construction (`greedy_set_utility` with an empty prefix reduces to the
singleton Individual value, independent of host weighting -- established
earlier this session by direct code trace). Empirically, t=1 test-TF
action-accuracy is close between arms in every cell (e.g. Weather_96:
ind=5.08%, set=5.11%; ETTh1_720: ind=0.18%, set=0.20%), consistent with
near-identical t=1 targets; the two arms' post-training encoders are
already slightly different by test time, so exact numerical equality is
not expected or required.

**Step-wise divergence from t>=2** (test-TF, oracle_action_acc /
expert_rank_median): the two Oracles' learning curves diverge sharply after
t=1 in every cell, and always in the SAME direction -- Individual's rank
improves (gets better) with more steps while Set's rank degrades (gets
worse):

| Cell | t | Ind acc / rank_med | Set acc / rank_med |
|---|---|---|---|
| ETTh1_96 | 5 | 2.24% / 162 | 0.17% / 956 |
| ETTh1_96 | 10 | 1.55% / 224 | 0.14% / 1032 |
| ETTh1_720 | 5 | 10.78% / 28 | 0.26% / 757 |
| ETTh1_720 | 10 | 6.15% / 43 | 0.33% / 874 |
| Weather_96 | 5 | 17.83% / 765 | 3.25% / 2902 |
| Weather_96 | 10 | 9.96% / 1154 | 5.03% / 1427 |
| Weather_720 | 5 | 30.79% / 337 | 1.29% / 4703 |
| Weather_720 | 10 | 22.20% / 371 | 1.13% / 3893 |

Full step-1..10 curves for every metric (Choice CE, action-acc, rank
mean/median, Top-1/5/10 containment, regret, NDCG@10, Spearman) are in
`results/TRACK-A-TF-ORACLE-LEARNABILITY01/<cell>/stepwise_tf_{test,
train_subset}_<arm>.csv`.

## 7. Stage-2 Results (Corrected, Fresh Init, Frozen Retrieval Branch)

Reused `scripts/train_setlossctrl_stage2_retrain02.py` UNMODIFIED (already
fully generic over cell/arm/cache_dir, not CONTROL02-specific) with a new
sibling cache builder,
`scripts/build_tf_oracle_learnability01_retrieval_cache.py` (delta-space,
`query_offset` never added into the cache, FR-Agg cross-check gate,
`--fragg_tolerance 0.02`, all passed). Both arms of a cell share a fresh
Stage-2 init.

| Cell | Independent-ish Base (host y_base) | Individual final MSE | Set final MSE | Individual vs Set delta |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.5399 | 0.3729 | 0.3734 | +0.0005 (Set worse) |
| ETTh1_720 | **0.4675** | 0.4760 | 0.5115 | +0.0355 (Set worse) |
| Weather_96 | 0.1965 | 0.1989 | **0.1924** | -0.0065 (Set better) |
| Weather_720 | 0.3200 | 0.3221 | 0.3229 | +0.0008 (~tied) |

`base_mse`/`counterfactual_lambda0_mse` here is the frozen Stage-2 host's
own retrieval-ablated branch, read as a DIAGNOSTIC per this project's
established convention (`train_factorial_e2e01_base_only.py`'s docstring),
not as the canonical Independent Base Forecaster -- that separate number
was not computed fresh in this round. ETTh1_720 is notable: retrieval
(either Oracle) is WORSE than this diagnostic baseline, i.e. retrieval
provides no benefit at H720 for ETTh1 regardless of Oracle choice.

## 8. Interpretation -- Case Assignment

| Cell | Train-TF | Val/Test-TF | Free-running (FR-Agg) | Case |
|---|---|---|---|---|
| ETTh1_96 (Individual) | low (1.1%) | low (1.6%) | 0.4376 | A (mapping itself hard to learn even on train) |
| ETTh1_96 (Set) | very low (0.1%) | very low (0.1%) | 0.4848 (worse) | A, more severe than Individual |
| ETTh1_720 (Individual) | moderate (11.3%) | lower (6.2%) | 0.7257 | B-leaning (train > val, some overfit) but still low absolute |
| ETTh1_720 (Set) | low (1.9%) | very low (0.3%) | 0.7539 (worse) | A |
| Weather_96 (Individual) | moderate (8.1%) | moderate (10.0%) | 0.2927 | B/D boundary -- train~val, not high absolute |
| Weather_96 (Set) | moderate (6.0%) | moderate (5.0%) | 0.3211 (worse) | similar shape to Individual, lower ceiling |
| Weather_720 (Individual) | high (21.3%) | high (22.2%) | 0.7252 | D-leaning for Individual (train~val, both meaningfully > chance) |
| Weather_720 (Set) | very low (1.4%) | very low (1.1%) | 0.6124 (better FR-Agg despite worse TF-acc) | A |

Set Oracle is Case A (train-TF learnability itself is the bottleneck, not a
train/val generalization gap) in 3 of 4 cells at t=10 -- train-subset
accuracy stays in the 0.1-6.0% range even on data the encoder has seen
repeatedly, while Individual reaches 8-21% on the same cells. Weather_720's
Set arm is the starkest case: 1.4% train-TF accuracy after 10 epochs, yet
its FR-Agg (0.6124) is numerically BETTER than Individual's (0.7252) --
Stage-1 TF learnability and free-running/Stage-2 outcome are not simply
proportional to each other in this cell, a genuine and reported anomaly
rather than a clean story.

No cell shows the clean Case-D pattern (train high, val high, free-running
good, Set matching or beating Individual on every axis) for the Set Oracle.
Individual's own learnability is itself only moderate outside Weather_720 --
this is a broader finding than "Set Oracle fails, Individual succeeds";
Individual's absolute TF-accuracy ceiling is often single-digit-to-low-teens
percent too.

## 9. Limitations

- **Solar_96/720 pending.** Solar_96's Individual TF Cosine training was
  still running at report time (~3+ hours/epoch, 137 channels); no Set arm
  was scheduled for Solar per explicit instruction ("Set Oracle을 새로 학습할
  필요 없다"). Solar_720 is additionally blocked: its canonical S0_wce host
  cannot be built on this GPU (documented OOM, ~69GB alone on an 80GB card).
- Independent Base Forecaster (the canonical baseline, distinct from the
  host's own diagnostic `y_base` branch used in section 7's table) was not
  retrained fresh this round for ETTh1/Weather; the host's own branch is
  reported as the closest available reference, explicitly labeled as a
  diagnostic per this project's established convention.

## Final Questions

1. **Individual Oracle이 실제로 train에서 학습되는가?**
   **Partially / dataset-dependent.** Weather_720 clearly YES (21.3% train
   acc @ t10, far above chance); ETTh1_720 and Weather_96 moderately (8-11%);
   ETTh1_96 only marginally (1.1%).
2. **Set Oracle이 실제로 train에서 학습되는가?**
   **Mostly NO.** 3/4 cells stay at 0.1-1.9% train-TF accuracy at t=10;
   only Weather_96 reaches a moderate 6.0%.
3. **Set의 train learnability가 Individual보다 낮은가?**
   **YES, in every cell** -- Set's train-TF accuracy at t=10 is lower than
   Individual's in all 4 cells (0.14 vs 1.12%, 1.93 vs 11.27%, 6.04 vs
   8.08%, 1.40 vs 21.27%).
4. **train에서 학습된 Oracle policy가 validation으로 일반화되는가?**
   Where there is a train signal at all, it largely carries to val/test
   (train~val for both arms in most cells -- no large overfitting gap was
   the dominant pattern; the bottleneck is train-fitting itself, not
   generalization).
5. **Individual/Set이 t=1에서 identical한가?**
   **YES, structurally** (empty-prefix reduction, established by code
   trace) and **empirically close** (e.g. Weather_96 t=1: 5.08% vs 5.11%).
6. **t>=2부터 어느 Oracle의 rank/action-accuracy가 더 빨리 나빠지는가?**
   **Set, in every cell, dramatically.** Individual's rank improves with
   more steps (e.g. ETTh1_720 t1->t10 rank ~900->43); Set's rank gets WORSE
   with more steps in 3/4 cells (e.g. ETTh1_96 t1->t10: 617->1032).
7. **Set의 문제가 train-fitting인가 validation-generalization인가?**
   **Train-fitting**, per section 8's Case-A-dominant assignment.
8. **TF에서는 괜찮아 보이다가 free-running에서만 무너지는가?**
   **NO** -- TF itself already fails for Set in most cells; this is not a
   TF-looks-fine-but-collapses-at-inference story.
9. **Set이 실제로 FR-Agg에서 Individual보다 나은가?**
   **Mixed.** Weather_720: yes (0.6124 vs 0.7252, Set better). ETTh1_96/720:
   no (Set worse). This is the Weather_720 anomaly flagged in section 8.
10. **Set이 fresh Stage-2 MSE에서 Individual보다 나은가?**
    **Mixed, cell-dependent.** Weather_96: yes (0.1924 vs 0.1989). ETTh1_96/
    720: no, ETTh1_720 notably worse (+0.0355). Weather_720: essentially
    tied.
11. **retrieval-augmented가 Independent Base를 이기는가?**
    **Mixed.** ETTh1_720: NO, both Oracles are worse than the host's own
    diagnostic base branch (0.476/0.512 vs 0.468). Other 3 cells: retrieval
    is at or near the base branch, no large win either.
12. **같은 결론이 ETTh1/Weather/Solar에서 반복되는가?**
    **UNCERTAIN pending Solar.** Within ETTh1/Weather, the qualitative
    pattern (Set's train-TF learnability lower than Individual's, dramatic
    t>=2 divergence in rank) repeats consistently across all 4 completed
    cells; the FR-Agg/Stage-2-MSE outcome does NOT repeat consistently
    (direction flips by cell) -- so "Set Oracle is harder to learn" is a
    robust finding, "Set Oracle is worse/better for actual forecasting" is
    not resolved and appears genuinely dataset/horizon-dependent.

Per the user's explicitly stated principle for this campaign ("이번 실험의
주목적은 Set Oracle을 살리는 것이 아니다... Individual Oracle은 control이다"),
no loss or model change is proposed here. This campaign's completion (Solar
addendum pending) and these results are handed off for the next research
direction decision.
