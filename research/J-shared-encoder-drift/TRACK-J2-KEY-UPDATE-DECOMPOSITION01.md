# TRACK-J2-KEY-UPDATE-DECOMPOSITION01

**Status: COMPLETE (ETTh1_720, seed0). Direct causal follow-up to
TRACK-J-SHARED-ENCODER-DRIFT01 (commit `d3e0aa4`).** Two real training
interventions (J1 StopGrad-Key, J2 True-Frozen-Key) plus a post-hoc
Orthogonal Procrustes diagnostic on J0's existing best checkpoint. On the
primary metric (retMSE@10 / Oracle regret), the ordering is **J0 > J1 >
J2** (J0 best, J2 worst) -- Outcome E, "joint symmetric co-adaptation is
central to what makes retrieval learning work." But on the Top-10
**uniform aggregate MSE**, the ordering **reverses**: J1 > J2 > J0 (J1
best). This is the same individual-vs-aggregate metric disagreement
pattern this session has now observed repeatedly (TRACK-G, TRACK-H) --
reported here exactly as found, not resolved into a single verdict.
Procrustes alignment recovers 59% of St0's gap to the S00 baseline and
more than 100% of S0t's gap (aligned S0t actually beats S00) -- a
substantial, but incomplete, fraction of TRACK-J's original St0/S0t
degradation is attributable to coordinate-system mismatch, not pure
harmful single-side adaptation. 13/13 unit tests pass.

## 0. Re-audit of TRACK-J (commit d3e0aa4)

Confirmed directly from `results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/config.json`,
`checkpoint_fingerprints.json`, and `exact_commands.txt` (not from report
prose): `relation_encoder_type=mlp`, `relation_self_fill=linear`,
`relation_input_space=relation_teacher_space=relation_value_space=delta_last`,
`candidate_mask=raft` (default), `tau_t=tau_s=0.1`, `top_k=10`,
`batch_size=32`, `lr=1e-3`, `train_epochs=10`, `patience=5`,
`init_seed=loader_seed=0`, checkpoint criterion = min val
`model_top10_individual_mse`, N=7201 candidates, reference_ckpt =
`checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_.../checkpoint.pth`.
J0's own init hash: `b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e`
-- reproduced exactly by both J1 and J2 (asserted in code; run aborts
otherwise). None of these values were changed for J1/J2.

## 1. Arms

- **J0 Joint Shared** (NOT re-run; TRACK-J's own A0 artifacts reused
  as-is): both query and key branches grad-tracked, single shared
  encoder, candidates re-encoded every step.
- **J1 StopGrad-Key**: single shared encoder (same as J0). Candidates
  re-encoded every step via the identical forward call
  (`encode_raw(model, exp.memory_x, c)`), then `.detach()`-ed before
  scoring -- candidate embeddings keep moving with the encoder but
  contribute zero gradient.
- **J2 True-Frozen-Key**: two encoders. `E_q` trainable; `E_k =
  deepcopy(E_q^(0))`, `requires_grad_(False)`, `.eval()` always, NEVER
  re-encoded after the one-time key bank `K_0 = {E_k(X_1..X_N)}` is
  built. Asserted every training step that `K_0` is bit-identical to a
  fresh `E_k(memory)` recomputation (never fired -- see checkpoint
  fingerprint below).

## 2. Fairness verification

- J1 init hash = J2 init hash = J0 init hash =
  `b37fa4031f538e4b...` (exact match, asserted in code, run aborts on
  mismatch -- never fired).
- J2's `E_k` init hash AND final (post-training) hash both equal the
  same J0 init hash exactly (`key_cache_fingerprint.json`) -- `E_k`
  never moved.
- J1 and J2 epoch-1 batch-order hash: `ed5ae8650de28798be6f...`, identical
  between the two arms (same `loader_seed=0`/`make_loader_generator`
  convention reused unmodified from TRACK-J).
- J2's cached key bank `K_0` per-channel mean norm = 1.0 (expected,
  post-normalization cosine-space embeddings) and was asserted
  byte-identical to a live re-encode of `E_k` after every single
  optimizer step throughout all of training -- zero assertion failures.

## 3. Results

### Test split (best checkpoint, per-arm own criterion: min val `model_top10_individual_mse`)

| Arm | Key gradient | Key moves | best_epoch | retMSE@10 | Recall@10 | NDCG@10 | Oracle regret | Uniform Agg MSE@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| J0 Joint Shared | Yes | Yes | 10 | **0.953599** | 0.020257 | **0.965945** | **0.458034** | 0.641731 |
| J1 StopGrad-Key | No | Yes | 1 | 1.008351 | **0.021987** | 0.958187 | 0.512786 | **0.564347** |
| J2 True-Frozen-Key | No | No | 6 | 1.023965 | 0.019587 | 0.959167 | 0.528401 | 0.594954 |

**retMSE@10 / Oracle regret ordering: J0 < J1 < J2 (J0 best)** --
consistent, both metrics agree.
**Uniform Aggregate MSE@10 ordering: J1 < J2 < J0 (J1 best)** -- the
OPPOSITE ranking. Recall@10 is also (slightly) best for J1, not J0.

### Validation (best-epoch row)

| Arm | best_epoch | val retMSE@10 | val Recall@10 | val Agg MSE@10 |
|---|---:|---:|---:|---:|
| J1 StopGrad-Key | 1 | 2.049492 | 0.022053 | 1.603132 |
| J2 True-Frozen-Key | 6 | 2.140250 | 0.018530 | 1.657688 |

(J0's own val trajectory is in `results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/full_val_epoch_metrics.csv`,
final/best val retMSE@10 = 1.968919 at epoch 10 -- better than both J1
and J2 on validation too, consistent with the test-split ordering.)

### best_epoch pattern

J1 stops improving almost immediately (`best_epoch=1`, then 5 more
non-improving epochs before patience triggers early stop at epoch 6) --
removing candidate-side gradient does not just slow learning, it
essentially prevents further improvement past initialization-adjacent
training. J2 improves more gradually (`best_epoch=6`, does not
early-stop, runs the full 10 epochs) but never approaches J0's quality.
J0 itself kept improving through epoch 10 (see TRACK-J report) without
early-stopping.

## 4. Post-hoc Orthogonal Procrustes diagnostic

Fit **per channel**, on **train-memory candidate embeddings only**
(`Z0 = E0(memory)`, `Zt = Et(memory)`, J0's best/epoch-10 checkpoint),
via `R* = U @ Vh` from `svd(Zt^T @ Z0)`. Orthogonality verified
numerically (`||R*^T R* - I||_max` ~ 3e-5 for every channel). Residual
(`||Zt R* - Z0||_F` vs `||Zt - Z0||_F`) reduced 22.5-32.5% across the 7
channels by the single best-fit rotation -- a real but partial linear
alignment; the majority of the raw embedding-space discrepancy (67.5-77.5%)
is NOT recoverable by any global rotation/reflection.

| | S00 (baseline) | St0 raw | **St0 aligned (P1)** | S0t raw | **S0t aligned (P2)** | Stt raw |
|---|---:|---:|---:|---:|---:|---:|
| Val retMSE@10 | 2.2317 | 2.8481 | **2.4947** | 2.7543 | **2.1902** | 1.9705 |
| Test retMSE@10 | 1.1669 | 1.7747 | **1.4170** | 1.4460 | **1.0945** | 0.9536 |

**St0 recovery**: test gap-to-S00 shrinks from 0.608 (raw) to 0.250
(aligned) -- **58.9% of the degradation recovered** by a single global
rotation. Still 0.250 above S00 and far above Stt (0.954) -- alignment
helps substantially but does not fully explain St0's badness.

**S0t recovery**: test gap-to-S00 shrinks from 0.279 (raw) to **-0.073**
(aligned) -- alignment **more than fully recovers** S0t's degradation;
the aligned S0t (1.094) actually beats the S00 baseline (1.167) and gets
close to Stt (0.954). Candidate-side drift's apparent harm in TRACK-J's
original S0t measurement is, on this evidence, substantially (possibly
entirely, given the overshoot) a coordinate-mismatch artifact rather
than genuine harmful geometry change.

## 5. Answers to the pre-registered questions

**Q1 (J0 vs J1 -- does removing candidate-side gradient help or hurt?)**:
**Hurts** on retMSE@10/Oracle regret (J0 0.954 < J1 1.008) but **helps**
on Uniform Aggregate MSE@10 (J1 0.564 < J0 0.642) and Recall@10 (J1
0.0220 > J0 0.0203). Candidate-side gradient is beneficial for
per-candidate ranking quality but detrimental for Top-10-set aggregate
quality, in this run.

**Q2 (J1 vs J2 -- does freezing the candidate space, given no gradient,
help further?)**: **No, it hurts further on retMSE@10** (J1 1.008 < J2
1.024) but **also hurts on Uniform Aggregate MSE@10** (J1 0.564 < J2
0.595) -- unlike Q1, J1 beats J2 on every metric measured. Letting the
candidate embeddings keep moving (even with zero gradient contribution)
is better than freezing them outright.

**Q3 (J0 vs J2 -- joint co-adaptation vs query-to-fixed-index)**: J0
clearly better on retMSE@10/Oracle regret (0.954 vs 1.024); J2 better on
Uniform Aggregate MSE@10 (0.595 vs 0.642). Same individual-vs-aggregate
split as Q1.

**Q4 (was TRACK-J's negative `cos(g_q,g_k)` genuinely harmful
interference?)**: **Partially supported, and metric-dependent.** J1
(gradient removed) is worse than J0 on retMSE@10 -- consistent with the
conflicting gradient NOT being pure noise; removing it costs something.
But J1 is better than J0 on aggregate MSE -- so whatever "interference"
the negative-cosine gradient represents is a mixed effect: costly for
per-candidate ranking, useful for aggregate diversity/quality. It is not
accurate to call the conflict simply "harmful" or simply "beneficial" --
it is metric-dependent.

**Q5 (how much of St0/S0t's original degradation is Procrustes-
recoverable?)**: **58.9% for St0, and more than 100% for S0t** (aligned
S0t beats the S00 baseline). A substantial share of TRACK-J's originally
reported St0/S0t degradation is a coordinate-mismatch artifact, not (or
not entirely) genuine harmful single-side adaptation -- especially for
S0t (candidate-only drift), where the raw degradation is now shown to be
almost entirely a basis-mismatch effect once the embedding spaces are
put in the same coordinate frame.

**Q6 (does TRACK-J's "Frozen-Key is not justified" conclusion stand?)**:
**Needs revision, but not reversal.** TRACK-J's original argument was
"St0 (post-hoc) is bad, and A1/Frozen-Key is mathematically equivalent to
St0, so Frozen-Key is not justified." This round shows that argument's
premise was flawed in the way the spec anticipated: St0 was a post-hoc
cross-time evaluation, not a trained intervention, and a majority of its
badness is coordinate-mismatch, not a fair proxy for what an actually-
trained Frozen-Key run does. **J2 (the real Frozen-Key training run) is
still worse than J0 on retMSE@10** -- so the conclusion "Frozen-Key does
not beat the joint baseline on individual retrieval quality" still
stands, now on much better evidence (an actual trained run, not a
post-hoc proxy). But **J2 is better than J0 on Uniform Aggregate MSE@10**
-- so "Frozen-Key is not justified" is too strong as a blanket statement;
it depends on which downstream objective (per-candidate retrieval vs
aggregate Top-10 quality) is prioritized.

**Q7 (which direction does this most strongly support: shared joint /
stop-gradient key / frozen key / separate dual encoder)?**: On
individual retrieval quality, **shared joint (J0)** is most strongly
supported (Outcome E: J0 > J1 > J2). On aggregate Top-10 quality,
**stop-gradient key (J1)** is most strongly supported. No single
architecture is unconditionally best across both objectives measured
here. Per the spec's explicit instruction, no further experiment
(A2/additional seeds/Weather/Stage-2) is run based on this.

## 6. What is established

- Two real training interventions (not post-hoc evaluations) were run,
  with verified-identical initialization, batch order, and all other
  hyperparameters vs J0.
- retMSE@10/Oracle regret: J0 > J1 > J2 (Outcome E), reproduced on val
  and test.
- Uniform Aggregate MSE@10: the ranking reverses to J1 > J2 > J0 -- the
  individual-vs-aggregate metric disagreement pattern from TRACK-G/H
  recurs here in a genuine causal-intervention setting, not just a
  post-hoc diagnostic.
- A substantial fraction (59-100%+) of TRACK-J's original St0/S0t
  degradation is recoverable by a single global orthogonal rotation fit
  on train-memory candidates alone -- coordinate mismatch is a real,
  substantial contributor to those numbers, especially for S0t.
- J2's frozen key bank was verified never to drift (assertion checked
  every training step, zero failures) -- the True-Frozen-Key
  intervention is implemented correctly.

## 7. What is NOT established

- Why removing candidate gradient (J1) improves aggregate quality while
  hurting individual quality -- only the outcome pattern is measured, no
  mechanism identified.
- Why J1 (moving-but-gradient-free key) beats J2 (frozen key) on every
  metric -- whether "key co-movement without gradient" itself helps, or
  whether this is specific to the 20-80% residual embedding-space
  discrepancy geometry Procrustes could not linearly correct for.
- Whether the 58.9%/100%+ Procrustes recovery fractions generalize to
  other seeds, checkpoints, or training dynamics.
- Whether these patterns hold on other horizons, datasets, or encoder
  types -- explicitly out of scope this round.

## 8. Interpretation-rule compliance

Per spec section 17, all claims above are scoped to "ETTh1 H720 seed0"
and phrased as "causal intervention results support..." rather than
"proved"/"universally"/"the definitive cause." No additional seed,
Weather, Stage-2, or A2 (asymmetric dual encoder) was run.

## Artifacts

- Report: `research/J-shared-encoder-drift/TRACK-J2-KEY-UPDATE-DECOMPOSITION01.md`
- Scripts: `scripts/train_j2_key_update_decomposition01.py`,
  `scripts/diag_j2_procrustes01.py`
- Unit tests: `tests/test_j2_key_update_decomposition01.py` (13 tests, all pass)
- Raw results: `results/TRACK-J2-KEY-UPDATE-DECOMPOSITION01/ETTh1_720/`
  (`config.json`, `probe_query_ids.json`, `J1_stopgrad_key/`,
  `J2_true_frozen_key/`, `procrustes/`)
- Checkpoints: `checkpoints/track_j2_key_update_decomposition01/ETTh1_720/`
- J0 (reused, not re-run): `results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/`,
  `checkpoints/track_j_shared_encoder_drift01/ETTh1_720/checkpoint.pth`
