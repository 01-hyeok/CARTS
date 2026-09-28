# TRACK-G-DECOUPLED-METRIC-ADAPTATION01

**Status: COMPLETE. Verdict: PARTIAL GO / INCONCLUSIVE on the decoupling
hypothesis, GO on metric adaptation itself.** A frozen p120 trunk + a
fresh, identity-init asymmetric metric head (A1) beats the frozen-cosine
baseline (A0) decisively and beats continuing to train the whole encoder
with no metric head (A4) decisively — metric adaptation on top of an
already-trained representation is real and large. But the central
question this round was commissioned to answer — **does decoupling
(freeze trunk, adapt metric only) beat joint optimization (train trunk +
metric together, A5)?** — resolves the OPPOSITE of the user's stated
"most wanted" result: **A5 beats A1 on individual retMSE@10, in all 3
loader-order replications, with cluster-bootstrap 95% CI excluding zero
in A5's favor every time.** On aggregate (Top-10 mean-prediction) MSE,
however, the ranking flips: **A1 beats A5 in all 3 replications**, CI
excluding zero in A1's favor every time. The two required metrics
disagree on which arm is "best," exactly as the spec warned they might.
Per the spec's explicit instruction, this report does **not** conclude
"decoupled training is better" — the A1-vs-A5 comparison is not settled
in one direction; it is metric-dependent.

## 1. Correction carried over from CONTROL01

`train_patch_retrieval_expert01.py` (p120's own training script) puts
`model.parameters()` — the whole encoder, `norm`/`proj` included — into
the Adam optimizer (`p.requires_grad_(True)` for all params, then
`Adam(model.parameters(), ...)`), confirmed by direct code read this
round. CONTROL01's report framing ("bypassing a never-retrained
norm→proj bottleneck") was factually wrong on this point. This round's
question is reframed per the user's correction: *does a frozen,
already-fully-trained temporal representation benefit from a second,
decoupled metric-adaptation stage, and if so, why?*

## 2. Six arms, same p120 checkpoint

All six arms load `checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth`
(`patch_len=stride=120`, `num_patches=6`, `d_model=128`, `tau_t=0.02`,
`tau_s=0.1`, `top_k=10`).

| Arm | Trunk | Representation | Metric | Trained params |
|---|---|---|---|---|
| A0 Frozen-Final-Cosine | frozen | existing `forward()` final embedding | cosine | none (eval-only) |
| A1 Frozen-Final+FreshAsym | frozen | existing `forward()` final embedding | fresh identity-init Wq/Wk | Wq/Wk only |
| A2 Frozen-RawCLS+FreshAsym | frozen | raw pre-norm/proj CLS | fresh identity-init Wq/Wk | Wq/Wk only (= CONTROL01 B0) |
| A3 Frozen-PatchMean+FreshAsym | frozen | patch tokens, projected + mean-pooled | fresh identity-init Wq/Wk | Wq/Wk only (= CONTROL01 B1) |
| A4 Continue-Encoder-NoHead | trainable | existing `forward()` final embedding | cosine | whole encoder |
| A5 Joint-Encoder+FreshAsym | trainable | existing `forward()` final embedding | fresh identity-init Wq/Wk | whole encoder + Wq/Wk |

3 loader-order replications each (`loader_seed` ∈ {0,1,2}; only training
batch order varies — all heads are identity-initialized, deterministic,
no model-init randomness). 18 runs total, one sequential batch, GPU 1.

## 3. Fairness / reproducibility checks

- **Baseline reproduction**: A0 test retMSE@10 = 0.969165454248431 in
  all 3 seeds, matching F0-Pooled to ~1e-10 (spec required 1e-5). PASS.
- **A2/A3 reproduce CONTROL01's B0/B1 exactly**: A2 = [0.814926,
  0.840332, 0.808579] vs CONTROL01 B0 = [0.814926, 0.840332, 0.808579];
  A3 = [0.849665, 0.821818, 0.834813] vs CONTROL01 B1 = [0.849665,
  0.821818, 0.834813]. Byte-identical. PASS.
- **Step-0 identity-init equivalence**: A1 at initialization (Wq=Wk=I)
  reproduces A0's cosine score exactly on a live batch — max abs score
  diff = 0.0, Top-10 sets 100% equal. PASS.
- **Batch-order equality across arms**: `batch_order_sha256` for
  epoch 1 identical across all 5 trained arms at each seed (3/3 seeds
  checked, single hash per seed). PASS.
- **Frozen-trunk no-grad**: `train_epoch` asserts every frozen-arm
  parameter has `grad is None` after every backward call; the assertion
  never fired across A0–A3 × 3 seeds. PASS.
- **Frozen-trunk immutability**: `state_sha(model.state_dict())` compared
  before/after training, asserted equal for every frozen-arm run; held
  in all cases (script aborts on failure, none aborted). PASS.
- **A4/A5 encoder actually moved**: `encoder_param_displacement`
  (L2 over all trunk params, checkpoint vs best-epoch reload) is 0.0 for
  every frozen arm and 89.6–114.2 for every A4/A5 run — confirms A4/A5's
  optimizer touched the trunk and A0–A3's did not. PASS.

## 4. Results (test split, 3-seed mean; ETTh1_720, p120)

| Arm | retMSE@10 | Δ vs A0 | Uniform Agg-MSE@10 | Recall@10 | Binary-Oracle NDCG@10 | Oracle mean rank |
|---|---|---|---|---|---|---|
| A0 | 0.969165 | 0% | 0.569328 | 0.0293 | 0.0293 | 734.8 |
| A1 | 0.849041 | -12.39% | 0.539882 | 0.0592 | 0.0614 | 746.3 |
| A2 | 0.821279 | -15.26% | 0.534539 | 0.0771 | 0.0810 | 675.6 |
| A3 | 0.835432 | -13.80% | **0.520942** | 0.0862 | 0.0904 | 761.2 |
| A4 | 0.968038 | -0.12% | 0.597523 | 0.0267 | 0.0268 | 770.5 |
| A5 | **0.816703** | **-15.75%** | 0.574880 | 0.0732 | 0.0765 | 640.1 |

(retMSE@10, Recall@10, NDCG@10, oracle rank all computed over the FULL
candidate memory, N=7201, no shortlist.)

**Key disagreement**: A5 has the single best individual retMSE@10
(-15.75%) and the best oracle mean rank (640.1, lowest = best), but its
Top-10 uniform-aggregate MSE (0.574880) is *worse than the untrained A0
baseline* (0.569328) and clearly worse than A1/A2/A3. A3, conversely, has
mediocre individual retMSE (3rd of 5 trained arms) but the best aggregate
MSE by a clear margin. Individual-candidate quality and aggregate
prediction quality are not the same objective here.

## 5. Bootstrap analysis (cluster bootstrap, query_start_idx × 7 channels jointly, 10k reps)

**On individual retMSE@10** (diff = A − B; negative favors A):

| Comparison | seed0 | seed1 | seed2 | All 3 significant, same direction? |
|---|---|---|---|---|
| A0 − A1 | +0.1336 [0.128,0.139] | +0.1238 [0.118,0.129] | +0.1029 [0.097,0.108] | Yes — A1 beats A0 |
| A1 − A4 | -0.1224 [-0.129,-0.116] | -0.1264 [-0.132,-0.121] | -0.1082 [-0.114,-0.102] | Yes — A1 beats A4 |
| A1 − A2 | +0.0207 [0.017,0.024] | +0.0050 [0.001,0.009] | +0.0577 [0.052,0.063] | Yes — A2 beats A1 |
| A1 − A3 | -0.0141 [-0.018,-0.010] | +0.0235 [0.019,0.028] | +0.0314 [0.026,0.037] | **No — seed0 favors A1, seed1/seed2 favor A3** |
| A1 − A5 | +0.0147 [0.010,0.019] | +0.0091 [0.004,0.014] | +0.0733 [0.067,0.079] | Yes — **A5 beats A1**, all 3 seeds, CI excludes zero every time |

**On uniform aggregate MSE@10** (same diff convention):

| Comparison | seed0 | seed1 | seed2 | All 3 significant, same direction? |
|---|---|---|---|---|
| A0 − A1 | +0.0310 [0.027,0.035] | +0.0287 [0.025,0.032] | +0.0286 [0.025,0.032] | Yes — A1 beats A0 |
| A1 − A4 | -0.0664 [-0.071,-0.062] | -0.0551 [-0.059,-0.051] | -0.0514 [-0.055,-0.047] | Yes — A1 beats A4 |
| A1 − A3 | +0.0194 [0.016,0.023] | +0.0194 [0.016,0.023] | +0.0181 [0.014,0.022] | Yes — A3 beats A1 |
| A1 − A5 | **-0.0248 [-0.029,-0.021]** | **-0.0526 [-0.056,-0.049]** | **-0.0276 [-0.032,-0.023]** | Yes — **A1 beats A5**, all 3 seeds, CI excludes zero every time — **reverses the retMSE result** |

Full per-seed cluster and moving-block (96/240/720) bootstrap results:
`results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720/bootstrap_summary.json`.

## 6. Case-framework interpretation

- **Case 1 (A1 ≫ A4, metric adaptation matters more than more encoder
  training)**: **SUPPORTED.** A1 beats A4 by ~12 percentage points on
  retMSE and by a wide, significant margin on aggregate MSE, in every
  seed. Continuing to train the whole encoder with no metric head barely
  moves the needle (-0.12%) and is not reliably better than not training
  at all (2/3 A4 seeds are numerically worse than A0).
- **Case 2 (A1 > A5, decoupling beats joint optimization — the "most
  wanted" outcome)**: **REJECTED on individual retMSE**, **SUPPORTED on
  aggregate MSE.** The two pre-registered primary metrics point in
  opposite directions, both with non-overlapping bootstrap CIs in all 3
  seeds. This is not sampling noise on either metric individually — it is
  a genuine metric-objective conflict: A5's jointly-trained
  representation is measurably better at ranking the single best
  neighbor per query, but measurably worse at producing a spread of
  Top-10 candidates whose plain average is a good forecast.
- **Case 5 (A1 ≈ A5, decoupling hypothesis weakens)**: not a clean fit
  either — the two metrics don't converge to "roughly equal," they
  diverge to opposite significant verdicts.
- A1 vs A2/A3 (does *which* frozen representation the metric head sits on
  matter): A2 (raw CLS) beats A1 (final embedding) on retMSE in all 3
  seeds; A3 (patch-mean) is mixed against A1 on retMSE (1 seed favors A1,
  2 favor A3) but beats A1 on aggregate MSE in all 3 seeds. The specific
  frozen representation chosen for the metric head measurably affects
  outcome, and raw/patch-token representations are consistently
  competitive with or better than the model's own final `forward()`
  output for this purpose.

## 7. What is established

- A second, decoupled metric-adaptation stage on top of a frozen,
  already-fully-trained p120 trunk produces a large, robust improvement
  over that trunk's own cosine retrieval (A1, A2, A3 all beat A0 by
  12–15%, every seed, both metrics, bootstrap CI excluding zero).
- This adaptation-stage gain is not explained by "the norm/proj path was
  never trained" (that framing was factually wrong, per §1) — the trunk
  including norm/proj WAS trained end-to-end originally, and the gain
  still appears on top of it.
- Metric adaptation clearly beats plain continued encoder training with
  no metric head (A1 ≫ A4, both metrics, all seeds).
- Individual-candidate retrieval quality (retMSE@10, oracle rank) and
  aggregate Top-10 prediction quality (uniform-aggregate MSE) are
  measurably different objectives on this data: the arm that is best on
  one (A5 on retMSE) is not the arm that is best on the other (A3/A1 on
  aggregate MSE), and this disagreement is bootstrap-significant, not
  noise.

## 8. What is NOT established

- Whether decoupled (frozen-then-adapt) training is unconditionally
  "better" than joint training — it depends on which downstream metric
  is prioritized (per-candidate retrieval vs aggregate prediction), and
  this round does not adjudicate which metric the eventual Stage-2
  forecasting pipeline actually cares about.
- Why A5's representation trades per-candidate ranking quality for
  aggregate quality — no mechanism (e.g., reduced Top-10 diversity,
  score-distribution collapse) is directly measured this round; only the
  outcome is measured.
- The effective-rank / mean-pairwise-cosine / per-epoch parameter
  displacement diagnostics requested in the original spec were **not**
  logged during the 18 runs (only final-epoch scalar
  `encoder_param_displacement` and a coarser per-epoch curve of
  train_loss/val_retMSE/val_uniform_agg/val_recall/val_oracle_mean_rank
  were captured — see `epoch_curve_{arm}_seed{seed}.csv`). Re-running to
  add representation-collapse diagnostics would require a fresh 18-run
  batch and is out of scope for this report without separate approval.
- Weather, other horizons, Stage-2 integration, new losses, temperature
  or patch-size sweeps, new architectures — all explicitly out of scope
  per the spec and not touched.

## 9. Alternative explanations considered

- **A5's aggregate-MSE weakness could be a Top-10 diversity collapse**
  (jointly-trained representation clusters candidates tightly around the
  single best match, so the mean of the 10 nearest is a worse forecast
  than a more diverse set) — plausible given A5 has both the best oracle
  mean rank (best individual matches) and the worst aggregate MSE among
  the metric-adapted arms, but not directly measured (see §8).
- **Overfitting of the joint encoder to the retrieval objective** at the
  expense of the representation's forecasting-relevant structure — also
  plausible (A4, which trains the encoder without a metric head, shows
  *no* aggregate-MSE improvement either, ruling out "any extra encoder
  training helps aggregate MSE" but not ruling this out for A5
  specifically).
- **Simple metric-specific noise given only 3 loader-order
  replications** — considered but bootstrap CIs are non-overlapping and
  computed within-seed over 2161×7 queries with 10k cluster resamples;
  the direction is consistent across all 3 independent loader orders for
  both metrics (just in opposite directions from each other), which is
  hard to attribute to noise.

## 10. Code / protocol issues found this round

- None beyond the pre-existing A0-empty-optimizer bug (fixed before the
  full run; see script docstring / commit) and the CONTROL01 framing
  error already documented in §1.

## 11. GO / NO-GO / INCONCLUSIVE

- **Metric adaptation on a frozen trunk (the general finding)**: **GO.**
  Robust, large, bootstrap-significant improvement over the trunk's own
  cosine retrieval, confirmed on 3 independent frozen representations
  (A1/A2/A3) and 2 independent metrics.
- **Decoupling vs joint optimization (A1 vs A5, the round's central
  question)**: **INCONCLUSIVE — metric-dependent.** Individual retMSE
  favors joint (A5); aggregate MSE favors decoupled (A1). Per the spec's
  explicit instruction, this report does not declare "decoupled training
  is better" — that claim is not supported when aggregate and individual
  metrics disagree this cleanly. The next research decision (which
  metric the eventual system should optimize for, and/or a mechanism
  study of A5's aggregate-MSE weakness) is left to the reviewer per
  standard workflow, not decided here.
- Per the spec's explicit prohibition, no Weather / other horizon /
  Stage-2 / new-loss / architecture experiments were started this round
  regardless of these results.

## 12. Artifacts

- Script: `scripts/train_g_decoupled_metric_adaptation01.py`
- Bootstrap analysis: `scripts/analyze_g_bootstrap01.py`
- Raw results: `results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720/`
  (`{arm}_seed{seed}_metrics.json`, `per_query_{arm}_seed{seed}.csv`,
  `epoch_curve_{arm}_seed{seed}.csv` × 18, plus `bootstrap_summary.json`)
- Checkpoints: `checkpoints/track_g_decoupled_metric_adaptation01/`
