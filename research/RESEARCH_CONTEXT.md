# RESEARCH_CONTEXT.md

Stable research context for CARTS (Cross-channel Aligned Retrieval for Time Series).

Scope rule: this file contains only statements verifiable from this repository
(source code, `README.md`, `RESULTS_SUMMARY.md`, `metrics/`, `logs/`). Anything
not verifiable is listed under **Unknown / Unverified**.

Last verified against the repository: 2026-09-02 (branch `main`, HEAD `c306def`).

---

## Verified Implementation Facts

### Research objective and problem formulation

Source: `Retrieving_Target_Futures_Cross_Channel_Contexts.docx` (repository root,
Korean manuscript draft). Read on 2026-09-02; summarized here in English.

**Objective.** For multivariate time-series forecasting, predict a *target
channel's* future by retrieving *cross-channel context* from the past. Instead of
retrieving past windows that merely look like the current target-channel window,
retrieve past situations whose full multi-channel context resembled the current
one, and use what actually followed for the target channel as the reference.

**Formulation.** With target channel `c` and source channel `j`, lookback `L`,
horizon `H`:

```
Q_{c<-j}(t)   = [ X^c_{t-L+1:t} , X^j_{t-L+1:t} ]     query   (past only)
K_{c<-j}(tau) = [ X^c_{tau-L+1:tau} , X^j_{tau-L+1:tau} ]   key (past only)
V_{c<-j}(tau) =   X^c_{tau+1:tau+H}                    value  (target future)
```

The value is **always** the target channel's historical future, never a source
channel's future — this is what the manuscript calls the leakage-free property.
Query/key raw dimension (multi-channel, ~`kL`) and value dimension (`H`) differ by
construction; query and key are mapped by encoders `f_q`, `f_k` into a shared
`d`-dimensional retrieval space, and similarity is computed there (Appendix A).

**Prediction assembly.**

```
Y^ret_{c<-r} = sum_i alpha_i^r V_i^r          (top-K weighted sum per relation r)
Y^ret_c      = sum_r beta_r Y^ret_{c<-r}      (relation mixer over source channels)
Y_c          = Y^base_c + lambda_c Y^ret_c    (gate over base forecaster)
```

This maps onto the implemented `alpha_*` (within-branch top-k weights), `beta_*`
(relation-branch mixer), and `lambda` (gate) diagnostics listed further below.

**Future-aware retriever.** During training the true target future `Y_c(t)` is
known, so the retriever is supervised to make the retrieved value close to it
(`V(tau) ~ Y_c(t)`), not merely to make past shapes match. The manuscript
proposes hard negatives whose target-channel past matches but whose source-channel
conditions differ, so that the futures diverge.

**Scalability.** With many channels, a target-specific sparse relation graph
`N(c) = Top-M source channels for c` limits retrieval. The manuscript's stated
differentiator is selecting those source channels by a *future-aware* criterion
rather than plain correlation. Implemented via `--source_mode`, `--relation_top_n`,
`--relation_graph_path` (`metrics/relation_graphs/` holds generated graphs).

**Claimed contributions (manuscript §7).**

1. Leakage-free retrieval structure: multi-channel context as query/key, target
   channel's historical future as value.
2. A future-aware retriever that finds references useful for prediction rather
   than merely similar in the past.
3. Target-specific sparse relation graph for high channel counts.
4. Interpretability: relation-wise retrieval + adaptive mixer expose which source
   channel contributes to each target.

Note: these are the manuscript's *claims*. Whether the experiments in this
repository support them is a reviewer question, not a settled fact — see
**Established Experimental Findings** and **Open Questions**.

### Project identity

- CARTS is a relation-retrieval time-series forecasting project built on
  [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and adapted
  from RAFT-style retrieval forecasting (`README.md`).
- Entry point: `run.py`. Task selected by `--task_name` ∈
  `long_term_forecast` | `stage1_relation` | `stage2_relation` (`run.py:64`,
  dispatch at the bottom of `run.py`).
- Orchestration classes: `Exp_Long_Term_Forecast`, `Exp_Stage1_Relation`
  (`exp/exp_stage1_relation.py`), `Exp_Stage2_Relation`
  (`exp/exp_stage2_relation.py`).

### Two-stage architecture

- **Stage 1** (`models/RelationStage1.py`, `exp/exp_stage1_relation.py`) trains a
  *relation encoder* that embeds windows so that retrieval by embedding
  similarity selects useful neighbors. Encoder types are selectable via
  `--relation_encoder_type` (`transformer` default, `tcn`, and a Chronos-based
  encoder in `models/ChronosRelationEncoder.py`). Pooling via
  `--relation_pooling` (`cls` default; `tcn` requires `last` or `mean`, enforced
  in `run.py`).
- **Stage 2** (`models/RelationStage2.py`, `exp/exp_stage2_relation.py`) freezes
  or fine-tunes the Stage-1 encoder (`--freeze_stage1_encoder`, default `1`;
  `--stage1_ckpt_path`, `--stage1_encoder_init`), retrieves top-k neighbors from a
  precomputed memory bank, and fuses a base forecast with a retrieval forecast.
- Fusion: `--fusion_mode` (default `raft_concat`), `--stage2_relation_fusion`
  (default `gate`), gate in `layers/retrieval_gate.py`, `--gate_mode` (default
  `scalar`), optional `--fixed_lambda`.
- End-to-end mode: `--stage2_e2e`, `--stage2_e2e_full_online`.
- Memory: `--memory_cache_mode` is restricted to `precompute`; `run.py` raises
  `NotImplementedError` otherwise. `--refresh_memory_every_epoch` rebuilds the key
  bank; a trainable Chronos retrieval space requires it (checked in
  `Exp_Stage2_Relation.train`).

### Query / candidate construction and retrieval

- Cross-channel relation structure: `--source_mode` (`auto` | `all` | `topk_corr`),
  `--relation_top_n` (default `3`), `--relation_graph_threshold`,
  `--relation_graph_path`, `--target_mode` (`all` | `single`), `--target_channel`.
- Candidate masking: `--candidate_mask` (default `raft`) — controls which memory
  entries a query may retrieve.
- Retrieval scoring: `--stage1_retrieval_metric` (default `cosine`),
  `--stage1_retrieval_score` (default `cosine`), `--retrieval_similarity`
  (default `cosine`), pairwise scorer in `layers/pairwise_scorer.py` and
  `models/utility_pair_scorer.py`.
- Top-k selection: `--top_k` (default `10`), soft weighting temperature
  `--tau_topk` (default `0.1`), `--retrieval_soft_all`.
- Representation spaces: `--relation_input_space`, `--relation_teacher_space`,
  `--relation_value_space` (all default `delta_last`).

### Loss functions (Stage 1)

Implemented in `utils/losses.py`, `utils/rank_losses.py`, and
`exp/exp_stage1_relation.py`; selected by CLI flags:

- Distillation / imitation: `--stage1_loss_mode` (default `kl`),
  `--stage1_teacher_mode` (default `mse`), `--stage1_teacher_target` (default
  `future`), `--stage1_teacher_loss` (default `kl`), `--tau_student` /
  `--tau_teacher` (default `0.1`), `--stage1_teacher_tau` (default `0.05`),
  EMA teacher (`--use_ema_teacher` default `1`, `--stage1_ema_momentum_base`
  `0.99` → `--stage1_ema_momentum_final` `0.9995`).
  `--relation_teacher_type` is a convenience alias mapping
  `future_mse`→`mse` and `ema`→`ema_input` (`run.py`).
- Ranking: `--rank_loss_weight`, `--rank_margin`, `--rank_mining_mode`,
  `--stage1_use_rank_loss`, `--stage1_rank_weight`, `--stage1_rank_margin`.
- Contrastive: `--stage1_infonce_weight` (default `0.5`),
  `--stage1_infonce_positive_source` (default `target_mse`), RnC flags
  (`--rnc_temperature`, `--rnc_quality_source`).
- Expected-MSE / set-level: `--expected_mse_weight`,
  `--stage1_expected_mse_lambda`, `--stage1_set_mse_weight`, `--stage1_set_tau`,
  `--stage1_set_support_k/_weight`.
- Anti-collapse regularizers: `--stage1_variance_weight`,
  `--stage1_covariance_weight`, `--stage1_variance_target`, with collapse
  diagnostics `--stage1_collapse_metrics` (default on).

### Loss functions (Stage 2)

- Forecast loss `--loss` (default `MSE`), auxiliary base/retrieval losses
  (`--use_aux_base_loss`, `--use_aux_ret_loss`), retrieval KL to a teacher
  (`--retrieval_kl_weight`, `--retrieval_kl_teacher` default `ema`), ranking
  losses (`--stage2_rank_loss`, `--stage2_rank_weight`, `--stage2_rank_margin`,
  hard/random negative counts), `--beta_entropy_reg`.

### Datasets

- Registered loaders (`data_provider/data_factory.py`): `ETTh1`, `ETTh2` →
  `Dataset_ETT_hour`; `ETTm1`, `ETTm2` → `Dataset_ETT_minute`; `custom` →
  `Dataset_Custom`; `Solar` → `Dataset_Solar`.
- Reported experiments in `RESULTS_SUMMARY.md` use **ETTh1** and **ETTm1**.
  Driver scripts also exist for `Weather` and `Electricity` (`scripts/`).
- Expected dataset path (`README.md`):
  `../Dataset/Time-Series-Library_dataset/ETT-small/` containing `ETTh1.csv`,
  `ETTm1.csv`.
- Splits (`data_provider/data_loader.py`): ETT hour uses fixed borders
  12/4/4 months (train/val/test); ETT minute uses the same borders ×4;
  `Dataset_Custom` uses 0.7 / (remainder) / 0.2 by row count. Scaler is fit on
  the train split only.

### Metrics and diagnostics

- Forecast metrics: MSE, MAE (`utils/metrics.py`), logged per split.
- Retrieval recall: `recall_at_k(scores, oracle_distance, valid_mask, ks=(1,5,10))`
  in `utils/retrieval_scoring.py`; surfaced in Stage-2 as
  `student_relation_oracle_recall_at_{1,5,10}` (and `_sc` per-source-channel
  variants) in `exp/exp_stage2_relation.py`.
- Gate / selection diagnostics logged by Stage 2 include `lambda_mean`,
  `beta_*` (relation-branch selection: `beta_best_relation_top1_match`,
  `beta_entropy`, `beta_regret_vs_best_relation`), `alpha_*` (within-branch top-k
  weighting: `alpha_entropy`, `alpha_top1_mean`, `alpha_margin_mean`),
  `relation_mse_*`, `retrieval_gain*`, `valid_candidate_fraction`.
- Outputs: TensorBoard under `--tensorboard_dir` (default `./runs`), CSVs under
  `--metrics_csv_dir` (default `./metrics`), e.g.
  `metrics/stage2/<data>/seq<L>_pred<H>/<setting>/metrics_main.csv`.
  Console lines are prefixed `Stage1 Test` / `Stage2 Test`.

### Reproducibility mechanics

- `--seed` (default `0`) seeds `random`, `numpy`, `torch`, `torch.cuda`, and sets
  `cudnn.deterministic=True`, `cudnn.benchmark=False` (`run.py`).
- Default learning rate depends on task when `--learning_rate` is unset:
  `1e-3` (stage1), `1e-2` (stage2), `1e-4` (otherwise).
- Experiment directory names are built by `build_experiment_setting()` and
  truncated with a sha256 suffix past 200 bytes.
- Stage-2 checkpoint selection: best validation loss, saved to
  `checkpoints/stage2/<data>/seq<L>_pred<H>/<setting>/checkpoint.pth`;
  `--patience` (default `5`) early stopping, `--train_epochs` default `10`.

---

## Verified Experimental Protocol

From `RESULTS_SUMMARY.md` (aggregation of 295 `Stage2 Test` log lines plus
`metrics/stage2/**/metrics_main.csv` test rows) and the driver scripts:

- Datasets: ETTh1, ETTm1.
- Features: multivariate (`--features M`).
- Protocol: `seq_len = pred_len`.
- Prediction horizons: 96, 192, 336, 720.
- `top_k = 10`, `relation_top_n = 3`.
- Model size used in the reported stage-2 runs: `d_model 128`, `n_heads 4`,
  `e_layers 2`, `d_layers 1`, `d_ff 256`, `label_len 0` (read from the setting
  strings under `metrics/stage2/`).
- Primary metrics: test MSE and MAE; retrieval quality via
  `student_relation_oracle_recall_at_k` (R@1 / R@5 / R@10).
- Upper bound reference: `full_oracle` runs (oracle candidate selection).
- Lower bound / control: `no-retrieval (base only)`.
- Consolidated table: `metrics/summary_test_all.csv`.
- Seed noise estimated at ≈0.01 MSE from a 3-seed `random_backbone` run
  (0.3886 / 0.3988 / 0.3999) — differences below 0.01 MSE are not interpretable.

---

## Established Experimental Findings

Provenance is marked on every block:
**[repo]** = recomputed or read directly from artifacts in this repository;
**[user]** = supplied by the user on 2026-09-02 and not independently reproduced here.

### A. Pre-campaign results (through 2026-08-04) — [repo] `RESULTS_SUMMARY.md`

1. Best variant `e2e_scratch λ=1.0 + future_mse teacher`, avg MSE 0.3828 vs
   no-retrieval 0.3981 (−3.8%).
2. Within a cell, Spearman ρ(R@10, MSE) is −0.954 (ETTh1/96) … −0.331 (ETTh1/720).
3. Absolute recall is very low: best R@1 = 1.55%, R@10 = 6.90% (ETTh1/96).
4. Horizon 720 differs: recall ≈1/3 of other cells; no retrieval variant beats
   no-retrieval on ETTh1/720.
5. `beta_top1_match` ≈ 35% vs R@1 ≈ 1%: branch selection works, temporal
   neighbor selection does not.
6. Encoder identity barely matters (identity / pearson / random within
   0.387–0.389 MSE).

> Note: finding 2 is *within-variant-set correlation on the pre-campaign sweep*.
> The 2026-09 campaign (section B) shows it does not hold as a causal or
> cross-condition claim. Do not cite 2 as evidence that raising recall raises
> forecast quality.

### B. 2026-09 diagnostic campaign — "why does Stage-1 recall gain not transfer?"

The campaign's framing question: **why does improving Stage-1 retrieval not
improve Stage-2 forecasting?** Findings in the order they were established.

**B1. Stage-1 Recall@10 is substantially improvable.** [user]
ETTh1 Stage-1 score/loss sweep, R@10:

| Arm | H96 | H192 | H336 | H720 | Avg |
|---|---|---|---|---|---|
| KL + Cosine | 0.0578 | 0.0556 | 0.0489 | 0.0216 | 0.0460 |
| WCE + Cosine | 0.0578 | 0.0501 | 0.0451 | 0.0200 | 0.0433 |
| KL + Asymmetric | 0.0592 | 0.0649 | 0.0684 | 0.0608 | 0.0633 |
| WCE + Asymmetric | 0.0544 | 0.0926 | 0.1221 | 0.0942 | 0.0908 |
| KL + MLP | 0.0561 | 0.0504 | 0.0684 | 0.0785 | 0.0634 |
| WCE + MLP | 0.0664 | 0.0953 | 0.1311 | 0.0939 | 0.0967 |

Gains are largest at H336/H720 (0.0489 → 0.1311, 0.0216 → 0.0939).

**B2. Recall change and Stage-2 MSE change do not agree.** [user]
Weather H96, after the wiring fix:

| Arm | R@10 | Stage-2 final MSE |
|---|---|---|
| Cosine + KL | 0.0583 | 0.1925 |
| Asymmetric + KL | 0.0044 | 0.1794 |

R@10 falls ~92.5% while forecast MSE *improves* ~6.8%.
**R@10 does not directly express downstream utility.**

**B3. "Stage-2 ignores retrieval" is rejected.** [user]
Retrieval gain under residual fusion `Y = Y_base + λ·Y_ret`:
ETTh1 H96 ≈ 40.05%, H720 ≈ 6.18%; Weather H96 ≈ 7.75%, H720 ≈ −1.39%.
Stage-2 does use retrieval, and scales its contribution with signal quality.

**B4. Aggregate quality, not candidate identity, tracks Stage-2.** [user]
Spearman against Stage-2 MSE:

| Metric | ETTh1 | Weather |
|---|---|---|
| Recall@10 | +0.032 | −0.350 |
| HardAggregateMSE@10 | **+0.810** | **+0.935** |

where `HardAggregateMSE@10 = MSE(mean_i y_i, y_q)` and
`IndividualFutureMSE@10 = (1/K)·Σ_i MSE(y_i, y_q)`.

**B5. Rank-only encoder training causes representation collapse.** [user]
Effective rank 16.38 (step 0) → 3.23 (step 5) → 1.90 (step 10) → 1.07 (epoch 1);
pairwise cosine ↑, sv1 fraction → ≈0.99, score separation ↓, gradients weaken.
Causal reading: **collapse → cosine saturation → gradient weakening**, not
gradient vanishing → collapse.

**B6. Removing collapse does not fix retrieval.** [user]
Frozen healthy WCE encoder, train only the asymmetric scorer:

| | Frozen Cosine | Frozen + Asym Rank |
|---|---|---|
| PairAcc100 | 0.53185 | 0.54605 |
| LargeGapPairAcc | 0.56134 | 0.58531 |
| MissedBetter | 76.57 | 71.65 |
| Recall@10 | 0.05775 | 0.02046 |
| Spearman | 0.44973 | 0.30047 |
| RetrievedMSE@10 | 0.68637 | 1.01100 |

Local pair ordering improves; global full-memory Top-K retrieval gets worse.
**Local pairwise ordering improvement ≠ global retrieval improvement.**

**B7. A global KL anchor preserves ordering but not Top-K utility.** [user]
`L = L_rank + β·KL(p_base ‖ p_new)`. Retention rises with β
(Top10: 0.142 → 0.344 → 0.716; Top100: 0.270 → 0.564 → 0.833), but retrieval
quality is not fixed. `cos(g_rank, g_global) < 0`, conflict grows with β, and at
β=1 only ≈3.7% of the anchor gradient lands on Top-10 candidates — most goes to
rank 101+.

**B8. The rank scorer inserted *better* candidates, yet aggregate got worse.** [user]
Rank β=0 swap analysis: removed-candidate future MSE mean 0.73979, added 0.59782.
Top-10 individual MSE improved. But candidate variance collapsed:
−74% (rank β=0), −65% (anchor β=0.1), −42% (β=1).

**B9. Exact decomposition.** Under uniform Top-K aggregation,
`I = A + V` holds exactly, where `I` = mean individual MSE, `A` = aggregate MSE
of the mean future, `V` = mean candidate spread about their own mean.
[repo] verified: the `residual` column of
`logs/set_oracle/pool_k_sweep_pred*.csv` is ≤ 3e-9 on every row.
So lowering `I` while lowering `V` more can *raise* `A`.

**B10. Set Oracle beats Individual Oracle on aggregate — [repo] recomputed.**
Recomputed here from `logs/set_oracle/set_only_candidates_pred*.csv`
(per-query means over 855–896 query×channel units):

| | I_ind | I_set | A_ind | A_set | set gain on A |
|---|---|---|---|---|---|
| H96 | 0.2604 | 0.3417 | 0.1303 | 0.0965 | **25.9%** |
| H192 | 0.3720 | 0.4632 | 0.2071 | 0.1680 | **18.9%** |
| H336 | 0.4977 | 0.5832 | 0.3079 | 0.2701 | **12.3%** |
| H720 | 1.4645 | 1.5310 | 1.1945 | 1.1463 | **4.0%** |

Set Oracle is *worse* on individual candidate quality at every horizon and
*better* on aggregate at every horizon. The joint condition
`A_set < A_ind AND I_set ≥ I_ind` holds for **100.0% of query units at all four
horizons** in this recomputation. **Good ten candidates ≠ a good set of ten.**

**B11. Generic diversity is not the answer.** [user]
Good+Diverse Oracle (top-30 by individual MSE, then maximize pairwise future
distance) loses to Individual Oracle on aggregate at every horizon:
H96 −6.8%, H192 −1.8%, H336 −6.8%, H720 −3.8%. At H96 its variance (0.245785)
essentially matches Set Oracle's (0.245243) while aggregate MSE differs greatly
(0.139201 vs 0.096719). **Generic diversity ≠ useful complementarity**; what is
needed is *target-directed complementarity*.

**B12. Recall@10's ground truth is itself misaligned.** [repo]+[user]
R@10's oracle is Individual Oracle. Even a perfect R@10 = 1.0 lands at
A_ind = 0.1303 while Set Oracle reaches A_set = 0.0965 (H96). So the gap is not
only "recall correlates weakly" but **the ground-truth definition of Recall@10 is
structurally misaligned with Stage-2's set-level objective.**
This does *not* mean individual quality is irrelevant: current Cosine Top-10 is
far worse than Individual Oracle, so the chain has two distinct gaps:

```
Cosine Top10  --(candidate-wise ranking gap)-->  Individual Oracle
              --(set composition gap)-------->  Set Oracle
```

**B13. Phase 0 — temperature calibration. [repo] verified.**
Soft objectives (SetMSE, ExpectedMSE) are confounded by τ, which sets how many
candidates the objective actually optimizes over. Target band: `N_eff ≈ 30–60`,
`Mass@10 ≈ 0.5–0.8`. Read from `logs/tau_calibration/pred*.log`:

| Horizon | chosen τ | N_eff | Mass@10 |
|---|---|---|---|
| H96 | 0.015 | 52.6 | 0.6332 |
| H192 | 0.015 | 56.3 | 0.6068 |
| H336 | **0.01** | 35.9 | 0.7129 |
| H720 | **0.02** | 49.9 | 0.6139 |

The same τ gives very different support across horizons (τ=0.015 →
N_eff 52.6 / 56.3 / 93.6 / 24.3 for H96/192/336/720), which is why per-horizon
calibration was required. Status: **complete**.

**B14. Phase A — pool/K sensitivity oracle. [repo]+[user]**
`logs/set_oracle/pool_k_sweep_pred*.csv`. At pool=100, K=10, H96 [repo]:

| Arm | I | A | V |
|---|---|---|---|
| cosine | 0.594556 | 0.318727 | 0.275829 |
| good_diverse | 0.370567 | 0.128620 | 0.241947 |
| individual | 0.259714 | 0.126704 | 0.133010 |
| set | 0.348854 | 0.089579 | 0.259275 |

[user] On full memory, Individual-Oracle vs Set-Oracle Top-10 overlap is 0.265
— i.e. ≈73.5% of the selected candidates differ. At H96/K=10 the
candidate-selection gap (cosine → individual) is ≈213% and the set-composition
gap (individual → set) is ≈102%. **Both bottlenecks are large simultaneously.**
Status: **complete**.

**B15. Phase C — direct Oracle imitation fails on validation. [repo] verified.**
Frozen healthy WCE encoder + asymmetric scorer + fixed P100, direct imitation of
Oracle labels. ETTh1 VAL, epoch 3, read from `logs/imitation/pred*_*.log`:

| | H96 individual | H96 set | H720 individual | H720 set |
|---|---|---|---|---|
| TeacherSetRecall@10 | 0.1363 | 0.1110 | 0.1279 | 0.0847 |
| imitation loss | 4.589 | 4.606 | 4.608 | 4.621 |
| uniform aggregate MSE@10 | 0.7430 | 0.7304 | 1.7593 | 1.6394 |
| individual MSE@10 | 0.9954 | 1.1005 | 2.1804 | 2.1500 |

Reference points: random Top-10 overlap in P100 = 10/100 = **0.10**; uniform CE
= ln(100) ≈ **4.605**. Every arm sits at ≈ both. **Direct Oracle supervision does
not generalize — and this is true for the Individual target as well as the Set
target.** Status: **complete**.

**B16. The "Set Oracle target is too noisy" hypothesis is rejected.** [repo]
Greedy restart overlap 0.878 (H96) / 0.902 (H720); restart relative aggregate gap
≈1.6% (H96) / ≈0.5% (H720). Re-running greedy from a different seed candidate
returns ≈88–90% of the same set, so target instability does not explain the
imitation failure.

**B17. Set-target supervision moved aggregate more than membership. [user] — weak signal only.**
The Set-target arm has *lower* TeacherSetRecall than the Individual-target arm
but *better* aggregate MSE (H96: 0.7304 vs 0.7430; H720: 1.6394 vs 1.7593).
This is **not** a controlled comparison against a proper baseline and must not be
used as a strong conclusion. It is recorded only as a further signal that
**exact membership matching ≠ aggregate utility**.

---

## Invalidated Results — Do Not Cite

**The pre-fix ETTh1 Stage-2 score sweep is invalid.** A wiring bug meant the
configured scorer was not propagated into Stage-2's actual Top-K selection path,
so in some runs selection was performed by the cosine path regardless of the
asymmetric/pair scorer that had been trained. Those ETTh1 Stage-2 numbers
therefore **cannot** be read as the downstream effect of the asymmetric/MLP
scorer.

- **Weather** was re-verified *after* the fix → **valid downstream evidence**.
- **ETTh1** requires a rerun under the fixed wiring, over the full grid:
  {KL, WCE} × {Cosine, Asymmetric, MLP} × {H96, H192, H336, H720}, all under
  identical Stage-2 conditions.
- Until that rerun exists, present ETTh1 Stage-2 as a **historical diagnostic**,
  explicitly labelled as pre-fix.

---

## Current Hypotheses

The campaign moved the research question from

> "how do we raise Recall further?"

to

> "how do we *define* a retrieval utility that is useful for forecasting, and can
> it be learned from past-only information?"

Open hypotheses, to be adjudicated by the reviewer:

- **Q1 (predictability):** is future-derived Oracle retrieval utility predictable
  from past-only information in a way that *generalizes*? B15 is the direct
  evidence that it may not be.
- **Q2 (capacity):** is the asymmetric scorer simply under-capacity?
- **Q3 (objective):** is Oracle-membership matching the wrong intermediate
  objective altogether, making end-to-end forecasting-MSE-aligned retrieval the
  right formulation?

Superseded framing: **H-A / H-B / H-C from the pre-campaign period are subsumed
by B4, B12, and Q1–Q3 and should not be cited as current.**

---

## Open Questions

- Does a pointwise scorer have the *capacity* to fit Individual/Set Oracle labels
  at all, even on a tiny memorization set? (Experiment A/B)
- Is the failure ETTh1-specific or general across ETTm1 / Weather?
  (Cross-dataset feasibility)
- Can retrieval that never matches Oracle membership still lower final
  forecasting MSE? (Track C, E2E)
- What is the right formal object for "target-directed complementarity", given
  that generic variance maximization fails (B11)?
- Why does the set-composition gain shrink monotonically with horizon
  (25.9% → 4.0%)?

---

## Do Not Conclude (as of 2026-09-02)

Explicitly **not** supported by current evidence. The reviewer should treat any
of these appearing in a writeup as an error:

| Claim | Why it is not supported |
|---|---|
| "More diversity is always better" | Good+Diverse Oracle *lost* to Individual Oracle at every horizon (B11) |
| "A pointwise scorer cannot represent the Set Oracle" | Individual-Oracle imitation failed *equally* (B15); the failure is not set-specific |
| "ETTh1 itself is the problem" | Cross-dataset feasibility has not been run |
| "E2E is the answer" | Track C has not been run |
| "The existing ETTh1 Stage-2 sweep shows the downstream effect of asymmetric/MLP selection" | Wiring bug; rerun required (see *Invalidated Results*) |
| "Recall does not matter" | Current Cosine Top-10 is far worse than Individual Oracle, so the candidate-quality gap is real and large (B12, B14) |
| "Raising Recall@10 raises forecast quality" | B2 and B4 contradict this; pre-campaign finding A2 is a within-sweep correlation, not a causal claim |

The most accurate current framing:

> Retrieving good candidates and constructing a good Top-K set are two distinct
> problems; and more fundamentally, it is not yet established that future-derived
> retrieval supervision generalizes from past-only information at all.

---

## Result Artifact Index

Where the results of the current diagnostic campaign live. Directory names,
file names, counts, and timestamps were verified on disk on 2026-09-02; the
phase labels were supplied by the user.

**Convention:** each probe writes to its own directory and existing files are
never touched. New work in progress (A1/B1 memorization probe) writes to
`logs/memorization/`; Track C writes to its own separate directory.

### Current campaign (phase-labelled)

| Phase | Directory | Contents | Ran |
|---|---|---|---|
| Phase 0 | `logs/tau_calibration/` | `pred{96,192,336,720}.log` — per-horizon tau. **Calibrated values: 96 = 192 = 0.015, 336 = 0.01, 720 = 0.02** | 2026-09-01 15:45 |
| Phase A | `logs/set_oracle/` | `pool_k_sweep_pred{96,192,336,720}.csv` — pool x K sweep | 2026-09-02 03:40 |
| Phase C | `logs/imitation/` | `pred{96,720}_{individual,set}.log` (4 files) — imitation gate | 2026-09-02 04:00–04:05 |

`logs/set_oracle/` also holds the per-candidate raw CSVs from the earlier
set-utility oracle diagnostic: `set_only_candidates_pred{96,192,336,720}.csv`
(≈250–290 KB each).

The remaining phases of the stated 7-phase plan are **UNKNOWN** — only Phase 0,
Phase A, and Phase C have been named.

### Preceding diagnostics

| Directory | Diagnostic | Files |
|---|---|---|
| `logs/swap_conflict/` | swap / conflict diagnostic | `swap_rows.csv`, `fingerprints.txt` |
| `logs/global_anchor/` | global anchor KL | `beta0p1.log`, `beta1p0.log` |
| `logs/frozen_scorer/` | frozen encoder + rank isolation | `A_cosine.log`, `B_asym_rank.log` |
| `logs/collapse_diag/` | representation-collapse onset | `WCE.log`, `Dynamic.log`, `Persistent.log` |
| `logs/persistent_probe/` | persistent pair mining | `Dynamic.log`, `Persistent.log` |
| `logs/rank_arms/` | boundary rank-loss arms | `ETTh1_pred{96,720}_wce_rank.log` (+ `.done` markers) |
| `logs/score_geometry/` | score geometry / cosine compression | `ETTh1_pred{96,720}.log` |
| `logs/headroom_redundancy/` | oracle headroom + redundancy substitution | 40 files, `ETTh1_pred<H>_<metric>_<loss>.log` over metric ∈ {cosine, asymmetric, pair2, …} x loss ∈ {kl, wce} |
| `logs/utilization_diag/` | retrieval-off counterfactual | 40 files, same naming scheme |

### Final Stage-2 numbers

| Directory | Contents |
|---|---|
| `logs/stage2_learned_score_corrected_selection/` | Stage-2 final MSE. 96 files, `ETTh1/pred{96,192,336,720}/<arm>_{stage2,e2e}.log` (+ `.done`) |
| `logs/weather_stage2/` | Weather Stage-2, `weather/pred{96,192,336,720}/` (64 files) |
| `logs/weather_top3/` | Weather top-3 baseline, `stage1/` and `stage2/` |

**None of these results are aggregated or interpreted in this document.** Doing
so requires a completed `EXPERIMENT_LOG.md` entry plus a reviewer pass; see
*Established Experimental Findings* for what is currently settled (all of it
predating this campaign, from `RESULTS_SUMMARY.md`).

---

## Known Repository Issues

Baseline established 2026-09-02 by running `pytest tests/` in
`/data/pjh_workspace/ts-env`:

```
396 passed, 2 failed
```

The two failures are **pre-existing at HEAD `c306def`**, not caused by the
uncommitted work. Verified by re-running both tests in a clean detached worktree
at `c306def`, where they fail identically.

| Test | Failure |
|---|---|
| `tests/test_stage1_new_losses.py::test_topk_coverage_reuses_target_indices_across_relations` | `assert target_pointers[0] == target_pointers[1]` — the target index tensor is not reused across relations (two distinct `data_ptr`s) |
| `tests/test_stage2_oracle_topk.py::test_identity_retrieval_uses_raw_target_source_relation_without_encoder` | `AttributeError: 'Model' object has no attribute 'retrieval_similarity'` |

**Use `396 passed, 2 failed` as the sanity-check baseline.** A future experiment
that reports these same two failures has not regressed; any *additional* failure
has.

These are not scheduled for repair as part of this workflow setup — no source
code was modified. Fixing them is a separate decision for the user.

---

## Uncommitted Work In Progress

The working tree carries ~2,600 lines of uncommitted changes plus 8 untracked
test files (verified 2026-09-02 against HEAD `c306def`). Inventory of what they
add, read from `git diff` and the test docstrings:

| Theme | New CLI flags | New functions | Tests |
|---|---|---|---|
| Boundary hard-pair rank loss | `--rank_loss_weight`, `--rank_margin`, `--rank_gap_threshold`, `--rank_pairs_per_query`, `--rank_pool_end`, `--rank_mining_mode`, `--rank_gap_weighted` | `boundary_hard_rank_loss`, `build_frozen_rank_pairs`, `frozen_pair_metrics` | `test_boundary_rank_loss.py` |
| Rank-failure diagnostics | — | `ranking_diagnostics`, `score_gradient_conflict`, `_persistent_batch`, `_pair_survival` | `test_rank_failure_diagnostics.py` |
| Collapse / score geometry | — | `collapse_geometry`, `score_geometry` | `test_collapse_geometry.py` |
| Frozen-encoder scorer | `--stage1_freeze_encoder` | (scorer initialized at cosine) | `test_frozen_encoder_scorer.py` |
| Global anchor on the frozen ranking | `--stage1_global_anchor_weight`, `--stage1_global_anchor_tau`, `--stage1_imitation_target`, `--stage1_imitation_pool` | `global_anchor_kl`, `retention`, `oracle_imitation_loss` | `test_global_anchor.py` |
| Set-level retrieval loss | `--stage1_set_mse_weight`, `--stage1_set_tau`, `--stage1_set_mse_normalization`, `--stage1_set_support_k`, `--stage1_set_support_weight`, `--stage1_expected_mse_lambda` | `soft_set_mse`, `hard_aggregate_metrics` | `test_stage1_set_level_loss.py` |
| Set-utility oracle selectors | — | `set_utility_metrics`, `select_individual_oracle`, `select_good_diverse`, `select_greedy_set`, `greedy_set_stability` | `test_set_utility_oracle.py` |
| Retrieval-off counterfactual | `--stage2_retrieval_off` | Stage-2 gate neutralization | `test_retrieval_off_counterfactual.py` |
| Stage-2 selection/redundancy reporting | — | `report_retrieval_selection`, `_selection_overlap_with_cosine`, `_subset_diag`, `_redundancy_diag`, `_subset_report` | `test_stage2_learned_score.py` (modified) |

The test docstrings state the motivating observations behind this work (recorded
here as *reported observations from those runs*, not as adjudicated findings):

- rank-only fine-tuning drove encoder effective rank from 16 to 1 within ten
  steps (`test_frozen_encoder_scorer.py`);
- training the scorer on 100 candidates moved the scores of all ~8,000, and the
  arm without a global anchor improved local ordering while R@10 fell by roughly
  two thirds (`test_global_anchor.py`);
- the rank scorer pulled individually-lower-error candidates into the Top-10 and
  made the aggregate worse, because error-cancelling spread was lost
  (`test_set_utility_oracle.py`);
- the first WCE+Rank run left validation flat with a NaN train diagnostic
  (`test_rank_failure_diagnostics.py`).

Recent run artifacts exist for this line of work but are **not aggregated**
anywhere. They are indexed under *Result Artifact Index* below.

---

## Unknown / Unverified

- Current approved experiment: **UNKNOWN** — `research/CURRENT_EXPERIMENT.md` is
  an unfilled template at the time of this setup.
- **Whether the in-flight work described under "Uncommitted Work In Progress"
  has produced reviewed results is UNKNOWN**; recent log directories exist but
  their numbers are not aggregated anywhere in the repository.
- Numbers in `RESULTS_SUMMARY.md` postdate commit history only loosely; the exact
  commit each row was produced at is **UNKNOWN**.
- Weather / Electricity results: driver scripts exist, results **UNKNOWN** (not
  present in `RESULTS_SUMMARY.md`).
- Target venue, submission deadline, and the intended headline contribution:
  **UNKNOWN**.
