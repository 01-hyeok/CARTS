# TRACK_A_STRICT_AUDIT.md

Strict correctness audit of Track A (the set-aware sequential retrieval line:
B0 → R0/R1/R2 → C1/C2 → E1/E2 → D1 → TF-DIAG → T1 → OPC1), ETTh1 H96.

**Read-only.** No training, no reruns, no code changes, no artifact
modification. Everything below comes from the committed artifacts, training
logs and source code, not from `REVIEW_FOR_CHATGPT.md`'s prose.

Confidence labels used throughout: `Confirmed` (verified in code + artifact),
`Likely` (strong converging evidence, one check short of proof), `Possible`,
`Unverified`.

---

# 1. Executive Summary

**Verdict: RED.**

The single fact that decides this audit is arithmetic, not interpretation:

| | A_weighted (retrieval aggregate, lower=better) | Stage-2 test MSE |
|---|---:|---:|
| **B0's own plain cosine Top-10** | **0.40734** | **0.37312** |
| best Track A arm (OPC1) | 0.41011 | 0.37340 |
| second best (T1) | 0.41039 | 0.37455 |
| D1 | 0.47786 | 0.38916 |
| C0 (the arm every Track A comparison is made against) | 0.48726 | 0.39526 |
| E2 | 0.51670 | 0.40418 |
| E1 | 0.52003 | 0.40668 |
| Set Oracle (headroom) | 0.10303 | — |

**No Track A arm ever produced retrieval better than B0's plain cosine, on
any aggregate metric, at any point in the campaign.** The arms the documents
rank as "best" (T1, OPC1) are precisely the arms whose retrieval output is
closest to *doing nothing* — reproducing the base retriever. Track A's
apparent progression is a progression from *actively harmful* to *inert*,
not from *inert* to *useful*.

Three specific documented conclusions are **wrong or unsupported**:

1. **`gap_recovery = −0.010` does not mean "essentially fully recovered".**
   By the metric's own formula (`scripts/eval_margutil01_stage2.py:280`)
   it means T1 recovered **−1%** of the B0→Set-Oracle gap. The document says
   the opposite, in three places.
2. **"Teacher-forcing/exposure bias was the largest-magnitude contributing
   factor"** is not supported. T1's own training log shows a near-stationary
   optimisation (train loss 0.88723 → 0.88294 over 9 epochs; SetConditioner
   grad-norm 0.02–0.05, i.e. 40–100× smaller than D1/OPC1; `val_overlap@10`
   flat at 0.0089–0.0091). The better-supported reading is that on-policy
   training supplied almost no usable gradient and the selector's output
   collapsed onto the base retriever.
3. **"T1 > D1 > E2" is an artifact of free-running-only evaluation.** Under
   the `oracle_first` anchor — already computed, in the same result files,
   never reported in `REVIEW_FOR_CHATGPT.md` — the ranking **inverts**:
   D1 = 0.34188, C0 = 0.35572, E2 = 0.35778, OPC1 = 0.36813, **T1 = 0.36913
   (worst)**. Given a correct first pick, T1 is the *worst* continuation
   policy of the four.

Additionally, **every Track A Stage-2 delta except C0/E1/E2-vs-B0 is below
this project's own declared 1-seed noise floor of 0.01 MSE**
(`research/RESEARCH_CONTEXT.md:226`), and every arm is a single seed. D1-vs-C0
(0.0061), T1-vs-OPC1 (0.00115) and OPC1-vs-B0 (0.00028) are all statistically
uninterpretable at n=1.

**What survives the audit:** the evaluation *plumbing* is sound. The
candidate mask, the free-running selection path, the Recall@K implementation,
the B0 fingerprint reproduction and the `duplicate_rate=0 / invalid_rate=0`
invariants are all correct and leak-free. The problem is not a broken
pipeline; it is that correct numbers were read wrongly.

---

# 2. Confirmed Findings

Things that are true, verified at code + artifact level.

### 2.1 No future leakage in free-running selection — `Confirmed`

`scripts/train_margutil01.py : run_sequence_dense()` lines 111–114: in the
`teacher_idx is None` branch the function appends `None` to
`u_target_steps`, never calls `dense_utility`, and takes
`nxt = u_hat.masked_fill(~valid_now, -inf).argmax(...)`. The query future
(`q_tgt`) is passed into the function signature but is provably unused on
that path. Free-running picks depend only on `u_hat`.

### 2.2 Candidate universe is correct — `Confirmed`

`utils/relation_memory.py : RelationMemorySampler.valid_indices()` (mode
`raft`, the project default). Memory is built from the **train split only**.
For a test query `query_start` is not in `self.starts`, so all 8449 train
windows are valid — no self-match is possible because the query is not in
the bank, and no temporal overlap is possible because the test split is
strictly later than train. For train queries a ±(seq_len+pred_len) window is
excised, removing both the self-match and every overlapping window. ✅

### 2.3 Structural selection invariants hold — `Confirmed`

`valid_now = cand_mask & ~selected_mask` makes duplicates and invalid picks
structurally impossible; every Track A `*_stage2.json` reports
`duplicate_rate = 0.0`, `invalid_rate = 0.0`. ✅

### 2.4 Protocol identity across arms — `Confirmed`

All five arms' `command.txt` use the **same** Stage-1 base checkpoint, the
**same** Stage-2 checkpoint, the **same** teacher cache
(`cache/seqfull01_teacher/ETTh1_pred96.pt`), `--top_k 10`, `--seed 0`, and
the **same** eval script (`scripts/eval_margutil01_stage2.py`). Every
`*_stage2.json` reproduces `b0_unforced_final_mse = 0.3731216918222617`
bit-identically and `n_test_rows_per_channel = 2785`, `n_channels = 7`.
The fingerprint check works and passes. ✅

### 2.5 `recall_at_k` is correctly implemented — `Confirmed`

`utils/retrieval_scoring.py:71`. Similarity ranked `largest=True`, oracle
distance `largest=False`, invalids pushed to the losing end of *both*
rankings, denominator = `k`, and queries with fewer than `k` valid
candidates excluded (`usable = valid_counts >= effective`). Direction,
denominator and masking are all right. ✅

### 2.6 t=1 utility target identity is correct — `Confirmed`

`scripts/eval_firstanchor_diag.py:293` uses `u_teacher0 = -dist_individual`
for the empty prefix. This is algebraically exact: with an empty prefix
`dense_utility` reduces to `MSE((w_i·Y_i)/w_i, Y_q) = MSE(Y_i, Y_q)`.
Ranking direction (`argsort(descending=True)` on a utility = −MSE) is
correct, and the global→local index remap at line 302 is correct. ✅

### 2.7 The Set Oracle headroom is real and enormous — `Confirmed`

`set_oracle_a_weighted = 0.10303` vs `b0_a_weighted = 0.40734`, identical in
all five files. The thing Track A set out to capture is worth ~0.30 in
A_weighted; Track A captured ≤ 0 of it.

### 2.8 E1's representation collapse and E2's prevention — `Confirmed`

`embedding_effective_rank` 17.48 → 2.96 → 1.49 (E1) vs 15.70–16.23 (E2),
from `results/EXP-ENCODER-ANCHOR01/representation_probe.csv` and the training
logs. The collapse finding and the anchor's effect on it are real. The
*downstream* conclusion drawn from it is a separate matter (§3.5).

---

# 3. Suspicious Findings

## 3.1 `gap_recovery` is being reported with its sign and meaning inverted — severity: **critical**, `Confirmed`

**The number.** T1 `gap_recovery = −0.010015`, OPC1 `−0.009111`.

**Why it is wrong.** `scripts/eval_margutil01_stage2.py:279-281`:

```python
denom = out['b0_a_weighted'] - out['set_oracle_a_weighted']
out['gap_recovery'] = (out['b0_a_weighted'] - out['seq_a_weighted']) / denom
```

so `gap_recovery = (A_B0 − A_seq) / (A_B0 − A_SetOracle)`. **0 = identical to
B0. 1 = identical to the Set Oracle. Negative = worse than B0.** The project
already states this definition correctly in `REVIEW_FOR_CHATGPT.md:515`
("negative = worse than B0, 0 = matches B0, 1 = matches the set oracle").

But the Track A write-ups say:

- `REVIEW_FOR_CHATGPT.md:1980` and `CURRENT_EXPERIMENT.md:339-340` —
  "gap_recovery -0.263 → **-0.010** (essentially fully recovered)"
- `REVIEW_FOR_CHATGPT.md:2097-2098` — "gap_recovery=-0.010 (essentially the
  full achievable gap recovered)"

**−0.010 is the opposite of "recovered".** T1 recovered −1.0% of the
achievable gap; it is still, marginally, *worse* than B0's own cosine.

**Related code:** `scripts/eval_seqfull01_stage2.py:198-201` (identical
formula).

**How to confirm:** already confirmed — the formula and the stored numbers
are both in the repository.

## 3.2 T1's headline result is an untrained-selector signature — severity: **critical**, `Likely`

**The numbers** (`logs/exp_onpolicy_prefix01/train_full.log`):

| epoch | train_loss | val_loss | val_overlap@10 | sc_grad_norm |
|---|---|---|---|---|
| 1 | 0.88723 | 0.88694 | 0.0089 | 0.02157 |
| 4 (**selected**) | 0.88324 | 0.88694 | **0.0091** | 0.03608 |
| 9 | 0.88294 | 0.88753 | 0.0089 | 0.05280 |

Total train-loss movement over nine epochs: **0.43%**. `val_overlap@10` moves
in the 4th decimal and the selected epoch (4) is indistinguishable from
epoch 1. For comparison, D1's grad-norm is 1.12–1.95 and OPC1's is 1.72–4.86
— **40–100× larger**.

**Why this matters.** `models/SequentialSetRetriever.py:42` computes
`h = normalize(LayerNorm(q + net([q, m])))` — a *residual* around the query
embedding — and `models/DenseUtilityRetriever.py:36` computes
`u = scale·cos(h, E) + b` with `scale` initialised to 1. A selector whose
`net` contribution stays small therefore scores candidates by
`cos(q, E)` — **which is exactly B0's own retrieval score**. An
under-trained selector in this architecture does not produce garbage; it
produces B0.

**Corroborating artifacts.** Every aggregate metric for T1/OPC1 sits within
0.75% of B0's, and their Oracle-agreement metrics are the *worst* of the
campaign:

| | A_weighted | HardAgg@10 | Stage-2 MSE | set_recall vs Set Oracle |
|---|---:|---:|---:|---:|
| B0 | 0.40734 | 0.40741 | 0.37312 | — |
| T1 | 0.41039 | 0.40960 | 0.37455 | 0.01142 |
| OPC1 | 0.41011 | 0.41098 | 0.37340 | 0.01110 |
| D1 | 0.47786 | 0.54802 | 0.38916 | 0.01656 |
| C0 | 0.48726 | 0.55139 | 0.39526 | 0.00973 |

and OPC1's `dense_first` t=1 aggregate is **0.66548** against a `b0_first`
t=1 aggregate of **0.66585** — i.e. replacing OPC1's own first pick with
B0's own first pick changes the aggregate by 0.0004. For C0 the same
substitution moves it by 0.0064, for D1 by 0.0121, for E2 by 0.0223.

**Caveat that keeps this at `Likely` rather than `Confirmed`.** A direct
weight comparison (this audit, CPU tensor diff of
`checkpoint_epoch1.pth` vs `checkpoint.pth`) shows the SetConditioner *did*
move: relative displacement 0.358 (T1), 0.269 (OPC1), 0.653 (D1). So the
parameters are not frozen. And T1's own t=1 aggregate (0.75448) is clearly
*not* B0's (0.66585) — T1's first pick differs, it only converges to B0's
aggregate by t=10. The claim "T1 output ≈ B0 output" is therefore strongly
indicated by six independent aggregate metrics but has not been proven at
the level of selected indices.

**The one check that settles it:** compute
`|picks_T1 ∩ B0_top10| / 10` and `|picks_OPC1 ∩ B0_top10| / 10` on test.
Both index tensors already exist inside `eval_margutil01_stage2.py`
(`picks` at line 151, `b0_topk_idx` at line 182) — the overlap is three lines
of inference, no training. **I have not run it (it requires a GPU inference
pass, which is outside the read-only scope of this audit).**

## 3.3 The arm ranking inverts under a different anchor, and only one anchor was reported — severity: **critical**, `Confirmed`

`eval_firstanchor_diag.py` computes three anchors for every arm. Only
`dense_first` reaches `REVIEW_FOR_CHATGPT.md`. All three are in the
`*_summary.json` files:

| Arm | `dense_first` MSE (reported) | `b0_first` MSE | `oracle_first` MSE (never reported) |
|---|---:|---:|---:|
| C0 | 0.39526 | 0.39164 | 0.35572 |
| D1 | 0.38916 | 0.38662 | **0.34188 (best)** |
| E2 | 0.40418 | 0.39559 | 0.35778 |
| OPC1 | **0.37340 (best)** | 0.37349 | 0.36813 |
| T1 | 0.37455 | 0.37422 | **0.36913 (worst)** |

Under `oracle_first` the order is **D1 ≫ C0 ≈ E2 ≫ OPC1 > T1** — the exact
reverse of the reported ranking. Note also that `oracle_first` beats B0
(0.37312) for C0, D1 and E2 but **not** for T1/OPC1: handed a perfect first
candidate, T1 and OPC1 are the only arms that *cannot* turn it into a
below-B0 forecast.

**Reading:** D1 learned a continuation policy that is genuinely good *given a
good state*, and bad at reaching one. T1/OPC1 learned a policy that is
inert — indifferent to the state it is given. The reported metric
(`dense_first`) rewards inertness because the base retriever is already
better than anything Track A learned.

**Why only one anchor is reported is itself a finding:** the `oracle_first`
column is the one that contradicts the headline, and it was computed, stored,
and omitted.

## 3.4 `hard_aggregate_mse_individual` is not a hard aggregate — severity: **medium**, `Confirmed`

`scripts/eval_margutil01_stage2.py:195`:

```python
diag['hard_agg_individual_se'] += float(dist_individual.gather(1, best_idx).squeeze(1)[vq].sum() * h)
```

`best_idx = dist_individual.argmin(...)` is the **single best candidate**.
Multiplying by `h` and dividing by `horizon` at line 288 returns that single
candidate's MSE — not the mean of the Individual Oracle's Top-10. Proof from
the artifacts: in every Track A json,
`hard_aggregate_mse_individual = 0.19101074653034067` and
`individual_oracle_mse = 0.19101074628625567` — the same number to 8
significant figures. The other three rows (`set_oracle`, `seq`, `b0`) use
the real `hard_agg()` at line 173.

**What it distorts:** the "Individual" row of the HardAggregate comparison
is a top-1 oracle masquerading as a Top-10 aggregate, which makes Individual
look far better than Set (0.191 vs 0.305) in a column where
`research/RESEARCH_CONTEXT.md` B10 records the correct H96 values as
`I_ind = 0.2604`, `A_ind = 0.1303`. It does **not** affect any
seq-vs-B0 comparison.

## 3.5 `A_weighted` can hide bad set composition — severity: **medium**, `Likely`

`a_weighted(idx)` (line 167) weights the arm's own picks by **B0's** score
`learned_ref`, softmaxed at `tau = 0.1`. Over 10 candidates a cosine spread
of 0.2 gives a weight ratio of e² ≈ 7.4, so when an arm picks candidates B0
scores very differently the aggregate collapses toward whichever pick B0
likes most. `A_weighted` then partly measures *"does the pick set contain at
least one candidate B0 rates highly"*, not *"is this a good set of ten"*.

This is the most plausible mechanical explanation for E2's headline
disagreement (`A_weighted` 0.5167, middling; `HardAgg` 0.6768, worst of all
arms). **It is a real phenomenon, not a bug** — but it means `A_weighted`
should not be used as the set-quality metric, and `gap_recovery`, which is
built on it, inherits the problem. No `alpha_entropy` is recorded for these
picks, so the degree of collapse is currently unmeasured.

## 3.6 Selection-relevant diagnostics are measured at states the on-policy arms never visit — severity: **medium**, `Confirmed`

`eval_margutil01_stage2.py:228,231,254` build the utility diagnostic's state
and target from `teacher_idx_c` — the **Oracle prefix** — for every arm. T1
and OPC1 were never trained at oracle-prefix states, so their strongly
negative `utility_spearman_mean` (−0.514, −0.554, vs C0's −0.103) is
expected and carries no information about their selection quality.

Symmetrically, the t≥2 diagnostics run under `--anchor_policy dense_first`,
i.e. each arm's **own** first pick, so "t2 selected rank median" compares
arms at *different* states. **Neither the t1/t2 family nor the utility
family is a like-for-like comparison between teacher-forced-trained and
on-policy-trained arms**, and the campaign's cross-arm rankings on those
metrics should not be used.

(t=1 metrics are the exception: the empty prefix is identical for all arms,
so `t1_spearman_top1pct`, `t1_oracle_rank_median` and `t1_*_containment`
*are* comparable.)

## 3.7 The C0 baseline never learned to rank — severity: **high**, `Confirmed`

`logs/exp_toptail_rank01_r2/chain.log`:

```
epoch 1 ... pairwise_loss=0.70236 pos_score_mean=0.11101 neg_score_mean=0.12929 margin=-0.01828 score_std=0.01114
epoch 6 ... pairwise_loss=0.69733 pos_score_mean=0.11526 neg_score_mean=0.12345 margin=-0.00819 score_std=0.01244
```

`pairwise_loss` sits at **0.697–0.702 against ln(2) = 0.6931** for the whole
run, and `margin = pos − neg` is **negative at every epoch** — the model
ranks the *worse* candidate higher on average. `score_std ≈ 0.011`: the
utility head has essentially no dynamic range. `best_epoch = 1`.

C0 is therefore a chance-level ranker that actively degrades retrieval
(A 0.487 vs B0's 0.407). **Every Track A claim of the form "X beats C0" is a
claim about beating a chance-level, actively-harmful baseline**, which is a
much weaker statement than the documents make it sound.

## 3.8 `val_loss = 0.00000` for all six epochs of both E1 and E2 — severity: **medium**, `Confirmed`

`logs/exp_encoder_unfreeze01/train_full.log` and
`logs/exp_encoder_anchor01/train_full.log` print `val_loss=0.00000` at every
epoch. A validation loss that is *identically* zero is not a plausible value
for a SmoothL1+pairwise objective. Checkpoint selection used
`val_overlap@10` rather than `val_loss`, so the selected checkpoints are not
affected — but a logged metric that is silently broken passed unremarked
through two experiments' sanity checks, which is itself an audit finding
about the sanity-check regime.

## 3.9 Checkpoint selection runs on a noise-level criterion — severity: **high**, `Confirmed`

`val_overlap@10` across every arm:

| Arm | range over all epochs | selected | chance level (10/8449) |
|---|---|---|---|
| C0 | 0.0039–0.0082 | 0.0082 (ep1) | 0.0012 |
| E1 | 0.0042–0.0083 | 0.0083 (ep1) | 0.0012 |
| E2 | 0.0047–0.0094 | 0.0094 (ep1) | 0.0012 |
| D1 | 0.0119–0.0146 | 0.0146 (ep10) | 0.0012 |
| T1 | 0.0087–0.0091 | 0.0091 (ep4) | 0.0012 |
| OPC1 | 0.0087–0.0089 | 0.0089 (ep4) | 0.0012 |

For T1 and OPC1 the total spread across nine epochs is 0.0004 and 0.0002.
Selecting `best_epoch` on that is indistinguishable from selecting at random,
and the project has already recorded (EXP-ASYM-SCORER01) that this proxy
disagrees with every downstream metric. **The `best_epoch` values for T1 and
OPC1 should be treated as arbitrary.**

## 3.10 The on-policy objective is self-referential — severity: **high**, `Likely`

In `scripts/train_onpolicy_prefix01.py : run_sequence_onpolicy()` the target
at step *t* is `-dense_utility(prefix_own, ...)` where `prefix_own` is the
model's **own** running selection (line 72, line 81). The target therefore
moves with the model. A low, flat loss (T1: 0.887 → 0.883) is consistent with
the model steering into states where its own utility estimate is easy to fit,
with no improvement in selection quality. This is structurally the **same
endogenous-metric artifact** this project already documented for the
Set-Onpolicy arm in `research/AUDIT_ORACLE_RANK_GAIN01.md` §4.4 — but Track A
was written before that audit and never accounts for it.

Consequence: **T1's and OPC1's training losses are not comparable to C0's or
D1's** (which have fixed Oracle-trajectory targets), and neither is evidence
about learning progress.

## 3.11 Selective metric reporting in the E2 table — severity: **low**, `Confirmed`

`REVIEW_FOR_CHATGPT.md:1658-1666` presents E2 as "best of 3" on t1 (every
metric) and "most of t2", listing `t2 continuation_regret`. It omits
`t2_selected_rank_median`, which is in E2's own `metrics.csv`:
C0 = 1934.0, E1 = **539.5**, E2 = **1929.5**. On that metric E2 is not best
of three; it is barely distinguishable from C0 and 3.6× worse than E1.

---

# 4. Invalid / Confounded Conclusions

Stated flatly, as requested.

### 4.1 "Teacher-forcing / exposure-bias mismatch was the largest-magnitude contributing factor" — **do not make this claim.**

The evidence offered is T1's Stage-2 MSE. That number is fully explained by
the competing hypothesis "the on-policy objective gave near-zero gradient and
the selector reverted to the base retriever", which is *better* supported by
T1's own artifacts (§3.2, §3.10): near-stationary loss, 40–100× smaller
gradients, every aggregate metric converging on B0's, the worst Oracle-
agreement metrics of the campaign, and the worst `oracle_first` continuation
of all four arms. The current data cannot distinguish the two hypotheses, and
where it leans, it leans against the published one.

### 4.2 "gap_recovery = −0.010 means essentially the full achievable gap was recovered" — **this is arithmetically false.**

It means −1.0% of the gap was recovered. See §3.1.

### 4.3 "T1 > D1 > E2, ranked by Stage-2 impact" — **do not make this claim.**

It holds under `dense_first` and inverts under `oracle_first` (§3.3). A
ranking that flips with the choice of anchor, where both anchors were
computed and only one reported, is not a ranking.

### 4.4 "D1 is the first arm to beat C0 on Stage-2" — **true as a number, uninterpretable as a result.**

ΔMSE = 0.0061 at **one seed**, against this project's own declared noise
floor of 0.01 (`RESEARCH_CONTEXT.md:226`: 3-seed spread 0.3886/0.3988/0.3999).
Compounded by an admitted training-budget confound (D1 `best_epoch=10`, all
others 1 or 4) and a 10× difference in gradient magnitude. **Do not claim D1
beat C0 until a multi-seed and/or matched-budget run exists.**

### 4.5 "OPC1 came within 0.00028 of B0" — **do not present this as near-parity.**

0.00028 is 3% of the noise floor. The defensible statement is "OPC1 is
indistinguishable from B0 at one seed" — and, since OPC1's *retrieval*
aggregate is strictly worse than B0's on both A_weighted and HardAggregate,
the most likely reason it is indistinguishable is that it is doing
approximately what B0 already does.

### 4.6 "D1's/T1's gains combine sub-additively" — **not supported.**

Sub-additivity is computed from three numbers whose pairwise differences
(0.0061, 0.0012) are all inside the noise floor. There is no measurable
"gain" to be additive or sub-additive about.

### 4.7 "Preventing collapse recovered only a small fraction of the gap, so representation was not the primary bottleneck" — **confounded.**

E2 vs E1 is 0.0025 MSE, a quarter of the noise floor. The *collapse* finding
(§2.8) is solid; the *downstream* inference from a 0.0025 delta at one seed
is not.

### 4.8 Everything in §3.6's metric families, used cross-arm — **do not compare.**

`utility_spearman_mean`, `utility_regret_by_step`, and all t≥2 metrics are
measured at arm-dependent or training-regime-mismatched states.

---

# 5. Metric Audit

| Metric | Where | Direction | Denominator | Masking | Self/overlap | Leakage | Split-consistent | Support | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `recall_at_k` | `utils/retrieval_scoring.py:71` | sim ↓desc, dist ↑asc ✅ | `k` ✅ | invalid → losing end of both ✅ | via cand_mask ✅ | none | yes | `valid_counts>=k` filter ✅ | **PASS** |
| `seq_set_recall_at_k` | `eval_margutil01_stage2.py:209` | n/a | `k` ✅ | `vq` filter ✅ | ✅ | none | test only | 8449 | **PASS** |
| `A_weighted` | `:167` | lower=better ✅ | `.mean(-1)` over H ✅ | `vq` ✅ | ✅ | uses `q_tgt`, diagnostic-only ✅ | test only | 8449 | **PASS (but see §3.5)** |
| `hard_agg` (seq/b0/set) | `:173` | lower=better ✅ | `.sum(-1)/horizon` ✅ | `vq` ✅ | ✅ | diagnostic-only | test only | 8449 | **PASS** |
| `hard_aggregate_mse_individual` | `:195` | — | — | — | — | — | — | — | **FAIL — top-1, not Top-10 (§3.4)** |
| `gap_recovery` | `:279-281` | 0=B0, 1=Oracle | `A_B0 − A_SetOracle` ✅ | ✅ | ✅ | none | test only | 8449 | **Code PASS / reporting FAIL (§3.1)** |
| `utility_spearman/pearson` | `:243-244` | — | rank-centred ✅ | `valid_now` ✅ | ✅ | oracle prefix by design | test, 200-query subsample | 8449 cands ✅ | **Code PASS / cross-arm INVALID (§3.6)** |
| `utility_regret_by_step` | `:252` | student pick vs oracle best ✅ | per-step count ✅ | ✅ | ✅ | oracle prefix | test | 8449 | **Code PASS / cross-arm INVALID** |
| `oracle rank`, `top-N containment` | `eval_firstanchor_diag.py:301-314` | desc on utility ✅ | `n_valid−1` for pct ✅ | `valid_b` ✅, local↔global remap ✅ | ✅ | empty prefix, common to all arms ✅ | test+train | 8449 | **PASS — comparable** |
| `spearman_within_teacher_top1pct` | `:316-325` | desc ✅ | top-1% of valid ✅ | ✅ | ✅ | ✅ | both | 8449 | **PASS** |
| `NDCG@10/@50` | `:123-137,329-332` | rel = shifted utility, ranked by student ✅ | ideal DCG ✅ | ✅ | ✅ | ✅ | both | 8449 | **PASS** |
| t≥2 continuation metrics | `eval_continuation_diag.py` | — | — | — | — | — | — | — | **Code not faulted; cross-arm INVALID (arm-specific anchor, §3.6)** |
| `dense_utility` | `utils/dense_utility.py:39` | A = MSE ✅ | `.mean(-1)` ✅ | caller masks ✅ | ✅ | `@torch.no_grad`, target-only ✅ | all | full memory, chunked ✅ | **PASS** |
| `val_overlap@10` (checkpoint criterion) | training scripts | higher=better | `k` | ✅ | ✅ | none | val | 8449 | **Code PASS / selection power ≈ 0 (§3.9)** |
| `val_loss` (E1/E2) | `train_encoder_*01.py` | — | — | — | — | — | — | — | **FAIL — identically 0.0 (§3.8)** |
| Stage-2 MSE | `:266-273` | lower=better ✅ | element count ✅ | `valid_query` ✅ | ✅ | none | test | 2785×7 | **PASS** |

**Answering the checklist item "does the metric evaluate what the selector
actually selected?"** — For Stage-2 MSE, `A_weighted`, `hard_agg` and
`seq_set_recall`: **yes**, they all consume `picks` from the same
`run_sequence_dense(..., teacher_idx=None)` call at line 151, and that same
`picks` tensor is what is injected into Stage-2 via `set_forced_selection`.
There is **no** diagnostic-vs-production selection divergence. For the
utility and t≥2 families: **no** — they re-derive a different (oracle or
anchor-specific) trajectory, as designed.

---

# 6. Pipeline Audit

`dataset → query/candidate → mask → encoder → score → selector → state
update → indices → futures → aggregation → Stage-2 → metric`

| Stage | Code | Finding |
|---|---|---|
| dataset / splits | `data_provider/data_loader.py`, `utils/relation_memory.py:6` | ETT-hour fixed 12/4/4 borders; scaler fit on train only. ✅ |
| memory bank | `RelationMemorySampler.__init__` | built from the **train** dataset; `memory_x`/`memory_y` sliding windows truncated to `num_windows`. ✅ |
| candidate mask | `valid_indices()` (raft) | test → all 8449 train windows; train → ±(L+H) excised. No self-match, no overlap. ✅ (§2.2) |
| index space | throughout | `picks`, `teacher_idx_c`, `b0_topk_idx`, `best_idx` are **all** global memory indices into the same `[B, N]` axis; `all_tgt.gather(1, idx)` and `learned_ref.gather(1, idx)` are consistent. The only local↔global remap is `eval_firstanchor_diag.py:302`, and it is correct. **No batch-local/global index confusion found.** ✅ |
| encoder | `encode()` `:70-72` | `F.normalize(encoder(_relation_tensor(x,c,c)))` under `no_grad`, identical helper in every Track A script. ✅ |
| score | `UtilityHead` `:36` | `scale·cos(h,E)+b`; `scale` init 1.0 — hence the "untrained ⇒ B0" property (§3.2). Noted, not a bug. |
| selector / state | `run_sequence_dense` `:93-116` | free-running: `m = E[stack(picks)].mean(1)`, t=0 → `empty_token`. **Identical** to T1's training-time construction (`train_onpolicy_prefix01.py:68-73`) and to OPC1's. Train/eval state transition matches. ✅ |
| aggregation | `a_weighted` `:167`, `hard_agg` `:173` | weights from **B0's** score, deliberately fixed across arms. Correct as specified; metric-design weakness at §3.5. |
| Stage-2 injection | `:257-265` | `set_forced_selection(forced)` then `set_forced_selection(None)`; B0 unforced pass run first in the same loop and reproduces 0.3731216918222617 exactly in all five files. ✅ |
| metric | `:266-273` | `valid_query` filter, sum-of-squares / element count. ✅ |

**Conclusion of the pipeline audit: no implementation defect was found in the
retrieval → selection → aggregation → Stage-2 path.** The two code-level
defects found (§3.4, §3.8) are in *reporting* metrics, not in the pipeline.
This is why the verdict is RED for *conclusions* and not for *code*.

---

# 7. Required Experiments

Priority order. **None started; approval required.**

### R1 — B0-overlap measurement (no training, ~minutes) — **do this first**

- **Purpose:** settle §3.2. Decide whether T1/OPC1 learned a selector or
  reverted to the base retriever.
- **Changed variable:** none — pure measurement.
- **Fixed:** everything. Reuses the existing checkpoints and
  `eval_margutil01_stage2.py`'s own `picks` and `b0_topk_idx` tensors.
- **Measure:** `|picks ∩ B0_top10| / 10` per query, per arm, on test; plus
  Kendall-τ between `u_hat` and `cos(q,E)` over the full 8449 candidates;
  plus mean `‖h − q‖ / ‖q‖`.
- **Success condition:** overlap ≲ 0.3 and low τ ⇒ T1/OPC1 really are
  distinct selectors and the exposure-bias reading survives.
- **Failure interpretation:** overlap ≳ 0.7 or τ ≳ 0.9 ⇒ §4.1 is confirmed
  and Track A's headline must be retracted.

### R2 — 3-seed confirmation of B0, C0, D1, T1, OPC1

- **Purpose:** make any Δ ≤ 0.01 interpretable at all.
- **Changed variable:** seed ∈ {0, 1, 2}. **Only** the seed.
- **Fixed:** every hyperparameter, checkpoint-selection rule, eval script.
- **Success condition:** an arm's mean beats B0's mean by more than the
  pooled std.
- **Failure interpretation:** if no arm separates from B0, Track A's
  conclusion is "a learned set-aware selector did not beat plain cosine at
  ETTh1 H96", which is a publishable negative result — but only if stated
  as one.

### R3 — Matched-budget C0 (resolves the D1 confound)

- **Purpose:** separate "CE loss is better" from "D1 trained 10 epochs".
- **Changed variable:** C0's `train_epochs`/patience only, set to whatever
  lets it run the same 10 epochs D1 ran.
- **Fixed:** loss, encoder, scorer, seed, eval.
- **Success condition:** D1 still beats matched-budget C0 by > seed std.
- **Failure interpretation:** D1's win was a training-budget artifact.

### R4 — Report `oracle_first` and `b0_first` for every arm

- **Purpose:** close §3.3. No new compute — the numbers exist in the
  `*_summary.json` files already.
- This is a documentation action, not an experiment.

### R5 — Fix or retire `hard_aggregate_mse_individual`

- §3.4. A one-line change, but it touches a production eval script used by
  completed experiments, so **it must not be edited in place** — see §9.

---

# 8. Track A Current Verdict

## `RED`

Not because the code is broken — §6 found no defect in the retrieval→Stage-2
pipeline, and §5 passes most metrics. **RED because the conclusions currently
recorded in `research/REVIEW_FOR_CHATGPT.md` and
`research/CURRENT_EXPERIMENT.md` are not supported by the artifacts those
documents cite, and in two cases are contradicted by them.**

The four load-bearing reasons:

1. **A metric is being read with its meaning inverted.** `gap_recovery =
   −0.010` is reported as "essentially fully recovered" when the formula in
   the repository says it means "recovered −1% of the gap; still worse than
   B0". Three separate passages depend on this reading (§3.1).
2. **The headline arm's own training log contradicts the headline.** T1
   moved its training loss 0.43% in nine epochs with gradients 40–100× below
   its siblings', and every aggregate metric it produces converges onto the
   base retriever it was supposed to beat (§3.2, §3.10).
3. **The reported ranking inverts under an anchor that was computed and not
   reported.** Under `oracle_first`, T1 is the worst arm and D1 the best —
   the reverse of the published order (§3.3).
4. **Nothing is statistically resolvable.** Every arm is one seed; every
   decisive delta (D1−C0 = 0.0061, OPC1−T1 = 0.00115, OPC1−B0 = 0.00028) is
   below the project's own 0.01 noise floor, and the checkpoint-selection
   criterion that picked those checkpoints has a total dynamic range of
   0.0002–0.0004 for the two arms in question (§3.9, §4.4–4.6).

**The most defensible summary of Track A as it currently stands:**

> Across eight interventions (loss formulation, scorer capacity, encoder
> freezing, anchor regularisation, choice-CE supervision, on-policy prefix,
> and their combination), **no learned set-aware selector produced retrieval
> better than B0's plain cosine Top-10 on any aggregate metric**, and none
> produced a Stage-2 MSE below B0's. The arms that score best do so by
> converging on the base retriever's own behaviour. The Set-Oracle headroom
> (A 0.103 vs B0's 0.407) is real and remains entirely uncaptured.

That is a coherent, honest and genuinely informative negative result. It is
also a *different* result from the one currently written down. Track A can go
to `YELLOW` as soon as R1 and R2 are run; it cannot be published in its
current form.

---

# 9. Proposed corrections — NOT EXECUTED

Per the standing rule, nothing below was applied. Each needs explicit
approval.

### [ISSUE-1] `gap_recovery` misreported

- **위치:** `research/REVIEW_FOR_CHATGPT.md:1980, 2097-2098`;
  `research/CURRENT_EXPERIMENT.md:339-340`
- **문제:** `gap_recovery = −0.010` described as "essentially fully recovered"
- **현재 동작:** 문서가 지표를 정반대로 해석
- **예상 동작:** "B0와 동등, 달성 가능한 gap의 −1% 회수"
- **기존 결과 영향:** 숫자는 정확함. 결론 문장만 오류.
- **최소 수정안:** 해당 3개 문장을 정정하고, 기존 문장은 삭제하지 않고
  `RETRACTED` 표기 + provenance 유지
- **재실행 필요 범위:** 없음
- **수정할까요?**

### [ISSUE-2] `hard_aggregate_mse_individual` computes top-1, not Top-10

- **위치:** `scripts/eval_margutil01_stage2.py:195` (+ label at `:288`)
- **문제:** `dist_individual.gather(1, best_idx)` 는 단일 최우수 candidate
- **현재 동작:** `individual_oracle_mse` 와 동일한 값을 다른 이름으로 출력
- **예상 동작:** Individual Oracle Top-10의 `hard_agg`
- **기존 결과 영향:** 완료된 모든 Track A json의 해당 필드가 오라벨.
  seq/b0 비교에는 영향 없음.
- **최소 수정안:** 프로덕션 스크립트를 **수정하지 말고**, 필드명을
  문서에서 `individual_oracle_top1_mse`로 정정 표기. 실제 Top-10 값이
  필요하면 별도 스크립트로 재계산.
- **재실행 필요 범위:** 수정 시 eval 재실행 필요 → **권장하지 않음**
- **수정할까요?**

### [ISSUE-3] `val_loss = 0.00000` in E1/E2

- **위치:** `scripts/train_encoder_unfreeze01.py`,
  `scripts/train_encoder_anchor01.py` (validation loss accumulation)
- **문제:** 6 epoch 전부 정확히 0.0
- **현재 동작:** 손상된 로그 값
- **예상 동작:** 유효한 val loss
- **기존 결과 영향:** checkpoint 선택은 `val_overlap@10` 사용 → 결과 불변.
  로그 신뢰도 문제만 존재.
- **최소 수정안:** 원인 조사 후 보고. 완료된 실험의 스크립트이므로
  in-place 수정은 재현성을 깨뜨림.
- **재실행 필요 범위:** 없음 (로그 전용)
- **수정할까요?**

### [ISSUE-4] `oracle_first` / `b0_first` 미보고

- **위치:** `research/REVIEW_FOR_CHATGPT.md` Track A 섹션 전체
- **문제:** 계산·저장되었으나 보고되지 않은 컬럼이 결론을 뒤집음
- **최소 수정안:** §3.3 표를 문서에 추가 (신규 계산 불필요)
- **재실행 필요 범위:** 없음
- **수정할까요?**

---

## Provenance

- git HEAD at audit time: `401381e` (`Add EXP-RETRIEVAL-QUALITY-DIAG01
  report and results`), branch `main`, clean tree.
- Artifacts read: `results/{EXP-TOPTAIL-RANK01/R2, EXP-ENCODER-UNFREEZE01,
  EXP-ENCODER-ANCHOR01, EXP-ORACLE-CHOICE01, EXP-TEACHER-FORCING-DIAG01,
  EXP-ONPOLICY-PREFIX01, EXP-ONPOLICY-CHOICE01}/` — `*_stage2.json`,
  `*_summary.json`, `metrics.csv`, `command.txt`.
- Logs read: `logs/{exp_toptail_rank01_r2, exp_encoder_unfreeze01,
  exp_encoder_anchor01, exp_oracle_choice01, exp_onpolicy_prefix01,
  exp_onpolicy_choice01}/`.
- Code read: `scripts/{eval_margutil01_stage2, eval_seqfull01_stage2,
  eval_firstanchor_diag, train_margutil01, train_onpolicy_prefix01}.py`,
  `models/{SequentialSetRetriever, DenseUtilityRetriever}.py`,
  `utils/{dense_utility, relation_memory, retrieval_scoring}.py`,
  `exp/exp_stage1_relation.py`.
- Only computation performed: CPU `torch.load` + parameter-norm diff of
  `checkpoint_epoch1.pth` vs `checkpoint.pth` for T1/D1/OPC1 (§3.2). No GPU
  job, no training, no file overwritten.
