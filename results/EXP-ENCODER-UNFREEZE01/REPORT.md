# EXP-ENCODER-UNFREEZE01 — Report

Scope: ETTh1, H96, seed 0, top_k=10. C0 = R2 (hybrid loss, cosine
`UtilityHead`, FROZEN B0 encoder) reused verbatim from `EXP-TOPTAIL-RANK01`,
not retrained. E1 = identical R2 hybrid loss, identical `UtilityHead`,
identical `SetConditioner`/`EmptySetToken`, TRAINABLE encoder initialised
from the same B0 Stage-1 checkpoint. Nothing else changed — no encoder
depth/width/architecture change, no scorer change, no loss/lambda change,
no LR sweep, no SetConditioner change. Weather/H720/3-seed explicitly not
run.

**Research question:** is the frozen B0 encoder representation a bottleneck
for the R2 set-conditioned utility objective, or can the existing encoder
architecture adapt to it end-to-end and improve the realized selector?

---

## 1. Implementation / online re-encoding verification

`scripts/train_encoder_unfreeze01.py` (new). The only structural change vs.
C0/R2 is `for p in model.encoder.parameters(): p.requires_grad = True`
(C0/R2 sets this to `False`). Query and candidate embeddings are BOTH
re-derived from the current encoder parameters at every training step —
`encode_raw()` always calls `model.encoder(model._relation_tensor(x, c, c))`
on raw input, never a cached tensor. Sanity check C
(`test_C_no_stale_candidate_bank_embeddings_track_current_encoder`, PASSED)
directly verifies this: perturbing the encoder's parameters changes the
candidate embeddings on the very next forward call, with nothing cached
across the perturbation.

Memory-safe streaming training (`encoder_unfreeze_step`), generalising
EXP-STRONG-SCORER-DIAG01's OOM fix one level deeper (now the CANDIDATE
EMBEDDING computation itself, not just the scorer head, carries a graph
through the encoder):
1. full-memory hard-negative mining from a `torch.no_grad()` chunked
   candidate-encoding pass (`_encode_candidates_nograd`) — exact global
   Top-K, never a shortlist, `FULL MEMORY -> DIRECT TOP-K` unaffected;
2. the pairwise loss re-encodes ONLY the small gathered positive/hard-
   negative raw candidate windows WITH grad, backward immediately;
3. the dense SmoothL1 term re-encodes the FULL candidate bank again,
   chunk by chunk, each chunk getting its own fresh encoder forward pass
   and its own immediate backward — no chunk's graph is ever held
   simultaneously with another's.
`q` and `m` (the SetConditioner's set-state summary — for a trainable
encoder, `m` at t>0 now also carries a candidate-encoder graph through the
oracle-prefix embeddings, not just through `EmptySetToken` at t=0) are each
produced by a zero-arg callable (`q_fn`, `m_fn`) that reruns the encoder
fresh on every call, generalising the `m_fn` fix EXP-STRONG-SCORER-DIAG01
needed only for the trainable `EmptySetToken` case to every tensor that now
touches the (trainable) encoder. `optimizer.step()` is called exactly once
per batch, never inside `encoder_unfreeze_step()`.

Mathematical equivalence verified BEFORE the real run by sanity check D
(`test_D_chunk_equivalence_scores_losses_and_all_gradients_match`, PASSED):
unchunked vs. chunked streaming training from identical initial weights
produce matching losses (atol 1e-4) and matching gradients on every
encoder/SetConditioner/UtilityHead parameter (atol 1e-3). Running this test
to completion with many chunk iterations (and the full real-data run
completing 6 epochs without a `RuntimeError`) also exercises the
`q_fn`/`m_fn` fresh-recompute-per-call contract that fixes the double-
backward bug class EXP-STRONG-SCORER-DIAG01 first found.

A real-data smoke run (1 epoch, `--patience 1`, full ETTh1 train/val) was
launched first and confirmed correct execution (97% GPU util, ~6GB memory,
no OOM/RuntimeError) before being stopped early once a targeted timing
probe gave concrete per-batch numbers; `cand_chunk_size=2048` was chosen
from that probe (0.56s per (batch, channel) K=10-step pass, vs. 0.95s at
1024 and 3.1s — including one-time warmup — at 512) to keep the full run to
an estimated ~20-25 min/epoch. Actual epoch times: 1305.2s (epoch 1,
includes CUDA warmup), then 650-690s/epoch thereafter.

## 2. Encoder gradient verification

`encoder_grad_norm` was nonzero and finite at every epoch (0.046, 0.028,
0.028, 0.087, 0.046, 0.070 for epochs 1-6) — the encoder received a real,
non-degenerate gradient throughout training, confirmed independently by
sanity check A (`test_A_encoder_gradient_flows_query_and_candidate_side`,
PASSED — gradient reaches the encoder from both the query-side and
candidate-side branches, ruling out an accidentally-detached candidate
path) and check B (`test_B_frozen_encoder_regression_receives_no_grad`,
PASSED — confirms the C0/R2 code path still produces exactly zero encoder
gradient, i.e. the `requires_grad` flip is the only behavioural difference
between the two arms).

## 3. Controlled-variable audit

- `trainable_params`: E1=140802 (`SetConditioner`=49664, `EmptySetToken`=128,
  `UtilityHead`=2, `Encoder`=91008) vs. C0's 49794 non-encoder-inclusive
  count — the encoder's 91008 parameters are the only addition, matching
  the base checkpoint's own encoder parameter count exactly (same
  architecture, same layer count, same `d_model`).
- Optimizer: single Adam over every trainable parameter (SetConditioner +
  EmptySetToken + UtilityHead + encoder), `lr=0.001` — the SAME learning
  rate C0/R2 used (inherited from the base checkpoint's own `args`, not
  overridden). No separate encoder LR or LR multiplier: an explicit search
  of `exp/exp_stage1_relation.py`, `models/RelationStage1.py`, and `run.py`
  before writing `scripts/train_encoder_unfreeze01.py` found no existing
  "encoder LR multiplier" policy anywhere in this codebase, so none was
  invented — this is the most controlled setting available, per the user's
  explicit instruction not to sweep LR.
- `train_epochs=10`, `patience=5`, `batch_size=32` — identical to C0/R2
  (also inherited unmodified from the base checkpoint's args).
- Loss formula, `lambda_rank=1.0`, oracle-prefix teacher forcing, dense
  utility target definition, `UtilityHead`'s `a*cosine+b` scorer, K=10,
  full-memory candidate support — all byte-identical to C0/R2's own code
  (`UtilityHead`, `mine_pairs`, `dense_utility`, `candidate_weights`,
  `normalize_utility` are imported/reused, not reimplemented; see
  `working_tree.diff`).
- Encoder initialisation: loaded from the exact same B0 Stage-1 checkpoint
  as C0/R2 (`checkpoint_fingerprints.txt` records both the shared base
  checkpoint's and each arm's own checkpoint's SHA-256).

## 4. Training stability / best_epoch

`best_epoch=1` (`val_overlap@10=0.0083`), early-stopped at epoch 6
(patience 5 exhausted). `val_overlap@10` trajectory: 0.0083 → 0.0070 →
0.0069 → 0.0042 → 0.0048 → 0.0059 — declines after epoch 1 and never
recovers, matching the `best_epoch=1`-then-decline pattern already seen in
every other arm this session (R1, R2, C1, C2). C0's own `best_epoch` was
also 1 (`val_overlap@10=0.008228`) — so on this specific metric E1 and C0
are statistically indistinguishable at their respective best epochs
(0.0083 vs. 0.0082), even though E1's underlying representation has already
changed drastically by that point (see §5).
`wall_clock_seconds=4631.1` (~77 min), `peak_gpu_memory_mib=515` — no OOM,
confirming the memory-safe streaming design holds at full scale.

## 5. Encoder representation drift

Fixed-probe diagnostics (channel 0, first 256 memory rows,
`utils.rank_losses.embedding_geometry`, same convention as
`results/EXP-SEQDIAG01/effective_rank_diagnostic.py`):

| | B0 (frozen ref.) | E1 epoch 1 (selected) | E1 epoch 3 | E1 epoch 6 (final) |
|---|---:|---:|---:|---:|
| `embedding_effective_rank` | 17.48 | 2.96 | 1.89 | 1.49 |
| `embedding_effective_rank_ratio` (of d_model=128) | 0.137 | 0.023 | 0.015 | 0.012 |
| `embedding_pairwise_cosine_mean` | 0.474 | 0.993 | 0.988 | 0.915 |

A separately-logged per-epoch drift probe (`encoder_drift.csv`, computed
DURING training) corroborates this: L2 drift from the B0 embedding grows
monotonically (0.826 → 0.931 epoch 1→6) and cosine-similarity-to-B0 falls
monotonically (0.650 → 0.559) — the representation moves steadily further
from B0 for as long as training continues, with no sign of settling within
the 6 epochs trained.

## 6. Collapse assessment

**Severe, monotonically-deepening representation collapse, already present
after a single epoch.** `embedding_effective_rank` drops by ~83% (17.48 →
2.96) after just 1 epoch and keeps falling for as long as training
continues (→ 1.89 at epoch 3, → 1.49 at epoch 6) — this is on the same
order of severity as `EXP-SEQDIAG01`'s own finding for a trainable encoder
under a (different, one-hot cross-entropy) full-memory sequential
objective (B0=19.576 → collapsed=4.297 on Weather), now replicating under
R2's more careful hybrid ranking+regression loss and a cosine scorer that
was specifically NOT the cause of prior negative results
(`EXP-ASYM-SCORER01`, `EXP-STRONG-SCORER-DIAG01`). `embedding_dead_dimension_fraction`
stays exactly 0.0 throughout — this is not individual dimensions dying, but
the WHOLE embedding population collapsing toward a shared direction
(`pairwise_cosine_mean` 0.474 → 0.99+), consistent with the "satisfy a
margin/ranking loss by degenerating the space until nothing is comparable"
failure mode `embedding_geometry`'s own docstring describes. Per the user's
explicit instruction, no collapse-prevention regularisation was added in
response to this finding.

## 7. t=1 diagnostics (first-anchor)

Test split, C0 vs. E1:

| metric | C0 (frozen) | E1 (trainable) | Δ (E1−C0) |
|---|---:|---:|---:|
| Spearman (true top-1%) | 0.2128 | 0.1455 | worse |
| oracle-best predicted rank (median) | 99 | 210 | worse |
| Top-50 containment | 30.5% | 20.5% | worse |

E1 train split (E1 only — C0 is reused/not retrained, no C0-train
evaluation exists): Spearman=0.0732, oracle rank median=1218, Top-50=8.5% —
**worse than E1's own test split**, the same train-worse-than-test
direction EXP-STRONG-SCORER-DIAG01's C2 showed (opposite of classic
overfitting), here plausibly explained the same way: `best_epoch=1` means
the selected checkpoint reflects only 1 epoch's worth of (already-collapsing)
adaptation, not a model that has had the chance to specifically overfit
the training distribution.

## 8. t=2 exhaustive continuation diagnostics

Test split, C0 vs. E1:

| metric | C0 (frozen) | E1 (trainable) | Δ (E1−C0) |
|---|---:|---:|---:|
| hurt_frac | 0.232 | 0.194 | **better** |
| selected true rank (median) | 1934.0 | 539.5 | **better** |
| oracle predicted rank (median) | 560.0 | 318.0 | **better** |
| Spearman (true top-1%) | 0.1404 | 0.1358 | ≈ (worse, small) |
| continuation_regret | 0.3498 | 0.3987 | worse |

This is the most internally mixed result in the whole report: E1 is
noticeably BETTER than C0 on `hurt_frac` and both rank-median metrics, but
WORSE on `continuation_regret` (the metric that most directly reflects
downstream utility loss, not just rank position) and roughly flat on
top-tail Spearman. E1's own train split is worse than its test split here
too (hurt_frac 0.242, spearman 0.082, regret 0.287 — regret is actually
*better* on train, the one exception to the train-worse-than-test pattern
in this report, likely reflecting `mean_A1`'s different scale between
splits rather than a genuine train/test asymmetry in continuation quality;
see `t2_continuation.json` for full train-split numbers). This mixed
picture is reported as-is rather than forced into a single direction.

## 9. HardAggregateMSE@10 / gap_recovery

| metric | C0 (frozen) | E1 (trainable) | Δ (E1−C0) |
|---|---:|---:|---:|
| HardAggregate@10 (seq) | 0.5514 | 0.5466 | **better** (small) |
| gap_recovery | -0.2626 | -0.3703 | worse |

Another internally mixed pair: E1 is marginally BETTER on the unweighted
HardAggregateMSE but clearly WORSE on `gap_recovery` (which uses B0's own
`a_weighted` aggregation, the metric this project's Stage-2 evaluation
protocol treats as primary for gap-closing claims).

## 10. Stage-2 evaluation

| metric | C0 (frozen) | E1 (trainable) | Δ (E1−C0) |
|---|---:|---:|---:|
| Stage-2 MSE | 0.39526 | 0.40668 | worse |
| Δ vs B0 (0.37312) | +0.02214 | +0.03355 | worse |
| gap_recovery | -0.2626 | -0.3703 | worse |

E1 is worse than C0 on every Stage-2 metric. For reference, this places E1
between C0 (best) and C1/EXP-ASYM-SCORER01 (worst, Δ=+0.03945,
gap_recovery=-0.488) — worse than C0 and C2 (strong scorer, Δ=+0.02396),
better than C1 (asymmetric scorer). Stage-2 architecture/gate/fusion were
untouched (per spec); `E1_CKPT`'s own `model_state_dict` (including the now
-adapted encoder) is what the evaluation script loads and uses for both
query and candidate encoding, verified by the checkpoint fingerprint in
`checkpoint_fingerprints.txt` differing from the shared B0/C0 checkpoint's
fingerprint (E1's encoder is not silently falling back to the frozen B0
encoder).

## 11. Closest outcome (A / B / C / D)

**Outcome D — end-to-end training destabilizes/collapses the
representation — is the best-supported reading**, though the downstream
metrics are not uniformly bad, which is itself informative:

- Not Outcome A (clear improvement): Stage-2 MSE and `gap_recovery` are
  both worse than C0, not better — the two metrics the spec explicitly
  weights most heavily for concluding a frozen-representation bottleneck
  was resolved.
- Not Outcome B (teacher-forced/ranking improves but free-running does
  not): t1 top-tail ranking is uniformly WORSE than C0 (Spearman, oracle
  rank, Top-50 all worse), not better — there is no local-ranking gain to
  contrast against a free-running failure.
- Not Outcome C (encoder moves but everything stays flat): the encoder
  moved a great deal (§5), and several metrics moved too (worse on
  Stage-2/gap_recovery/t1, better on HardAggregate/some t2 rank metrics) —
  "flat" does not describe what happened here.
- **Outcome D fits the primary, cleanest signal**: `embedding_effective_rank`
  collapses by ~83% within 1 epoch and keeps deepening every epoch trained
  (17.48 → 2.96 → 1.89 → 1.49), `pairwise_cosine_mean` rises to 0.99+ (the
  candidate population becomes nearly indistinguishable), and the two
  metrics most tied to this project's own definition of "did this help"
  (Stage-2 MSE, `gap_recovery`) are both worse than the frozen baseline.
  The t2 rank-median improvements (§8-9) are real but do not change this
  reading: they are consistent with a collapsed representation that
  happens to place the free-running selector's picks in a different,
  sometimes locally-favorable region of an otherwise degenerate space,
  not with a genuinely improved representation — `continuation_regret`
  (arguably the more decision-relevant of the two t2 metrics, since it
  measures realized utility loss rather than rank position) and Stage-2
  MSE both point the other way.

## 12. Is an encoder depth/capacity experiment now justified?

**No — per the spec's own decision rule, Outcome D does not justify a
depth/width/capacity follow-up.** The evidence here indicates that simply
letting THIS encoder architecture adapt end-to-end under this
sequential/teacher-forced set-utility objective is unstable and
representation-destroying, not that the current architecture lacks
capacity. A deeper or wider encoder trained the same (uncontrolled) way
would be expected to collapse the same way, plausibly faster. Per the
user's explicit stopping rule, no collapse-prevention regularisation,
partial-layer unfreezing, encoder LR tuning, or depth/width/architecture
sweep was started in response to this finding — this report and
`research/REVIEW_FOR_CHATGPT.md` are handed to the independent review step
instead.

---

### Primary result table (test split, ETTh1 H96)

| Metric | C0 Frozen Encoder | E1 Trainable Encoder |
|---|---:|---:|
| best_epoch | 1 | 1 |
| val_overlap@10 (best) | 0.008228 | 0.008300 |
| encoder drift from B0 (cosine, epoch of best ckpt) | — (frozen, =1.0) | 0.650 |
| embedding_effective_rank (epoch of best ckpt) | 17.48 | 2.96 |
| t1 top-1% Spearman | 0.2128 | 0.1455 |
| t1 oracle rank median | 99 | 210 |
| t1 Top-50 containment | 30.5% | 20.5% |
| t2 selected true rank median | 1934.0 | 539.5 |
| t2 oracle predicted rank median | 560.0 | 318.0 |
| t2 top-1% Spearman | 0.1404 | 0.1358 |
| t2 continuation regret | 0.3498 | 0.3987 |
| HardAggregate@10 | 0.5514 | 0.5466 |
| gap_recovery | -0.2626 | -0.3703 |
| Stage-2 MSE | 0.39526 | 0.40668 |

Source files: `stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`train_vs_val_test.json`, `representation_diagnostics.json`,
`encoder_drift.csv`, `metrics.csv`, `train_summary.json`,
`checkpoint_fingerprints.txt`, `working_tree.diff`, `sanity_summary.json`,
`command.txt`, `config.json`.
