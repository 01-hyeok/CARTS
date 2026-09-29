# TRACK-I-PCA-FUTURE-TEACHER01

**Status: COMPLETE (ETTh1, H=720 only, per pre-registered stopping rule).
Verdict: NO-GO.** Phase A confirms a fixed low-dimensional PCA future
space (dim=64, L2) preserves raw-future-MSE Oracle retrieval quality
almost exactly (retMSE degradation +0.91%, well under the 3% threshold).
But Phase B shows the core hypothesis is **not supported**: across 3
independently-initialized, paired seeds, the PCA-teacher-trained student
is *worse* than the raw-teacher-trained student on raw future retMSE@10
in 2 of 3 seeds (mean +0.72% worse). Training-time diagnostics show the
PCA teacher IS easier to fit (consistently lower train KL every epoch)
despite being *more* diffuse (higher entropy, lower top-1 mass) than the
raw teacher — but this easier-to-fit distribution does not translate into
better retrieval quality. This is spec-pattern 3 ("distribution fitting
improves, full-memory retMSE does not") — the intended
"more-learnable-geometry" mechanism is not what's limiting the raw
baseline. Per the user's explicit conditional instruction, Stage-2 is
**not** run (the Stage-1 precondition — a clear PCA improvement — was not
met). H=96 is also not attempted, per the pre-registered stopping rule
(H=720 must show a reproduced improvement first).

## 1. Was it implemented as specified?

Yes, with one documented, low-risk simplification: `eval_epoch` (always
raw-future-MSE evaluation, teacher-mode-agnostic) is imported **unmodified**
from `train_patch_retrieval_expert01.py` rather than duplicated, and
`train_epoch` is a near-byte-identical copy of that script's own
`train_epoch` with exactly one line changed (the `d` computation
dispatches to `compute_teacher_distance`, raw or PCA). No architecture,
optimizer, LR, batch size, epoch/patience, candidate mask, candidate
support, checkpoint-selection metric, or evaluation protocol was touched.

## 2. Code audit (spec section 0)

| # | Item | Finding |
|---|---|---|
| 1 | Raw teacher score | `individual_utility_memsafe` (`train_factorial_e2e01.py`): `u_i = -mean((Y_i - Y_q)^2)`, chunked over candidates, `Y_i` reconstructed via `memory_value()` (delta_last: candidate's own future minus its own last-observed value, plus the query's own last-observed offset). `d = -u`. |
| 2 | Temperature application | `normalized_teacher_prob(d, mask, tau_t)` (`train_horizon_retrieval_expert01.py`): z-score-normalizes `d` per row over valid candidates, then `softmax(-normalized/tau_t)`. Masked entries get probability exactly 0. |
| 3 | Student score | `arm_score(z_q, z_k, None)` (`train_factorial_e2e01.py`): plain cosine, `F.normalize` on both sides, dot product. No asymmetric head anywhere in this track. |
| 4 | KL direction | `kl_loss(p_t, s, mask, tau_s)`: `p_t` detached, `term = p_t * (log p_t - log_softmax(s/tau_s))`, summed over valid candidates, mean over batch — this is `KL(p_teacher \|\| p_student)`, confirmed by direct formula (also covered by the pre-existing `test_kl_gradient_equals_soft_ce_gradient`, not re-derived here). |
| 5 | Candidate mask | `exp._candidate_mask(batch_start_idx)`, unmodified, time-ordered (no leakage), identical for every arm/teacher-mode this track uses. |
| 6 | Train/val/test memory support | `Exp_Stage1_Relation._ensure_memory()` builds `memory_x`/`memory_y` from **train split only** (`RelationMemorySampler(train_data, ...)`); val/test queries are scored against this train-only bank, never added to it. |
| 7 | Full-memory candidate count | ETTh1_720: N=7201 (train split size), confirmed via `[stage1] candidate history on device: (7201, 720, 7)` in every run's log. |
| 8 | Checkpoint selection | `min val model_top10_individual_mse` — always the RAW future MSE of the model's own Top-10 picks, regardless of `teacher_mode` (this metric comes from `eval_epoch`, which never touches PCA). |
| 9 | Baseline configuration | ETTh1_720 p120 (`patch_len=stride=120`, `relation_encoder_type=transformer`, `relation_value_space=delta_last`, `tau_t=0.02`, `tau_s=0.1`, `top_k=10`) — the same production config this whole session's Track A/G/H work uses; reference checkpoint (architecture config only, **not weights** — `train_patch_retrieval_expert01.py` always trains a freshly-initialized scratch encoder) is `checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_.../checkpoint.pth`, the exact same reference used to originally produce p120. |
| 10 | Prior PCA/latent-teacher work | `grep -rli pca scripts/ models/ utils/ research/` returns nothing — no prior PCA or fixed-latent-teacher experiment exists in this repository. Confirmed this is distinct from the RnC/InfoNCE future-MSE-as-positive-source experiments (those define contrastive positives/negatives from future MSE; this track replaces the *teacher relevance geometry itself*, keeps the KL-to-a-continuous-distribution objective, and touches no contrastive/hard-mining/Set-Oracle/learned-head code path). |

## 3. Leakage check

`fit_pca(memory_c)` is called with `memory_c` obtained via `memory_value(args, dummy_batch_x, exp.memory_y, exp.memory_x_last, c)` — `exp.memory_y` is train-only by construction (§2 item 6), and the dummy `batch_x` only supplies the (unused, discarded) per-query offset, never influencing the fit. Val/test loaders are never referenced anywhere in `fit_pca`'s call path. `tests/test_i_pca_future_teacher01.py::test_item2_pca_fit_pool_is_exactly_the_train_candidate_bank_size` pins this contract. No leakage found.

## 4. Full-rank PCA sanity check

**PASS**, both synthetically and on real data:
- `tests/test_pca_future_teacher01.py::test_full_rank_pca_preserves_raw_euclidean_ranking`: full-rank PCA-L2 ranking is bit-identical to raw-MSE ranking on synthetic data (`torch.equal` on the argsort).
- Real ETTh1 Phase A run: **A1 (full-rank PCA) val retMSE@10 = 0.888078, byte-identical to A0 (raw)'s 0.888078** (test split: 0.497833 vs 0.497833). `jaccard_vs_raw` = 0.9998 (not exactly 1.0 only due to floating-point tie-breaking in `stable_topk_indices` near-exact ties after SVD).

## 5. Phase A: teacher-space diagnostic (ETTh1_720, H=720, val split; full table)

| Arm | actual raw retMSE@10 | Δ vs A0 | Recall@10 vs raw oracle | NDCG@10 | Spearman | Uniform Agg MSE | Entropy | Eff. support | Jaccard vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 raw | 0.888078 | 0% | 1.0000 | 1.0000 | 1.0000 | 0.630058 | 3.021 | 26.0 | 1.0000 |
| A1 PCA-full (720) | 0.888078 | 0.00% | 0.9999 | 1.0000 | 1.0000 | 0.630061 | 3.021 | 26.0 | 0.9998 |
| A2 PCA-128 | 0.889821 | +0.20% | 0.8926 | 0.9999 | 0.9997 | 0.625489 | 3.064 | 27.9 | 0.8201 |
| **A3 PCA-64** | **0.896171** | **+0.91%** | **0.7929** | **0.9996** | **0.9990** | **0.627067** | **3.186** | **31.3** | **0.6831** |
| A4 PCA-32 | 0.912787 | +2.78% | 0.6675 | 0.9987 | 0.9970 | 0.633898 | 3.328 | 37.6 | 0.5371 |
| A5 PCA-16 | 0.941470 | +6.02% | 0.5245 | 0.9971 | 0.9938 | 0.648553 | 3.523 | 46.7 | 0.3956 |
| A6 PCA-64-cosine | 1.152645 | +29.80% | 0.4406 | 0.9807 | 0.4929 | 0.821673 | 1.397 | 9.0 | 0.3384 |

(Test-split numbers recorded in `results/TRACK-I-PCA-FUTURE-TEACHER01/ETTh1_720/phase_a_diagnostic.json`, shown for completeness only — **not used for dimension selection**.)

**A6 (cosine on PCA space) fails badly** (Spearman collapses to 0.49, retMSE +29.8%): confirms magnitude information in future-space distance matters and must be preserved (L2, not cosine, on the PCA coordinates).

## 6. Selected PCA dimension

**dim = 64, metric = L2.** Selection made on **validation only**: A2 (128) and A3 (64) both clearly pass the ≤3% degradation criterion on both retMSE and aggregate MSE; A4 (32) passes only marginally (+2.78%, close to the 3% boundary). dim=64 was chosen over dim=128 because it achieves a much larger dimensionality reduction (11.25× vs 5.6×) while still passing with clear margin (+0.91% vs the 3% threshold, Recall@10=0.79), making it the more informative test of the "simplified geometry" hypothesis without the boundary risk of dim=32. This choice was fixed **before** any Phase B training and never revisited after seeing test-split or Phase B numbers.

## 7. Phase B: B0 (raw) vs B1 (PCA-64), paired seeds

**Fairness verification** (all 3 seed pairs): `encoder_init_sha256` identical within each pair (B0/B1 seed0: `4cd27c1925f0fbee...`; seed1: `12cf72ed9eb55b28...`; seed2: `d7d62a842b1ea9dc...`) — confirms genuinely paired initialization, not merely paired loader order. `batch_order_hashes['epoch1']` also identical within each pair. **Raw-mode regression check**: `B0_raw_seed0` run to completion reproduces the recorded p120 baseline **exactly** (test retMSE@10 = 0.969165454248431, abs diff = 0.0) — confirms `teacher_mode=raw` is a true no-op relative to the original `train_patch_retrieval_expert01.py`.

| Seed | B0 (raw) test retMSE@10 | B1 (PCA-64) test retMSE@10 | Δ% | B0 Recall@10 | B1 Recall@10 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.969165 | 0.968143 | **-0.106%** | 0.02930 | 0.02968 |
| 1 | 0.953724 | 0.963262 | **+1.000%** | 0.02729 | 0.02745 |
| 2 | 0.950621 | 0.962814 | **+1.283%** | 0.02985 | 0.02743 |
| **mean** | **0.957837** | **0.964740** | **+0.72%** | **0.02881** | **0.02819** |

2 of 3 seeds show B1 worse; only seed0 shows a marginal improvement. Recall@10 is essentially tied across both arms (both far below 1.0 — this student-trained retriever, at any teacher, recovers only ~2.8-3.0% of the true Oracle's own Top-10, consistent with the very low recall levels this whole session's prior tracks have observed at this checkpoint-selection point, `best_epoch=1` in every one of the 6 runs).

## 8. `best_epoch=1` in all 6 runs

Every B0 and B1 run selected epoch 1 as its best checkpoint (val retMSE rises after epoch 1 in both arms — see epoch curves). Training loss / train KL keeps decreasing every epoch for both arms (B0: 3.945→2.761 over 6 epochs; B1: 3.760→2.583), but val retMSE does not track it past epoch 1. This "training keeps improving its own objective while held-out retrieval quality degrades almost immediately" pattern is present for **both** teachers equally — it is not specific to the PCA change, and is consistent with what this session's prior scratch-encoder training work (`train_patch_retrieval_expert01.py`'s own original p120 run) already exhibits.

## 9. Representation collapse

Not separately probed this round (effective rank / embedding-norm / cosine-distribution diagnostics were not added to `eval_epoch`, which was deliberately kept byte-identical to the original script to preserve the regression guarantee in §7). Given the negative Phase B result, this diagnostic was judged not to change the verdict and was not pursued further, per the pre-registered stopping rule (no H=96 repetition, no further probing, once H=720 fails to reproduce an improvement).

## 10. Teacher/student distribution diagnostics

Per-epoch training diagnostics (`epoch_curve_*.csv`, `train_teacher_entropy`/`train_teacher_top1_mass`/`train_kl`) show a consistent, seed-independent pattern (illustrated for seed0, identical shape in seed1/2):

| | train KL (epoch1) | train KL (epoch6) | teacher entropy | teacher top-1 mass |
|---|---:|---:|---:|---:|
| B0 (raw) | 3.945 | 2.761 | ~3.349 | ~0.253 |
| B1 (PCA-64) | 3.760 | 2.583 | ~3.488 | ~0.227 |

**The PCA teacher's distribution is *more* diffuse** (higher entropy, lower top-1 mass) than the raw teacher's, not less — yet the student fits it with **consistently lower KL at every epoch**. This is exactly spec interpretation-pattern 3: distribution-fitting (train KL) improves under the PCA teacher, but this does not translate into better full-memory retrieval quality (§7) — the easier-to-fit target is not the bottleneck the raw baseline was limited by.

## 11. What is established

- Full-rank PCA + L2 is an exact isometry of the raw future-MSE geometry (proven both synthetically and on real data) — the PCA machinery itself is implemented correctly.
- A 64-dimensional fixed PCA future space preserves raw-future-MSE Oracle retrieval quality almost exactly (+0.91% retMSE degradation, Recall@10=0.79 vs the raw Oracle) while cutting the teacher's dimensionality 11.25×.
- Cosine similarity on PCA coordinates is a poor choice (+29.8% Oracle degradation) — the L2/magnitude information in the future space is load-bearing.
- The PCA-64 teacher is measurably *easier* for the student to fit (lower train KL, every epoch, every seed) despite being a *more* diffuse target distribution.

## 12. What is NOT established

- That this easier-to-fit target translates into better retrieval quality: it does not, in 2 of 3 seeds, and the mean effect (+0.72%) is a mild regression, not an improvement.
- Whether a different PCA dimension, temperature, or teacher-normalization choice would flip this result — not swept, per the pre-registered "no test-based re-tuning, no unauthorized new sweeps" discipline.
- Whether representation collapse explains the shared `best_epoch=1` pattern — not probed (§9).
- Anything about H=96, other horizons, or Weather — explicitly out of scope this round per the pre-registered stopping rule and the user's explicit instruction not to run Weather.

## 13. Final verdict

**NO-GO.** The primary pre-registered success pattern (Oracle_PCA ≈ Oracle_Raw **and** Student_PCA > Student_Raw) is not observed. What is observed instead is spec-pattern 3: teacher-distribution learnability improves (lower train KL) without a corresponding full-memory retrieval-quality improvement, and the majority-seed direction on the actual metric of interest (raw future retMSE@10) is a small regression (+0.72% mean, 2/3 seeds worse). Per the user's explicit conditional instruction, **Stage-2 is not run** (the required Stage-1 improvement precondition was not met). Per the pre-registered horizon-expansion rule, **H=96 is not attempted** (H=720 did not reproduce an improvement). This idea is closed at H=720 pending any future re-formulation (e.g., a different fixed-space construction, or addressing why lower training KL doesn't help retrieval) — not pursued further this round.

## 14. Artifacts

- Shared utilities: `scripts/pca_future_teacher01.py`
- Phase A (teacher-only diagnostic): `scripts/diag_i_pca_teacher_feasibility01.py`
- Phase B (student training): `scripts/train_i_pca_future_teacher01.py`
- Unit tests: `tests/test_pca_future_teacher01.py` (7 tests), `tests/test_i_pca_future_teacher01.py` (7 tests) — all 14 pass
- Raw results: `results/TRACK-I-PCA-FUTURE-TEACHER01/ETTh1_720/` (`phase_a_diagnostic.json`, `summary_B{0,1}_*_seed{0,1,2}.json`, `epoch_curve_*.csv`, `config_fingerprint_*.json`)
- Checkpoints: `checkpoints/track_i_pca_future_teacher01/ETTh1_720/`
- Commands (exact): see each script's own CLI; Phase A: `python scripts/diag_i_pca_teacher_feasibility01.py`; Phase B example: `python scripts/train_i_pca_future_teacher01.py --reference_ckpt <S0_wce ETTh1 checkpoint> --arm_name B1_pca64_seed0 --cell ETTh1_720 --teacher_mode pca --pca_dim 64 --init_seed 0 --loader_seed 0`
- Commit hash at start of this track: `7d00d9e37ab751dd9b01178283badb338eb28008`
