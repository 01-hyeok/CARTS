# AUDIT — EXP-ORACLE-RANK-GAIN01 (and cross-check vs EXP-ORACLE-SCRATCH01)

Date: 2026-09-10. Scope: correctness audit only. **No experiment code,
runner, checkpoint, cache or result file was modified during this audit.**
All new computations were written to `/tmp/audit_rg01/` only.

Every claim below is backed by a command that was actually run against
real artifacts (tensors, SHA/equality checks, file mtimes, git history, or
a re-execution). Docstrings/comments/README text were NOT treated as
evidence.

---

# 1. Executive Summary

| # | Audit item | Verdict |
|---|---|---|
| 1 | H720 Base-only discrepancy (0.76928 vs 0.48279) root cause | **FAIL (real defect found, fully explained)** |
| 2 | Stated cause in `REVIEW_FOR_CHATGPT.md` ("different random init") | **FAIL (documented explanation is factually wrong)** |
| 3 | `REVIEW_FOR_CHATGPT.md` SCRATCH01 H96-vs-H720 comparison | **FAIL (mixes corrected H96 with uncorrected H720)** |
| 4 | Stage-1 shared encoder init identical across 8 arms | **PASS** (bit-identical, both horizons) |
| 5 | Epoch0 checkpoint is genuinely pre-training | **PASS** |
| 6 | Asymmetric identity init (Wq=Wk=I) | **PASS** (exactly I) |
| 7 | Individual TF vs On-policy really differ | **PASS** (numerically demonstrated) |
| 8 | Set TF vs On-policy prefix construction | **PASS** |
| 9 | Rank metric convention / "top X%" interpretation | **PASS** (1-based, rank/N_valid) |
| 10 | Epoch0-vs-Best diagnostics use same queries/support | **PASS** (shuffle=False, first-N) |
| 11 | Full-memory scoring (no shortlist) | **PASS** (8449/8449) |
| 12 | Future leakage into model input / free-running selection | **PASS** (structurally impossible) |
| 13 | Retrieval cache ↔ checkpoint mapping (stale cache?) | **PASS** (regenerated exactly) |
| 14 | Y_ret aggregation formula | **PASS** (brute-force exact) |
| 15 | Stage-2 Base-only path (`Y_final == Y_base`) | **PASS** (exactly 0 diff) |
| 16 | Stage-2 trainable params / checkpoint selection | **PASS** |
| 17 | Stage-2 determinism / reproducibility | **PASS** (bit-exact reproduction) |
| 18 | Set Oracle target stationarity across checkpoints | **WARNING (endogenous target, by design)** |
| 19 | Set-Onpolicy high t>=2 ranking | **WARNING (measurement artifact, quantified — not a code bug)** |

### Direct answers to the three headline questions

**"Can we trust the current H96 results?"** — **YES, with one qualification.**
The H96 Stage-1 pipeline, Epoch0 baseline, caches, Stage-2 protocol and
Stage-2 numbers all pass every mechanical check. The qualification is
item 18/19: the *Set* arms' t>=2 ranking numbers (both TF and On-policy)
are measured against a target that is itself a function of the current
encoder, and the On-policy Set arms additionally suffer a self-referential
coupling. Individual arms are unaffected.

**"Can we trust the H720 Stage-1 ranking results?"** — **YES**, same
qualification as H96. Stage-1 for H720 was never touched by the defect in
finding #1 (that defect is Stage-2-only).

**"Can we trust the H720 Stage-2 MSE?"** — **YES for
EXP-ORACLE-RANK-GAIN01** (it used the correct protocol, is bit-exactly
reproducible, and its central claim was independently corroborated — see
§8). **NO for EXP-ORACLE-SCRATCH01's H720 Stage-2** — those numbers were
produced with unauthorized hyperparameters and are invalid.

---

# 2. H720 Base discrepancy: 0.76928 vs 0.48279 — ROOT CAUSE FOUND

## 2.1 What the two runs actually did

| Field | SCRATCH01 H720 base_only | RANK-GAIN01 H720 base_only |
|---|---|---|
| test MSE | 0.7692834184932135 | 0.4827854310622209 |
| epochs actually run | **20** | **10** |
| best_epoch | 10 | 6 |
| val_mse range | **[2.03, 20.60]** (10x swings) | [1.52, 1.81] (stable) |
| trainable params | 519120 | 519120 (identical) |
| implied hyperparameters | lr=0.01, epochs=50, patience=10 | lr=0.001, epochs=10, patience=5 |

The SCRATCH01 val curve contains `val_mse = 20.599` at epoch 9 followed by
`2.033` at epoch 10 — divergent, non-converging optimisation, the classic
signature of a too-large learning rate. Its "best" checkpoint is simply
whichever epoch of a random walk happened to land lowest.

## 2.2 Everything else is bit-identical (so init/data are ruled out)

| Compared artifact | Result |
|---|---|
| `shared_base_init` weight/bias tensors | **exact_equal = True**, max_abs_diff = 0.0 |
| `shared_gate_init` (all 4 tensors) | **exact_equal = True**, max_abs_diff = 0.0 |
| cache `batch_x` (train/val/test) | **exact_equal = True**, max_abs_diff = 0.0 |
| cache `Y_q` (train/val/test) | **exact_equal = True**, max_abs_diff = 0.0 |
| cache `Y_ret` | differs (expected — different Stage-1 encoders; **irrelevant**, Base-only zeroes it) |
| architecture / param count | identical (519120) |

(The two init *files* have different SHA256 only because of pickle
metadata; the tensors they contain are bit-identical.)

## 2.3 Causal proof by re-execution

Same code, same saved init, same cache — only lr/epochs/patience changed:

| Test | Protocol | Result |
|---|---|---|
| A | lr=0.001, 10 ep, pat 5 | test_mse = **0.48279**, best_epoch=6 — reproduces RANK-GAIN01 **bit-exactly, every epoch's train/val MSE matching** |
| B | lr=0.01, 50 ep, pat 10 | test_mse = **0.76928**, best_epoch=10, early stop @20 — reproduces SCRATCH01 **bit-exactly, including the `val_mse=20.599` spike at epoch 9** |

**Conclusion: 100% of the 0.2865 MSE gap is attributable to the Stage-2
training protocol. Initialization, data, cache and code version are ruled
out by direct tensor equality and by exact reproduction of both numbers.**

## 2.4 Why SCRATCH01's H720 used the old protocol despite the fix

Timeline from file mtimes and result timestamps:

| Time | Event |
|---|---|
| ~09:0x | `run_oracle_scratch01.sh` launched (orchestrator pid 2220732) |
| **10:57:57** | the runner file is edited (lr 0.01→0.001, 50→10 ep, pat 10→5) |
| 10:59:27 | H96 Stage-2 re-run completes — **correct** params (ran via a *separate, newly-invoked* script `rerun_stage2_h96.sh`) |
| **14:38:07** | SCRATCH01 H720 Stage-2 results written — **3h40m after the edit**, yet with 20-epoch/old-protocol history |

Mechanism (verified with a minimal reproduction): **bash parses a shell
function's entire body into memory when the definition is read, so editing
the file afterwards does not affect the already-running process.** Demo:
a script defining `myfunc(){ echo VALUE=ORIGINAL; }`, slept, then edited on
disk to `VALUE=EDITED`, still printed `VALUE=ORIGINAL`.

Because `run_horizon()` is defined at the top of `run_oracle_scratch01.sh`
and invoked at the bottom, the entire H96 **and H720** loop bodies were
already fixed in memory at launch. The `sed` edit only affected *future*
invocations of the script.

Classification per the audit's own options: **(A) old experiment ran a
different protocol — but unintentionally, because the in-flight fix could
not take effect.** Not stale artifacts, not a mixed-up summary, not a
later runner edit.

## 2.5 Impact

- **All 5 of SCRATCH01's H720 Stage-2 numbers are invalid** (`individual_cosine` 0.76144, `individual_asymmetric` 0.76953, `set_cosine` 0.72582, `set_asymmetric` 0.66269, `base_only` 0.76928). All ran 20–25 epochs with val ranges up to [1.91, 23.16].
- SCRATCH01's **H96** numbers as quoted in `REVIEW_FOR_CHATGPT.md` (0.38059 etc.) are the *corrected* `stage2_fixed/` values and are fine.
- Therefore `REVIEW_FOR_CHATGPT.md` currently compares **corrected H96 against uncorrected H720** and concludes that the Individual-vs-Set winner "flips between horizons". **That conclusion is not supported.**

---

# 3. Stage-1 audit

| Check | Method | Result |
|---|---|---|
| 8 arms share one encoder init (H96) | SHA + `torch.equal` on every `encoder.*` tensor of all 8 `checkpoint_epoch0.pth` | **PASS** — all 8 hash `9514e590...`, all match `shared_encoder_init_ETTh1_H96.pth` |
| 8 arms share one encoder init (H720) | same | **PASS** — all 8 hash `2a7dc520...` |
| Epoch0 is pre-training | epoch0 encoder == shared init file (tensor equality) **and** epoch0-vs-best encoder differs | **PASS** — `max_abs_diff(ep0, best)` = 2.4e-1 / 3.7e-1 / 8.0e-2 for the arms checked, i.e. training genuinely moved the weights |
| Asymmetric identity init | `Wq`, `Wk` vs `I` | **PASS** — `Wq==I: True`, `Wk==I: True`, `‖Wq−I‖=0.00e+00` for all 4 asym arms |
| SetConditioner init parity (TF vs OP) | tensor equality | **PASS** — identical for both cosine and asymmetric pairs |
| Individual TF ≠ Individual On-policy | toy case where model argmax ≠ Oracle order | **PASS** — TF picks `[3,11,7,8]` (= Oracle order), On-policy picks `[4,8,2,3]` (= model score order); trajectories differ |
| Set TF vs On-policy prefix | unit tests + target recomputation per step | **PASS** (`tests/test_exp_oracle_rank_gain01.py` items 12/13) |
| Full memory, no shortlist | candidate dim of score tensor vs memory size | **PASS** — 8449 candidates scored, `valid_counts min/max = 8449/8449` |
| Future leakage into model input | signature + body inspection of `free_running_topk` | **PASS** — signature is `(z_q, E, cand_mask, target, set_conditioner, metric, k)`; no future/Oracle tensor is even passed in |

---

# 4. Evaluation audit

## 4.1 Rank metric convention (verified on a synthetic 4-candidate case)

`oracle_rank_statistics` is **1-based**, and
`rank_fraction = rank / N_valid`:

| oracle candidate | rank | rank_fraction |
|---|---|---|
| best-scored | 1 | 0.25 (=1/4) |
| 2nd | 2 | 0.50 |
| 3rd | 3 | 0.75 |
| worst | 4 | 1.00 |

So reporting `rank_fraction = 0.05` as "top ~5%" is **correct**. Caveat for
the write-up: the best attainable value is `1/N` (≈0.00012 at N=8449), not
0, and the random-chance expectation is ≈0.5.

## 4.2 Epoch0-vs-Best comparability

`per_step_diagnostics_for_split` uses `exp._get_data(flag=split,
shuffle=False)` and stops at `rows_seen >= n_queries`, so Epoch0 and Best
consume **the same first-500 queries in the same order**, with the same
candidate mask and memory. **PASS.**

## 4.3 Set Oracle target is NOT stationary across checkpoints — WARNING

The Set Oracle target is `argmin_i MSE(Aggregate(prefix ∪ {i}), Y_q)`, and
`Aggregate` weights candidates by `alpha ∝ exp(b_i / tau_topk)` where
`b_i` is **the current model's own score**. `no_grad`/`detach` stop
gradients but do **not** freeze the target definition.

Measured on `set_tf_cosine`, ETTh1 H96, a real test batch, **holding the
prefix fixed** so only the checkpoint differs:

| Quantity | Epoch0 vs Best |
|---|---|
| t=1 target match rate | **1.0000** (stationary — with an empty prefix `dense_utility` reduces to `MSE(Y_i,Y_q)`, independent of `b_i`) |
| t=2 target match rate (same prefix) | **0.3750** |
| aggregation weight `w_base` | max_abs_diff = 8.63e-01 |

**Implication:** for Set arms, "Epoch0 → Best rank improvement" at t>=2 is
*not* "the model learned a fixed target better" — 62.5% of the t>=2 target
labels themselves changed. This affects **both** Set-TF and Set-Onpolicy.
Individual arms are unaffected (their target `-MSE(Y_i,Y_q)` never depends
on the model).

## 4.4 Set-Onpolicy t>=2 artifact — mechanism quantified — WARNING

Previously observed at Epoch0: `cos_sim(h_t, z_q) = 0.858` and
`cos_sim(conditioned score, b_i) = 0.995`, i.e. an untrained
residual-connected SetConditioner is nearly a no-op, so the t>=2 score
collapses onto the unconditioned base score `b_i`.

New measurement — **where does the Oracle target sit in the model's own
`b_i` ranking at Epoch0** (N = 8449 valid candidates, random expectation
≈ rank 4224):

| Prefix used | Oracle target's median `b_i` rank | = top |
|---|---:|---:|
| **On-policy** (prefix = model's own top-`b_i` picks) | **26** | **0.31%** |
| **TF** (prefix = true-future-based Oracle greedy) | 212 | 2.51% |

The sharp `tau_topk=0.1` softmax makes the aggregate dominated by
high-`b_i` candidates, so the Oracle target is biased toward high-`b_i`
candidates under *any* prefix — but when the prefix is *itself* built from
the model's top-`b_i` picks, that bias is ~8x stronger. The model then
"finds" the target trivially, because both sides of the comparison are
driven by the same random `b_i`.

**Classification: (B) intended design but endogenous metric — NOT an
implementation bug.** The code faithfully implements the pre-registered
spec (§9/§10: aggregation weight must use the BASE score). The defect is
in what the resulting metric *means*, not in the code.

---

# 5. Retrieval cache audit

| Check | Method | Result |
|---|---|---|
| cache built by the claimed checkpoint | regenerated `Y_ret` for a real test batch from `checkpoint.pth` of `set_tf_cosine` and compared to the stored cache | **PASS** — `max_abs_diff = 0.000e+00` (no stale/mismatched cache) |
| cache `batch_x` aligned with loader order | `torch.equal` against a fresh `shuffle=False` loader batch | **PASS** |
| `Y_ret` aggregation formula | independent brute-force `alpha = softmax(b_i/tau)` over the selected Top-K, then weighted sum of offset-adjusted futures | **PASS** — alpha diff 0.000e+00, Y_ret diff 0.000e+00 |
| free-running selection is Oracle/future-free | function signature contains no future tensor | **PASS** |

---

# 6. Stage-2 audit

| Check | Result |
|---|---|
| Base-only: `Y_final == Y_base` | **PASS** — max diff `0.000e+00`, verified twice: (a) `fixed_lambda=0` with deliberately non-zero `Y_ret`, (b) the real `--no_retrieval` path with zeroed `Y_ret` |
| Base-only excludes gate from optimizer | **PASS** — `trainable_params = base_head.parameters() + (gate.parameters() if not no_retrieval else [])` |
| Base-only MSE == MSE(Y_base, Y_q) | **PASS** — independent recomputation from the saved checkpoint: `0.48278542` vs summary `0.48278543`, diff `7.9e-09` |
| Stage-1 cannot receive gradient | **PASS** — Stage-1 is baked into the cache; `Exp_Stage1_Relation`/`RelationEncoder` are not imported by the Stage-2 script |
| checkpoint selection on validation MSE | **PASS** (test MSE computed only after best-checkpoint reload) |
| shared Base/Gate init across arms | **PASS** (same files consumed via `--shared_*_init_in`) |
| determinism | **PASS** — Test A reproduced every epoch's train/val MSE bit-exactly |

---

# 7. Cross-experiment provenance

Both experiments run the **same** Stage-2 script
(`scripts/train_oracle_scratch01_stage2.py`) — RANK-GAIN01 deliberately
reuses it unmodified. The only difference is the *arguments the runner
passed*:

| | SCRATCH01 H720 | RANK-GAIN01 H720 |
|---|---|---|
| runner | `run_oracle_scratch01.sh`, **as parsed at launch** (pre-fix) | `run_oracle_rank_gain01.sh` (launched after the fix) |
| lr / epochs / patience | 0.01 / 50 / 10 | 0.001 / 10 / 5 |
| Base init | *bit-identical to RANK-GAIN01's* | — |
| cache `batch_x`/`Y_q` | *bit-identical to RANK-GAIN01's* | — |

The current on-disk `run_oracle_scratch01.sh` shows `lr 0.001` (commit
`f47b2ea`), which is why reading the script today gives no hint that the
H720 results came from a different protocol. **This is precisely why the
audit's "don't trust the script text, check the artifacts" rule mattered.**

---

# 8. Minimal reproduction results

## 8.1 H720 Base-only, protocol isolation (identical init + cache + code)

| Run | lr / ep / patience | test MSE | Matches |
|---|---|---:|---|
| A (current protocol) | 0.001 / 10 / 5 | **0.48279** | RANK-GAIN01, bit-exact |
| B (old protocol) | 0.01 / 50 / 10 | **0.76928** | SCRATCH01, bit-exact |

The requested old/new × init/cache 2×2 matrix collapses to a 1-D test:
because the inits and the Base-only-relevant cache contents (`batch_x`,
`Y_q`) were proven **bit-identical**, "old init" and "old cache" are not
distinct conditions at all. The protocol is the only free variable, and it
fully explains the gap.

## 8.2 Does "retrieval hurts at H720" survive? — re-ran SCRATCH01's own H720 arms under the CORRECT protocol

Using SCRATCH01's **own** Stage-1 arms/caches (i.e. different encoders,
different retrieved sets than RANK-GAIN01's) with lr=0.001/10ep/pat5:

| SCRATCH01 H720 arm | invalid (old protocol) | **re-run (correct protocol)** |
|---|---:|---:|
| Individual+Cosine | 0.76144 | 0.51373 |
| Individual+Asymmetric | 0.76953 | 0.53339 |
| Set+Cosine | 0.72582 | 0.52239 |
| Set+Asymmetric | **0.66269** ("beats Base") | 0.56679 |
| Base-only | 0.76928 | **0.48279** |

Under the correct protocol, **all four SCRATCH01 retrieval arms are worse
than Base-only**, exactly as RANK-GAIN01 found — and SCRATCH01's Base-only
lands on the identical 0.48279 (expected: identical Base init + identical
`batch_x`/`Y_q`).

**Two consequences:**
1. SCRATCH01's "Set+Asymmetric beats Base at H720" is an artifact of
   divergent lr=0.01 training. It disappears entirely.
2. RANK-GAIN01's "at H720 every retrieval arm is worse than Base" is
   **independently corroborated** by a second, differently-initialised set
   of Stage-1 encoders. This raises rather than lowers confidence in that
   finding.

---

# 9. Valid / Confounded / Invalid conclusions

### VALID (survives audit)
- **H96 Stage-2 ranking of arms** (RANK-GAIN01): all 8 arms beat Base-only (0.39298), Individual+Asymmetric best (0.38059). Protocol correct, reproducible, caches verified.
- **Epoch0 → Best ranking gain for Individual arms** (both horizons, both prefix policies). Their target is model-independent and stationary; shared init and query subsets verified. H96: top ~15% → top ~4.5–5.4%.
- **H720: retrieval does not beat the Base Forecaster** — now supported by two independent Stage-1 populations under the correct protocol.
- **All mechanical pipeline properties**: shared init, Epoch0 provenance, identity init, full-memory, no leakage, cache↔checkpoint mapping, Y_ret formula, Base-only path, determinism.

### CONFOUNDED (numbers are real, interpretation is not safe)
- **Set arms' t>=2 Epoch0→Best "gain"** (both TF and On-policy): the target moved (only 37.5% label agreement at t=2 under a fixed prefix). Do not phrase as "learned a fixed Oracle ranking".
- **Set-Onpolicy t>=2 rank_fraction / Top-K containment** (e.g. Top-10 = 0.706): endogenous self-reference; the Oracle target sits at the top 0.31% of the model's own `b_i` ranking at Epoch0 *before any training*. Not comparable with the other six arms.
- **Set-Onpolicy checkpoint selection** (`best_epoch=1`, 6 epochs): selection metric partly rewards staying near the artifact-producing initialisation.

### INVALID (must not be cited)
- **All EXP-ORACLE-SCRATCH01 H720 Stage-2 numbers** (0.76144 / 0.76953 / 0.72582 / 0.66269 / 0.76928).
- **"Set Oracle wins at H720 while Individual wins at H96 — the ranking flips by horizon"** (`REVIEW_FOR_CHATGPT.md`, EXP-ORACLE-SCRATCH01 section and its cross-horizon summary).
- **"The SCRATCH01-vs-RANK-GAIN01 H720 difference is most likely due to different random Base/Gate initialization"** (`REVIEW_FOR_CHATGPT.md`). The inits are bit-identical; the cause is the Stage-2 protocol.

### UNKNOWN / not settled by this audit
- *Why* retrieval fails to help at H720 while helping at H96 (the finding is now solid; the mechanism is not established).
- Whether a non-endogenous Set-Oracle target definition (e.g. weights frozen from a fixed reference scorer) would change the Set arms' conclusions.

---

# 10. Required reruns (minimal)

No full 32-run rerun is warranted. Stage-1 is untouched by every defect
found; only Stage-2 numbers and two documents are affected.

| Priority | Action | Cost |
|---|---|---|
| 1 | **Documentation fix only** (no compute): correct the two false statements in `REVIEW_FOR_CHATGPT.md` (§9 INVALID list) and mark SCRATCH01's H720 Stage-2 table as produced under an unauthorized protocol. | 0 |
| 2 | Adopt the §8.2 re-run numbers as SCRATCH01's corrected H720 Stage-2 results (already computed during this audit, currently in `/tmp/audit_rg01/`; would need re-running into a permanent `stage2_fixed/` path to match the H96 convention). | 5 runs ≈ 1 min |
| 3 | *Optional, for the endogenous-target concern*: re-evaluate Set arms' Epoch0/Best diagnostics with `w_base` frozen to a single fixed reference scorer, so the target is stationary. Diagnostic-only, no retraining. | 16 eval runs, no training |
| 4 | Nothing else. Weather cells currently running are unaffected (they use the correct runner). | — |

**Awaiting approval before performing any of items 2–3.** Per the audit
brief, no code, result or document was modified during this audit.

---

# 11. Answers to the seven closing questions

1. **Is the code actually wrong?** No. Every mechanical check on
   `train_oracle_rank_gain01.py`, `eval_oracle_rank_gain01.py`,
   `train_oracle_scratch01_stage2.py` and the caches passed. The defect
   was **operational** (an in-flight `sed` edit to a running bash script
   that could not take effect), plus **two incorrect statements in the
   write-up**.
2. **If wrong, exactly where?** `research/REVIEW_FOR_CHATGPT.md`: (a) the
   "different random initialization" explanation, (b) the SCRATCH01
   cross-horizon flip claim built on corrected-H96 vs uncorrected-H720.
   And all SCRATCH01 H720 Stage-2 artifacts, which were generated with
   lr=0.01/50ep/pat10.
3. **Code right but protocols differed?** **Yes — this is the answer.**
   Proven by bit-exact reproduction of both 0.48279 and 0.76928 from
   identical init + identical cache + identical code, varying only
   lr/epochs/patience.
4. **Is H720 Base-only 0.48279 reproducible?** **Yes, bit-exactly**,
   including every epoch's train/val MSE, and independently recomputable
   from the saved checkpoint to 7.9e-09.
5. **Can we believe "8/8 retrieval arms worse than Base at H720"?**
   **Yes.** It reproduces under the correct protocol on a *second*,
   independently-initialised set of Stage-1 encoders (SCRATCH01's), where
   all 4 arms also land above Base-only.
6. **Are the Stage-1 ranking-gain results trustworthy?** **Yes for
   Individual arms** (stationary target, verified shared init and query
   subsets). **Qualified for Set arms** — real numbers, but measured
   against a checkpoint-dependent target (§4.3).
7. **Set-Onpolicy's high t>=2 ranking: bug, artifact, or capability?**
   **Measurement artifact from an endogenous (self-referential) metric —
   not a bug, and not capability.** Quantified in §4.4: at Epoch0, before
   any training, the Oracle target already sits in the top 0.31% of the
   model's own score ranking under an on-policy prefix, versus 2.51% under
   a TF prefix and ~50% for a genuinely independent target.
