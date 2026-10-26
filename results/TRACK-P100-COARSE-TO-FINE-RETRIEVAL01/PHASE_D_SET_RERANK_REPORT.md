# PHASE D -- Set-aware Reranking Within Shared-P100

Candidate support fixed to Shared-P100 throughout. Primary scorer
(spec section 24, selected on VALIDATION only, before any test number
was examined): compared average validation AggMSE@10 across all 4
cells for every deployable scorer Phases A-C produced --

| Scorer | avg VAL AggMSE@10 |
|---|---:|
| canonical V5 (beta=1.0) | 0.9444 |
| Phase A best fixed fusion | 0.9418 |
| **Phase C Residual-V5 (C1)** | **0.9384 (selected)** |
| Phase B adaptive fusion | excluded -- Phase B's own verdict already found no reliable contribution in 3/4 cells, and its per-epoch weights were never checkpointed to disk (only the JSON summary was saved); retraining it solely to compute this comparison was judged not worth the extra GPU time given the already-closed negative verdict. Recorded as a deliberate exclusion, not an oversight. |

**Phase C's Residual-V5 is this track's primary scorer for Phase D**,
using each cell's own already-trained checkpoint, no retraining.

## D0 -- Set Oracle headroom (evaluation-only, cheating oracle)

Within the primary scorer's own fixed Top-20 (reduced from P100,
M=20 never swept): Individual Top-10 (the scorer's own natural
choice) vs a greedy Set Oracle (reads the true future, repeatedly
adds whichever remaining Top-20 candidate most reduces the uniform
-mean aggregate future MSE, K=10 steps). Code:
`scripts/diag_p100_set_oracle01.py`.

| Cell | Split | Individual AggMSE@10 | Set Oracle AggMSE@10 | Oracle gain % |
|---|---|---:|---:|---:|
| ETTh1_96 | val | 0.7374 | 0.5346 | 27.50% |
| ETTh1_96 | test | 0.4259 | 0.2805 | 34.15% |
| ETTh1_720 | val | 1.6870 | 1.4542 | 13.80% |
| ETTh1_720 | test | 0.6174 | 0.4853 | 21.38% |
| Weather_96 | val | 0.5099 | 0.3456 | 32.22% |
| Weather_96 | test | 0.2688 | 0.1223 | 54.51% |
| Weather_720 | val | 0.8191 | 0.6650 | 18.82% |
| Weather_720 | test | 0.5835 | 0.4079 | 30.10% |

Every cell clears the >=5% "meaningful opportunity" bar from spec
section 27 by a wide margin (13.8-54.5%) -- far larger headroom than
Phase A's fixed-beta sweep or Phase B's oracle-beta diagnostic found.
This strongly motivated training D1 regardless of what D0 alone
showed, per the pre-registered rule.

## D1-D3 -- learned set-aware reranker vs mandatory controls

Tiny on-policy reranker (`scripts/train_p100_set_reranker01.py`):
candidate features (primary-scorer/coarse/residual-mix score+rank,
embeddings, cosine, past-only temporal summary) + current-set-summary
features (mean selected embedding/temporal-summary, selected-score
stats, t/K) -> scalar utility, trained via soft listwise KL against a
dense marginal-aggregate-utility target
(`u_i^(t)=AggMSE(S_{t-1})-AggMSE(S_{t-1}U{i})`), ON-POLICY (the
model's own no-grad greedy pick at step t becomes the actual prefix
for step t+1 -- explicitly NOT teacher-forced, to avoid
`EXP-SEQFULL01`/`EXP-MARGUTIL01`'s documented train/inference prefix
-mismatch failure mode; see the script's own docstring for the full
repo-audit record). Checkpoint selected by validation AggMSE@10
(never test), 10 epochs fixed.

| Cell | D0 (no selector) | D1 learned (aggregate target) | D2 (individual target control) | D3 (shuffled set-state control) |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.4259 | 0.4393 | 0.4300 | 0.4353 |
| ETTh1_720 | 0.6174 | 0.6083 | 0.6080 | 0.6030 |
| Weather_96 | 0.2688 | 0.2183 | 0.2245 | 0.2179 |
| Weather_720 | 0.5835 | 0.5607 | 0.5836 | 0.5598 |

(Test AggMSE@10, evaluated once after validation-only checkpoint
selection; D0/D3 need no separate training.)

## Mandatory-control check (section 31: "D1이 D2/D3를 못 이기면 set-aware aggregate utility contribution이 아님")

| Cell | D1 beats D0? | D1 beats D2? | D1 beats D3? | Verdict |
|---|---|---|---|---|
| ETTh1_96 | NO (-3.1%) | NO (-2.2%) | NO (-0.9%) | D1 is the WORST of all 4 arms |
| ETTh1_720 | YES (+1.5%) | ~tie (+0.05%) | NO (-0.9%) | marginal D0-beat, no set-aware-specific contribution |
| Weather_96 | YES (+18.8%) | YES (+2.8%) | NO (-0.2%, essentially tied) | large D0-beat, but D3 matches it |
| Weather_720 | YES (+3.9%) | ~tie (-0.02%) | NO (-0.2%, essentially tied) | same pattern as Weather_96 |

**The decisive, consistent finding across ALL 4 cells, with no
exception: D3 (shuffled set-state control) is always as good as or
slightly better than D1 (the properly-trained model with correctly
-paired set-state).** Scrambling which query's "already selected"
summary the model sees never hurts, and sometimes helps by a hair.
This means the specific mechanism this Phase was built to test --
conditioning each pick on an accurate running summary of the set
built so far -- contributes nothing measurable in any cell. Whatever
D1 gains over D0 in 3/4 cells (ETTh1_720, Weather_96, Weather_720)
comes from the pointwise candidate features and/or the
aggregate-vs-individual training target, not from genuine sequential
set-awareness; and even that gain evaporates to a loss in the fourth
cell (ETTh1_96), where D1 is actually the single worst of the four
arms tested.

D1 vs D2 (aggregate vs individual target, same architecture) is more
mixed: Weather_96 shows a real, if modest, edge for the aggregate
target (+2.8%) consistent with D0's own reasoning for why aggregate
-vs-individual should matter; the other 3 cells show no meaningful
difference (|delta|<1%) or a loss.

## Verdict

CASE 4 from the final cross-phase interpretation framework ("Set
-aware > all pointwise scorers" -- main bottleneck is candidate
interaction/aggregate utility) is **NOT supported**: D1 never
robustly beats D3, the control specifically designed to isolate
whether set-STATE tracking (as opposed to pointwise features or the
aggregate target) is doing anything. Per the pre-registered rule, no
hyperparameter (hidden width, tau_u, tau_s_set, epochs, M=20) was
changed after seeing this result, and the shuffled-control's
occasional edge over D1 was not chased with further architecture
changes. Per section 37, Stage2 unification proceeds next regardless,
using D1's own trained reranker as "Phase D's arm" (the
pre-registered choice, not D3, since D3 is a diagnostic control, not
a deployable arm).
