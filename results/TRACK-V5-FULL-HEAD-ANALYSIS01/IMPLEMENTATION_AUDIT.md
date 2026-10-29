# Full Candidate Head Analysis — Implementation Audit

## Authorized scope
Frozen Full V5 checkpoints, seed0 only. Each cell finishes Stage2 before the next:
ETTh1_96 → ETTh1_720 → Weather_96 → Weather_720. A separate P100 track shares physical GPU1.
No encoder/head/loss training. Only existing global lambda Stage2 fitting is run.

## Existing code reused
- `scripts/train_margutil01.py`: reference args, experiment/data construction, delta-last value reconstruction.
- `utils/full_candidate_bank.py`: exact self-channel encoder preprocessing; candidate embedding cached in eval.
- `scripts/train_k_multislot_predictive_retrieval01.py`: five normalized linear SlotHeads.
- `utils/mean_mixture_selection.py`: mean of per-head softmax, not softmax of mean scores; tau from selected checkpoint metadata.
- `models/RelationStage1.py`: stable Top-K candidate ID tie handling.
- `scripts/train_factorial_e2e01.py`: exact candidate future error metric, evaluation only.
- `scripts/train_patch_retrieval_expert01.py`: existing Recall/NDCG definitions.
- `scripts/train_r_stage2_lambda01.py`: UNMODIFIED Stage2 command; only global sigmoid lambda, Adam LR .001, batch32, max10 epochs, validation MSE checkpoint and patience5. Train-only fitting, frozen Base/retriever, test reported once.

## Canonical provenance
`results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{dataset}/H{horizon}/seed0/`
contains selected checkpoint path/epoch/tau, full cache, corrected Stage2 metrics and stored gate.
`pool_top100` is never used by this track. Per-cell CHECKPOINT_PROVENANCE.json records actual full candidate count, source git SHA, script hash, checkpoint hash, Base hash and selected epoch.

V2 independent model uses the corrected nested V2/cache+stage2. V0/V1 use prior MeanMix inference cache/metrics (single head/no head, so mixture and historical selection coincide). Every historical comparator is forward rechecked with this exact Base and split; missing evidence is marked MISSING, never fabricated. These reused metrics are labeled separately from newly fitted V5 ablations. V0→V1 includes projection, and a V5 subset is not an independently trained small-head model.

## Candidate mask and leakage scope
The existing RAFT mask is preserved, NOT silently converted to strict_causal. For a train query whose start is present in memory, all starts within L+H-1 on either side are excluded. For query starts outside train memory, all train candidates are valid. Train memory IDs are independently checked against the train dataset global starts. The arithmetic mask is independently compared to the sampler for real batches. Query future is used only for errors/oracle evaluation; score and selection APIs receive only history-derived probabilities, mask and head subset. Oracle subsets are explicitly non-deployable. Fixed subset is selected on validation only.

## Tests and reproduction gates
Nine CPU synthetic tests: mean-mixture equivalence; single/duplicate head; permutation; independent direct-dot D/C; remaining-head renormalization; raw-dot cosine and zero handling; selection mask/future-free API; independent RAFT mask; lambda0/1 identities.
Real one-batch verifies cached-bank score and exact selected indices against canonical full scoring, independent RAFT validity and train-only memory.
Per-cell gate recomputes full test Stage1 metrics and forwards existing cache through saved original lambda. After fresh caches, every split's query IDs and All5 predictions are compared with canonical cache and saved lambda is evaluated again with new test cache before ablation Stage2 begins.

## Numerical correction during smoke
A preliminary batch16/chunk1024 cached encoder pass had first-batch exact indices but full-test AggMSE differed by 1.28e-5, plausibly due to near-tie float32 GEMM differences. This preliminary gate is SUPERSEDED, not claimed exact equivalent. Production defaults match canonical batch32 and unchunked per-channel candidate encoding (cached once). Strict Stage1 tolerance 2e-6 plus per-prediction cache equivalence and Stage2 tolerance1e-5 guard acceptance. A reduced-memory OOM retry must pass the same gates, otherwise FAIL.

## Analysis and reporting
31 nonempty subsets on validation/test: RetMSE, AggMSE, AggMAE, Recall, NDCG, direct D/C. All head probabilities are reaveraged and Top10 reselected. Eleven singles/All5/leave-one-out configurations produce train/validation/test caches and separate lambda fitting; secondary uses canonical fixed lambda. Candidate union/oracle coverage/unique contributions and pairwise Top1/Top10 overlap are distinct from head prediction error cosine and candidate-set cross terms. Exact zero/near-zero pair fractions are logged, not interpreted as complementarity.

## Resource and output safety
GPU UUID `GPU-91418a73-202a-2a9b-57d5-2538b19e07ec` is verified to physical index1 with nvidia-smi, then CUDA_VISIBLE_DEVICES fixed to UUID. No other jobs stopped. Controller flock prevents duplicate launch. Each cell runs in an isolated process; OOM retries lower evaluation batch/chunk only, full support unchanged, failed output archived within this new experiment tree. Old repository artifacts are read-only. Timing is explicitly concurrent shared-GPU timing, not isolated efficiency evidence. Cache arrays are disk-backed then serialized one at a time. Root and per-cell CSVs are generated from actual records, Korean FINAL_REPORT marks missing cells PARTIAL. No automatic git commit/push or large-cache upload.

## Limitations
Single seed gives conditional evidence, not general specialization or statistical significance. No new router, loss, architecture or seed1/2. Sparse/unavailable independent model artifacts remain MISSING. Wall time under concurrency is not a speed benchmark. Successful test numerical identities are not a proof of semantic head roles.

## Reproduction runtime correction (2026-10-28)
The historical runtime is /usr/bin/python3, PyTorch 1.12.0.dev20220327+cu113, CUDA 11.3, cuDNN 8302. The newer ts-env 2.4.1/CUDA12.1 altered three near-boundary query-channel selections even after TF32 restoration. Existing legacy runtime reproduces all 19,495 ETTh1 H96 test query-channels exactly (max R difference 0), and saved-lambda MSE/MAE exactly. Evidence: legacy_parity_metrics_probe.json and legacy_parity.log. Production must use /usr/bin/python3. Retrieval TF32=True; Stage2 forward precision explicitly recorded. Error diagnostic dots use float64 independently of retrieval precision. No tolerance relaxation was used.
