# TRACK-R-FINAL-METHOD-GENERALIZATION01 -- Audit

Not architecture exploration: this track fixes the final method
(Relevance-Constrained Multi-Slot M2 + Uniform aggregate + Mixture
fusion + Trainable Global Lambda) and tests generalization across
datasets (ETTh1, Weather) x horizons (96/192/336/720) x seeds (0/1/2).

## Hyperparameters audited against TRACK-M/J2/N/O/P/Q (unchanged, reused verbatim)

| Component | Hyperparameter | Value | Source |
|---|---|---|---|
| M2 Stage1 | `S` (slots) | 10 | `train_m_relevance_constrained_multislot01.py:N_SLOTS` |
| M2 Stage1 | `L_soft` | 32 | `...:L_SOFT` |
| M2 Stage1 | `tau_t`, `tau_s` | 0.1, 0.1 | CLI defaults |
| M2 Stage1 | `lambda_agg`, `beta` | 1.0, 0.05 | `...:LAMBDA_AGG_DEFAULT`, `BETA_DEFAULT` |
| M2 Stage1 | `gamma`, `delta` | 1.0, 0.05 | `...:GAMMA_DEFAULT`, `DELTA_DEFAULT` |
| M2/J1 Stage1 | `top_k` | 10 | CLI default |
| M2/J1 Stage1 | optimizer/lr/batch/epochs/patience | Adam/1e-3/32/10/5 | CLI defaults, identical to J1/K2 |
| Stage2 lambda | init | `a=0` -> `lambda=0.5` | PART 3 |
| Stage2 lambda | optimizer/lr/batch/epochs/patience | Adam/1e-3/32/10/5 | matches every Stage2 training this session |
| Base forecaster | architecture | `BaseForecastHead(mode='per_channel_linear')` | `models/RelationStage2.py`, unchanged |
| Base forecaster | optimizer/lr/batch/epochs/patience | Adam/1e-3/32/10/5 | matches Stage2 convention (documented choice, PART 26 -- no other value specified by spec) |

All values read directly from the actual scripts (`N_SLOTS`, `L_SOFT`,
`BETA_DEFAULT`, `LAMBDA_AGG_DEFAULT`, `GAMMA_DEFAULT`, `DELTA_DEFAULT`
constants in `train_m_relevance_constrained_multislot01.py`), not
assumed -- PART 1's explicit requirement.

## PART 3/4: Stage2 structure (relation_mixer/gate bypass)

Per TRACK-Q's finding (`relation_mixer` structurally gradient-dead,
`num_source_slots()==1`), TRACK-R does not construct
`RelationStage2.Model`/`relation_mixer`/`RetrievalGate` at all for
Stage2. `scripts/train_r_stage2_lambda01.py`'s `GlobalLambdaGate` is a
bare `nn.Parameter(torch.zeros(1))` consuming the cached retrieval
aggregate `R` and the base forecaster's own prediction `B` directly,
exactly matching TRACK-Q's Q2 arm's design and its audited "no
relation_mixer construction" property.

## Documented ambiguity resolution: B1 "Raw Cosine Retrieval"

PART 9 specifies B1 as "학습 없는 기본 retrieval... query/key historical
representation의 기존 cosine retrieval" without pinning down whether the
embedding space is (a) an untrained (randomly-initialized) learned
encoder's output, or (b) the raw input window itself. (a) would be close
to noise (a random projection's cosine similarity carries no real
shape-similarity signal); (b) is the standard, meaningful non-learned
time-series retrieval baseline (nearest-neighbor by raw shape
similarity). This track uses **(b)**: cosine similarity computed
directly on the `delta_last`-space input window
(`x[:,:,c] - x[:,-1,c]`, the same space convention as every learned
retriever this session), no encoder, no training, future-blind by
construction (`scripts/build_r_retrieval_cache01.py:cosine_scores`).
Verified empirically sensible: on ETTh1 H720 seed0, cosine's
`retMSE=1.3596`/`Agg=0.8726` is clearly worse than J1's
(`retMSE≈1.006`) and M2's (`retMSE≈0.989`), the expected ordering for a
non-learned vs. learned-retrieval comparison.

## Base forecaster: freshly trained per setting (PART 11), not reused

Unlike TRACK-N/O/P/Q (which reused ONE frozen ETTh1_720 checkpoint,
TRACK-M's S0, for every experiment), PART 11 explicitly requires a FRESH
base forecaster per (dataset, horizon, seed), shared identically across
B1/B2/B3 within that setting. `scripts/train_r_base_forecaster01.py`
implements this. **Observed finding, flagged transparently**: a
freshly-trained ETTh1 H720 seed0 base forecaster (this track's own
pipeline, verified bit-for-bit deterministic across repeated runs with
the same seed) reaches `test_mse=0.560354`, notably worse than TRACK-M's
historical S0 checkpoint's `test_mse=0.488790`. Root-caused to
construction-order-dependent random initialization: TRACK-M's S0 was
constructed as part of a much larger `RelationStage2.Model`
(base_head+relation_mixer+relation_concat_projection+gate all
initialized together under one `torch.manual_seed(seed)` call), while
this track constructs `BaseForecastHead` alone -- the same nominal seed
produces a *different* actual weight draw because a different amount of
RNG state is consumed beforehand. `best_epoch=1` for this base (further
epochs monotonically overfit) is consistent with the established
session-wide pattern of `per_channel_linear` base heads converging
almost immediately and being sensitive to their specific SGD path in
that first pass. This is accepted as the correct arm to use (PART 11
mandates a fresh base; this pipeline is verified deterministic and
bug-free), but is reported honestly as a real, non-trivial seed
-init-order sensitivity finding rather than silently normalized away --
see the main report's Phase A section.
