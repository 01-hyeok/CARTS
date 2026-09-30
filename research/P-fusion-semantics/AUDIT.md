# TRACK-P-FUSION-SEMANTICS-AUDIT01 -- Fusion-Semantics Audit

NO RETRIEVER TRAINING. This track tests exactly one thing: is Stage2's
`residual` fusion (`y_final = y_base + lambda*y_ret`) mis-specified,
given that `y_ret` is a full alternative forecast (not a correction
term), versus `mixture` fusion (`y_final = (1-lambda)*y_base +
lambda*y_ret`, algebraically `y_base + lambda*(y_ret-y_base)`)?

## Files read (carried over from TRACK-O's audit, re-confirmed here)

- `layers/retrieval_gate.py` (`RetrievalGate.forward`): both fusion
  modes are ALREADY implemented --
  `if self.fusion_mode == 'residual': y_final = y_base + lam*y_ret`
  vs `else: y_final = (1-lam)*y_base + lam*y_ret`. `fusion_mode` is a
  constructor argument, read once at `RetrievalGate.__init__` time from
  `configs.fusion_mode`; nothing else in `RelationStage2.Model` branches
  on it except the earlier `raft_concat` vs `gate` module-selection
  check in `__init__` (irrelevant here -- `mixture` is not
  `raft_concat`, so the SAME `gate = RetrievalGate(...)` construction
  path is taken for both `residual` and `mixture`; only the string
  passed to `RetrievalGate.__init__` differs). This means switching
  `args.fusion_mode` from `'residual'` to `'mixture'` before model
  construction changes ONLY the gate's internal fusion formula --
  identical architecture, identical parameter count, identical
  everything else.
- `models/RelationStage2.py` (`Model.forward`, `num_source_slots`):
  reconfirms TRACK-O's finding -- `num_source_slots()==1` for the
  S2_720 host, so `relation_mixer`'s `softmax(scores, dim=1)` over a
  size-1 axis is the constant `beta≡1`. Per PART 7, this is NOT treated
  as a trainable consumer here (frozen, as in TRACK-O) -- and its
  `beta≡1` identity-pass property is directly asserted (`y_ret_c ==
  relation_outputs[:, 0, :]` exactly) rather than assumed, satisfying
  PART 7's explicit fallback instruction ("beta=1 identity pass임을
  assert").

## Checkpoints (frozen, never retrained; identical to TRACK-M/N/O)

| Role | Checkpoint | SHA-256 |
|---|---|---|
| J1 retriever | `checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth` | `1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c` |
| M2 retriever | `checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth` | `9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2` |
| S0 common frozen base | `checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth` | `a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609` |

## Reused verbatim, not rebuilt

- Uniform retrieval caches (J1, M2): `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/{G0_J1,G2_M2}/{train,val,test}.pt`.
- Common frozen base predictions: `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions/base_predictions_{train,val,test}.pt` (P0).
- **P1 = TRACK-O's O1_J1_uniform** (J1+Uniform+Residual+learned gate) --
  identical definition, checkpoint reused directly:
  `checkpoints/track_o_frozen_base_retrieval_consumer01/ETTh1_720/O1_J1_uniform/checkpoint.pth`.
- **P2 = TRACK-O's O3_M2_uniform** (M2+Uniform+Residual+learned gate) --
  checkpoint: `checkpoints/track_o_frozen_base_retrieval_consumer01/ETTh1_720/O3_M2_uniform/checkpoint.pth`.
- P5/P6 fixed-lambda targets: TRACK-N's `validation_lambda.json`
  (`J1/U/global_lambda=0.37`, `M2/U/global_lambda=0.42`) and
  `test_calibrated_fusion.csv` (`test_mse_val_global_lambda`:
  J1=0.463349, M2=0.463969) -- reused as the exact reproduction targets,
  recomputed here via the identical closed-form (no model forward
  needed: `MSE(B+lam*(R-B), Y) = (|t|^2 - 2*lam*dot(c,t) + lam^2*|c|^2)/H`).

## New for this track

- `scripts/train_p_fusion_semantics01.py`: extends
  `train_o_frozen_base_consumer01.py` with a `--fusion_mode
  {residual,mixture}` flag that overrides `args.fusion_mode` BEFORE
  `Exp_Stage2_Relation(args)` construction (the only change from
  TRACK-O's script). P3 (`fusion_mode=mixture`, retriever=J1) and P4
  (`fusion_mode=mixture`, retriever=M2) are the only genuinely NEW
  training runs this track performs.
