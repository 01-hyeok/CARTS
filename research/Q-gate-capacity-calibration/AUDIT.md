# TRACK-Q-GATE-CAPACITY-CALIBRATION01 -- Audit

Everything fixed per PART 2: retriever=M2, aggregation=Uniform,
fusion=mixture, common frozen base=TRACK-M's S0. Single question: how
much gate complexity does exploiting M2's retrieval actually need?

## Loss/config audit (S2_720 host, read directly from checkpoint args)

`use_aux_base_loss=0`, `use_aux_ret_loss=0`, `beta_entropy_reg=0.0`,
`retrieval_kl_weight=0.0`, `stage2_rank_weight=0.0` -- every auxiliary
loss term this host config could add is off. `exp._loss` therefore
reduces to plain `mean((y_final-batch_y)**2)` (`exp/exp_stage2_relation.py`
line 969, confirmed no other term is added given these flags). This
justifies computing the loss directly in plain PyTorch for Q2/Q3/Q5
(bypassing `Exp_Stage2_Relation`/`Model.forward` entirely) without
losing any fidelity to the real training objective. `gate_hidden=128`,
`gate_mode='scalar'`, `pred_len=720`, `learning_rate=1e-3`,
`batch_size=32`, `train_epochs=10`, `patience=5` -- identical to every
Q2/Q3/Q5 run in this track (PART 7).

## `relation_mixer` (PART 4)

Per TRACK-O's finding (re-confirmed in TRACK-P): `num_source_slots()==1`
makes `relation_mixer`'s `softmax(..., dim=1)` the constant `beta≡1`
with zero gradient, for any input -- a structural, not incidental, fact
of this session's single-aggregated-source cache convention. This track
does not construct `RelationStage2.Model` or `relation_mixer` at all for
Q2/Q3/Q5: the cached Uniform-aggregate `R` and the common frozen `B` are
consumed directly by each arm's own lightweight gate module, exactly as
PART 4 prescribes ("가능하면 cache aggregate R을 gate에 직접 전달한다").
This is mathematically identical to passing `R` through the frozen
identity-pass `relation_mixer` (already proven exact in TRACK-P) and
strictly simpler/faster.

## Frozen artifacts (identical hashes to every prior track)

| Role | Checkpoint | SHA-256 |
|---|---|---|
| M2 retriever | `checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth` | `9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2` |
| S0 common frozen base | `checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth` | `a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609` |

## Reused verbatim, not rebuilt

- Common base predictions (`B`): `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions/base_predictions_{train,val,test}.pt`.
- M2 Uniform retrieval cache (`R`): `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G2_M2/{train,val,test}.pt`.
- Q0 = TRACK-N/O/P's own base-only evaluation (0.488790).
- Q1 = TRACK-P's P6 (fixed lambda=0.42, closed-form, 0.463969).
- Q4 = TRACK-P's P4 (query-conditioned MLP gate, mixture, 0.475686) -- checkpoint reused directly:
  `checkpoints/track_p_fusion_semantics_audit01/ETTh1_720/P4_M2_mixture/checkpoint.pth`.

## New for this track

`scripts/train_q_gate_capacity01.py` -- one script, `--gate_type
{global,per_channel,query_prior}`, training Q2/Q3/Q5 directly on cached
`(B, R, Y)` tensors with plain-PyTorch minibatch SGD (Adam), no
`Exp_Stage2_Relation`/`Model` construction needed at all.
