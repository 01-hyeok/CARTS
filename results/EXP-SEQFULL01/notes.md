# EXP-SEQFULL01 -- factual notes

## Design choices recorded per the experiment spec's request

- **Empty-set token (t=1 state)**: a single LEARNED vector (`EmptySetToken`),
  not zero. Zero is a fixed, out-of-distribution point the conditioner never
  revisits after step 1; a learned vector can be placed wherever training
  finds useful in the same space as later `m_{t-1}` values, at the cost of one
  extra length-128 parameter. See `models/SequentialSetRetriever.py`.
- **Similarity function**: cosine (`h_t` and every candidate embedding are
  L2-normalised, scored by dot product), matching every other arm in this
  project's default convention. No new metric was introduced.
- **SetConditioner**: `concat(q, m) -> Linear(2D,D) -> GELU -> Linear(D,D)`,
  added to `q` (residual) and LayerNorm'd, then re-normalised to unit length.
  No attention, no cross-attention, no pairwise MLP, per the spec's explicit
  prohibition for this pilot.
- **Small-N gate's own teacher**: scored by a FROZEN snapshot of the same
  (randomly initialised) encoder architecture, taken before training starts.
  There is no pretrained checkpoint restricted to the synthetic 256-candidate
  universe to score with instead; a frozen-at-start snapshot is offline/
  non-circular even though it shares its initial weights with the trainable
  encoder (see `scripts/train_seqfull01.py::small_n_teacher`).

## Section 8 (Stage-2 connection): implemented exactly as specified

The sequential selector's final Top-10 membership is injected via
`RelationStage2.set_forced_selection` -- the existing, already-reviewed
oracle-intervention mechanism (EXP-1/EXP-2), reused verbatim. Stage-2's own
base forecaster/gate/fusion/tau_topk/aggregation are B0's, completely
unmodified. The final Stage-2 aggregation weight for the sequential arm's
selected 10 candidates is B0's own frozen retrieval score (cosine), not
anything the sequential model produces -- so is the Weighted Set Oracle
teacher's weighting, for the same reason. No part of section 8 required a
smaller alternative; the existing mechanism covered it exactly.

## Fingerprint check (section 12)

`scripts/eval_seqfull01_stage2.py` runs B0's own (unforced) selection through
the identical evaluation loop before reporting the forced-selection number:
`b0_unforced_final_mse = 0.373122`, matching the recorded B0 baseline
(0.37312) to 5 significant figures. The forced-selection number below is
measured under a verified-identical protocol, not merely an assumed one.

## Sanity

`pytest tests/`: 458 passed both before and after this experiment's code
(14 new tests in `tests/test_exp_seqfull01.py`), exactly the 2 pre-existing
failures (`test_topk_coverage_reuses_target_indices_across_relations`,
`test_identity_retrieval_uses_raw_target_source_relation_without_encoder`),
no regression.

## A dataloader bug caught before it corrupted a result

The first full-run attempt crashed on batch 1 of epoch 1
(`AttributeError: 'numpy.ndarray' object has no attribute 'float'`) because
`exp._get_data(flag=...)` returns `(dataset, loader)` and the training script
had unpacked it as `loader, _ = exp._get_data(...)` -- silently binding the
Dataset object to the `loader` name instead of the DataLoader. Caught
immediately by the crash (not a silent wrong-shape bug); fixed and the run
was restarted from scratch. The small-N gate was unaffected (it uses
`Exp_Stage1_Relation._configure_tiny_overfit`'s own returned loaders, a
different code path that was correct).

## GPU sharing note

GPU 1 was, for part of this session, concurrently running an unrelated
external job (`train_pretrain.py`, ETTm1 mixer pretraining, a different
project/user) also using CUDA_VISIBLE_DEVICES corresponding to GPU 1. Per the
user's explicit instruction, this project's own runs were still executed
sequentially among themselves on GPU 1 (never in parallel with each other),
sharing the physical GPU with that unrelated job rather than switching GPUs
or waiting for it to finish.

## Effective rank / collapse diagnostic

Computed post-hoc on the trained encoder's full candidate bank (channel 0):
effective rank 5.43 (out of 128 embedding dimensions), mean pairwise cosine
0.0122. Not a total collapse (a collapsed encoder would show effective
rank near 1 and mean pairwise cosine near 1), but substantially more
compressed than B0's own encoder (recorded effective rank ~20.8 for the same
architecture trained with WCE in EXP-3's closure). Consistent with, though
not proof of, a representation that discriminates far less sharply among the
8449 candidates than B0's does -- plausibly part of why generalisation from
the training queries to held-out ones failed so completely.
