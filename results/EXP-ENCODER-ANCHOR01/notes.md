# EXP-ENCODER-ANCHOR01 — notes

## lambda_anchor selection

**Method** (`lambda_selection_probe.py`, copied here verbatim): a single
fixed-batch, single-channel, single-K-step (t=1) gradient-norm-ratio
diagnostic — NOT full GPU training, NOT a lambda sweep, and never looks at
test Stage-2 (per the spec's explicit prohibition).

- Trainable encoder state = `EXP-ENCODER-UNFREEZE01`'s own **epoch-1
  checkpoint** (`checkpoints/exp_encoder_unfreeze01/.../checkpoint_epoch1.pth`),
  not the pristine B0 init — chosen because at pristine B0 init the anchor
  loss/gradient is ~0 by construction (S1/S2), which would not tell us
  anything about how the two terms compare once the encoder has actually
  started moving (the regime this experiment cares about).
- Reference encoder = the **original, pristine B0** checkpoint (the same
  one E2's `encoder_ref` will use).
- One real ETTh1 train batch (batch_size=32, channel 0, t=1), `SetConditioner`/
  `EmptySetToken`/`UtilityHead` loaded from the same E1 epoch-1 checkpoint.
- Measured, with `grad_scale=1.0` for both terms (matching each term's
  real per-batch normalisation convention: R2's `1/k` division means its
  real per-batch magnitude is close to one step's magnitude; anchor is
  called once per batch-channel with no `1/k` division, so `grad_scale=1.0`
  for a single term matches its real per-batch contribution exactly):

| quantity | value |
|---|---|
| R2-only encoder gradient norm (1 step) | 0.045958 |
| Anchor-only encoder gradient norm (`lambda_anchor=1`) | 0.282380 |
| anchor_loss at this state (query=0.628, candidates=0.659) | 1.287234 |
| ratio (R2 / anchor@lambda=1) | 0.1628 |

**Chosen value: `lambda_anchor = 0.16`** — sets the anchor term's real
per-batch gradient contribution to roughly the same order of magnitude as
R2's own per-batch gradient contribution (neither dominating the encoder
update, which would risk Outcome D / over-constraining, nor negligible,
which would risk Outcome C / no effect on collapse), evaluated on an
already-partially-collapsed encoder state (the regime this experiment
actually trains through), not the trivial pristine-B0-init state. This
single value is used for the entire E2 run — no sweep.

## Implementation summary

New file `scripts/train_encoder_anchor01.py` (does not modify
`scripts/train_encoder_unfreeze01.py`). Adds:
- `encode_with(encoder_module, model, x, c, grad)`: generalises E1's
  `encode_raw` to an explicit encoder module, so the same helper serves
  both the trainable `model.encoder` and the frozen `encoder_ref`.
- `encoder_ref = copy.deepcopy(model.encoder)` taken immediately after
  loading B0 weights, BEFORE `requires_grad=True` is set on
  `model.encoder` — permanently frozen/eval/no_grad, never touched by the
  optimizer.
- `anchor_step(...)`: the B0-anchor loss (`1 - cos(z, z0)`), evaluated once
  per (batch, channel) covering the query batch and the FULL candidate bank
  (chunked for memory only, sum-then-divide-by-global-N normalisation so
  chunking is exactly loss/gradient-equivalent to an unchunked pass —
  verified by `tests/test_exp_encoder_anchor01.py::test_S7_...`), with
  immediate backward per term/chunk, never calling `optimizer.step()`
  itself (verified by S8).
- `run_epoch`'s train branch runs the existing (imported, unmodified) R2
  K-step loop first, snapshots each encoder parameter's gradient (the "R2
  contribution"), then runs `anchor_step` once per valid channel (adding to
  the SAME `.grad` tensors), then decomposes the FINAL gradient into
  `encoder_grad_norm_r2` / `encoder_grad_norm_anchor` / `encoder_grad_norm_total`
  by exact tensor subtraction (gradients add linearly, so this
  decomposition is exact, not approximate) before the single
  `optimizer.step()` call.

## Sanity checks (S1-S8)

All 8 mandatory checks PASSED before the GPU run —
`tests/test_exp_encoder_anchor01.py`, 8/8:
- S1 zero-step equivalence (`max|z-z0|` ~0 at construction)
- S2 anchor loss ~0 at initialization
- S3 anchor gradient nonzero after a controlled perturbation
- S4 candidate-side gradient flows through the anchor term
- S5 query-side gradient flows through the anchor term
- S6 no stale candidate bank (embeddings track the current encoder; the
  reference encoder stays frozen)
- S7 chunk equivalence for the anchor loss (exact, atol tightened to 1e-5
  after fixing a normalisation bug — see below)
- S8 `anchor_step` never calls `optimizer.step()` itself

**Bug caught by S7 during development**: the first `anchor_step`
implementation averaged each candidate chunk's own mean (not a
sum-then-divide-by-global-N), which silently over/under-weighted a
smaller final chunk whenever the candidate count was not an exact multiple
of `cand_chunk_size` — caught by the chunk-equivalence test failing at a
0.0013 discrepancy (just above the original 1e-3 tolerance). Fixed by
switching to the same sum/global-N normalisation pattern
`encoder_unfreeze_step`'s SmoothL1 term already used, re-verified exact
(atol 1e-5) before the GPU run.

## Training results

`best_epoch=1` (`val_overlap@10=0.0094`, best of C0/E1/E2 on this proxy),
early-stopped at epoch 6 (patience 5), `wall_clock_seconds=8609.7`,
`peak_gpu_memory_mib=516` (no OOM, comparable to E1's own 515MiB). Encoder
gradient decomposition (exact, via tensor subtraction, per epoch, LAST
batch of the epoch — matches this project's existing single-batch grad-norm
logging convention): R2's contribution (`enc_grad_r2`) ranged 0.14-0.31
across epochs, the anchor's own contribution (`enc_grad_anchor`) stayed a
comparatively small and stable 0.026-0.037 throughout — i.e. even though
`lambda_anchor` was calibrated (§ above) to make the anchor's raw gradient
magnitude comparable to R2's on the E1-epoch-1 probe state, in the ACTUAL
E2 training run R2's own contribution ended up 4-10x larger than the
anchor's at most epochs. This is plausibly why the encoder still moved a
real (if small) amount rather than being fully pinned to B0 (§2 in
REPORT.md), while still staying safely inside the non-collapsing regime.

`train_anchor_loss`/`val_anchor_loss` stayed small and stable throughout
(train: 0.019-0.048, val: 0.006-0.008) — no sign of the anchor term
fighting the R2 loss to a stalemate or dominating it.

Full per-epoch numbers: `results/EXP-ENCODER-ANCHOR01/representation_probe.csv`
and `logs/train_full.log`.

## Result summary

See `REPORT.md` for the full 8-question breakdown. Headline: collapse
prevented (`embedding_effective_rank` stayed 15.7-16.6 vs B0's 17.48, vs
E1's 2.96-1.49), t1/t2 top-tail metrics improved to the best of all three
arms, but Stage-2 MSE (0.40418) is still worse than C0 (0.39526, though
slightly better than E1's 0.40668) and HardAggregateMSE is the WORST of
all three arms (0.6768). Verdict: **Outcome B** (collapse prevented,
downstream Stage-2 does not improve), with a Stage-2/HardAggregate vs.
t1/t2 disagreement reported as-is, not resolved. Per the user's explicit
instruction, no further experiment in this direction was auto-started.
