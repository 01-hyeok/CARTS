# TRACK-C-HORIZON-RETRIEVAL-CLEAN02 — §3 Code Audit

Audit performed by direct code inspection (not by trusting prior reports).
All existing `TRACK-A-HORIZON-RETRIEVAL-EXPERT01` code/checkpoints/results
were read-only during this audit; nothing under those paths was modified.

## 1. Did the original C actually use `relation_encoder_type='mlp'`?

**[ISSUE] CONFIRMED.** `scripts/train_horizon_retrieval_expert01.py:330-333`:

```python
exp, args = build_experiment(cli.reference_ckpt, {
    'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size,
    'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
})
```

No `relation_encoder_type` override anywhere in this file. It inherits
whatever the `S0_wce` reference checkpoint's own `args` says, which (per
this session's earlier, independently-verified finding on the same
checkpoints — a patch_len=16 vs patch_len=24 smoke test producing
byte-identical outputs) is `relation_encoder_type='mlp'`. Every other
scratch single-encoder script this session (`train_patch_retrieval_expert01.py`,
`train_experiment_e2_multisubspace01.py`, etc.) explicitly overrides this to
`'transformer'` with `relation_self_fill='zero'` for exactly this reason —
`train_horizon_retrieval_expert01.py` does not. Original C's Shared encoder
was a flat MLP over the raw 720-length input, not a Transformer.

## 2. Did the original block head apply only to the query?

**[ISSUE] CONFIRMED.** `scripts/train_horizon_retrieval_expert01.py:69-85`,
`BlockCorrectionHeads`:

```python
class BlockCorrectionHeads(nn.Module):
    """Three zero-initialized, bias-free linear corrections on the QUERY
    embedding -- ... Past-only (function of `z_q` alone) ... `E` (the
    candidate embedding bank) is never re-encoded per block, only the query
    side gets an extra small linear map before the same cosine-against-E
    lookup."""
    def forward(self, z_q, block_idx):
        return z_q + self.heads[block_idx](z_q)
```

The docstring says so explicitly and the code matches: `forward` takes only
`z_q`. The candidate side is scored with the SAME shared `E` for every
block, always.

## 3. Was the original Stage-1 aggregate uniform mean?

**Confirmed correct (not an issue).** `scripts/diag_horizon_retrieval_headroom01.py:51-55`:

```python
def _gather_mean(memory_c, offset_c, picks, h_lo, h_hi):
    ...
    return gathered.mean(dim=1) + offset_c.view(-1, 1)
```

Plain arithmetic `.mean(dim=1)` over the Top-K picks, reused for both the
Global and Block arms' Stage-1 checkpoint-selection metric. Stage-1 itself
never used host-weighting.

## 4. Did the original Stage-2 cache use `HostScorer` weight?

**[ISSUE] CONFIRMED.** `scripts/build_horizon_retrieval_expert01_retrieval_cache.py:130,152,160`:

```python
host = HostScorer(stage2_host, device)
...
host_scores = host.scores(batch_x, c, cand_mask)
...
alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)
```

The cached `relation_outputs` (the retrieval branch's contribution fed into
Stage-2 fusion) is a host-alpha-weighted combination of the Top-10 futures,
for BOTH the Global and Block arms — never a uniform mean. So Stage-1
(uniform mean, item 3) and the actual Stage-2 input signal (host-weighted,
this item) used two DIFFERENT aggregation rules for the "same" retrieval
branch — a real Stage-1/Stage-2 aggregation mismatch.

## 5. Were Global/Block Stage-2 base branches independently trained?

**[ISSUE] CONFIRMED.** `scripts/run_horizon_retrieval_expert01_stage2.sh`:
the FIRST arm processed (`global`) writes `--shared_init_out $init` (a
shared Stage-2 initial state), the second arm (`block`) reads it back via
`--shared_init_in $init` — so both arms start from an IDENTICAL initial
Stage-2 state (fair at t=0). But `train_setlossctrl_stage2_retrain02.py`
then trains each arm's full Stage-2 model (including its base/forecast
branch, gate, everything not explicitly frozen) completely independently
per arm-specific cache — nothing keeps the base branch's WEIGHTS identical
across the two arms after training starts. Directly reproduced this exact
failure mode this session (independently, in the host-free Set-Oracle
Stage-2 reuse of the same trainer): ETTh1_720's `individual` arm converged
to `base_mse=0.482999` while the `set` arm (same cell, same shared init,
same trainer) converged to `base_mse=0.724047` — a >49% relative
difference in what should be a "common" base-only counterfactual. Any
Global-vs-Block final-MSE difference in the original C's Stage-2 is
therefore confounded with base-branch drift, not attributable to the
retrieval branch alone.

## 6. `memory_value()` delta-last semantics

`scripts/train_margutil01.py:62-72`: `memory_value_c = memory_y[..., channel]`,
and IF `args.relation_value_space == 'delta_last'`, subtracts
`memory_x_last[channel]` (the query's own last-observed value) — i.e.
candidate futures are stored in delta-space relative to the QUERY's own
last observation, restored to absolute by adding `query_offset` back
exactly once downstream. Confirmed consistent with every other Track-A
script's documented convention this session; no discrepancy found.

## 7. `candidate_mask='raft'` train/val/test behavior

`utils/relation_memory.py:44-46,68`: `mask_mode` defaults to `'raft'`.
Consistent with this session's own earlier, independently-verified finding
(Patch-MoE feasibility track, §10.1 of that report): under `'raft'`, TRAIN
queries get self/overlap exclusion (candidates near the query's own start
excluded) while VAL/TEST queries get no such exclusion (full candidate
pool). This is an existing, intentional anti-leakage design, not a bug —
already investigated and the "candidate-support mismatch" hypothesis it
raises for router-style experiments was already tested and REJECTED
(deployment-matched pseudo-query experiment) in that unrelated track. Not
itself a defect for Clean02, but worth remembering if any train-vs-val/test
oracle-headroom asymmetry appears again here.

## 8. Gradient flow to both query and candidate encoders

Original C (`train_horizon_retrieval_expert01.py`): `E = encode_raw(model,
exp.memory_x, c)` is recomputed live every batch (not cached/detached),
exactly like every sibling Track-A script — so candidate-side gradients DO
flow in the original. This was already correct; Clean02 keeps the same
convention (spec section 5 explicitly requires it).

## 9. ETTh1_720 candidate bank size / channel count

From this session's own repeated direct measurements on the same S0_wce
ETTh1_720 checkpoint (E1/E1.5/D-track logs): candidate bank = 7,201
windows, 7 channels. (Weather_720: 35,448 windows, 21 channels.)

## 10. Read-only guarantee for Clean02

This audit only READ `results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01/`,
`results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01-STAGE2/`,
`checkpoints/track_a_horizon_retrieval_expert01/`, and
`research/C-horizon-block-retrieval/` (existing files listed, none opened
for writing). All Clean02 code/checkpoints/results/logs will be written
under new `*_clean02` / `TRACK-C-HORIZON-RETRIEVAL-CLEAN02` paths per
section 19 of the spec, verified by construction (new script/directory
names, never the original ones) rather than by a runtime check.

## Summary

| # | Question | Result |
|---|---|---|
| 1 | mlp trunk? | **[ISSUE] confirmed** |
| 2 | block head query-only? | **[ISSUE] confirmed** |
| 3 | Stage-1 uniform mean? | confirmed correct |
| 4 | Stage-2 cache host-weighted? | **[ISSUE] confirmed** |
| 5 | Global/Block base independently trained? | **[ISSUE] confirmed** |
| 6 | delta-last semantics | confirmed correct |
| 7 | raft mask train/val/test asymmetry | confirmed (known, intentional) |
| 8 | gradient flow both sides | confirmed correct |
| 9 | ETTh1_720 bank size | 7,201 windows × 7 channels |
| 10 | no artifact overwrite | guaranteed by new path naming |

All five problems the user's spec (section 1) attributed to the original C
experiment are independently confirmed by direct code inspection, not
merely repeated from the spec text. Clean02 proceeds per the spec's
architecture (symmetric adapters, uniform-mean-only aggregation throughout,
no HostScorer anywhere, common frozen base for Stage-2).
