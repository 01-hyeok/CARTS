# TRACK-L-EVAL-ALIGNMENT-FULLTEST01 -- Checkpoint Audit

NO TRAINING. All four checkpoints below are loaded and evaluated exactly
as saved by their originating tracks -- never re-selected, never
re-trained.

| Arm | Checkpoint path | File SHA-256 | Best epoch |
|---|---|---|---|
| J0 | `checkpoints/track_j_shared_encoder_drift01/ETTh1_720/checkpoint.pth` | `6d51c52be5b9...` | 10 |
| J1 | `checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth` | `1b14877fce3c...` | 1 |
| K1 | `checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K1_multislot_relevance/checkpoint.pth` | `ac7120b5d0cc...` | 1 |
| K2 | `checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth` | `9827f76d4698...` | 1 |

(Full hashes in `results/TRACK-L-EVAL-ALIGNMENT-FULLTEST01/ETTh1_720/checkpoint_audit.json`.)

Confirmed identical across all four: `encoder=mlp`, `relation_self_fill=linear`,
`delta_last` input/value/teacher space, `candidate_mask=raft`,
`N=7201` train-only candidates, `top_k=10`, `channels=7`, `seq_len=pred_len=720`,
`init_seed=loader_seed=0`, encoder init hash
`b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e`
(asserted at load time for every arm; would abort the run otherwise --
never fired).

## Root cause of the original population mismatch

`build_model` + `build_probe_set(exp, test_loader, 256, device)` was used
by TRACK-J/J2/J3 to compute `final_test_metrics.json` for J0/J1 -- a
FIXED 256-query subset of the test split. TRACK-K's
`train_k_multislot_predictive_retrieval01.py` computed K1/K2's
`final_test_metrics.json` differently: it iterated the FULL test
`DataLoader` (all 2161 queries, all batches) for the final test-metrics
block. Both were internally correct evaluations of their own
checkpoints -- the two numbers were simply never computed on the same
query population, so any J-vs-K comparison using the previously-saved
numbers directly was comparing 256-query and 2161-query results as if
they were the same measurement. This track eliminates that by
evaluating every arm on BOTH populations with one shared evaluator.

## Note on `probe_query_ids.json`

`results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/probe_query_ids.json`
is TRACK-J's own **validation** probe (saved before training, used for
its step-level diagnostics) -- NOT a saved test-probe file. TRACK-J/J2's
`final_test_metrics.json` numbers came from a test probe built fresh,
inline, via `build_probe_set(exp, test_loader, n_probe=256, device)`,
whose indices were never written to a file. `build_probe_set` is a pure,
deterministic function (evenly-spaced `linspace` over the val/test split,
no randomness) -- already unit-tested for this property in TRACK-J's own
suite (item 12). This track's P256 population is built the identical way
and its correctness is established by the reproduction gate (Section 0
below), not by comparing against the (differently-scoped)
`probe_query_ids.json` file -- an initial attempt to do the latter
correctly failed and revealed this exact distinction.
