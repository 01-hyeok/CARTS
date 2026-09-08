# EXP-ONPOLICY-PREFIX01 — notes

## Training-time state diagnostics (per epoch, val split)

| epoch | val_overlap@10 | first_div_mean | prefix_overlap_t1 | prefix_overlap_t10 | regret_t1 | final_agg |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0089 | 0.006 | 0.0062 | 0.0090 | 0.755 | 0.7037 |
| 2 | 0.0090 | 0.006 | 0.0057 | 0.0089 | 0.773 | 0.7040 |
| 3 | 0.0091 | 0.005 | 0.0049 | 0.0090 | 0.788 | 0.7073 |
| 4 (best) | 0.0091 | 0.004 | 0.0043 | 0.0092 | 0.813 | 0.7000 |
| 5 | 0.0089 | 0.005 | 0.0051 | 0.0088 | 0.804 | 0.6999 |
| 6 | 0.0088 | 0.004 | 0.0043 | 0.0087 | 0.818 | 0.7040 |
| 7 | 0.0088 | 0.004 | 0.0045 | 0.0089 | 0.839 | 0.7040 |
| 8 | 0.0087 | 0.004 | 0.0042 | 0.0088 | 0.830 | 0.7006 |
| 9 | 0.0089 | 0.004 | 0.0040 | 0.0090 | 0.852 | 0.7041 |

`best_epoch=4`, early-stopped at epoch 9 (patience 5), `wall_clock=1932.8s`
(~32 min), `peak_gpu_mem=729MiB`, no OOM.

**Observation, flagged as-is, not over-interpreted at training time**:
`prefix_overlap` with the Oracle's own sequence stayed near-zero throughout
training (0.004–0.009, both at t=1 and t=10) with no improving trend, and
`regret_t1` (the model's own on-policy step-1 regret, measured against the
best achievable continuation from the model's OWN current state) actually
ROSE slightly epoch over epoch (0.755→0.852) rather than falling. Read in
isolation, this looked like it might indicate the on-policy training
target was not being learned well. **The downstream evaluation (REPORT.md)
shows this reading would have been wrong** — despite this training-time
trajectory, T1 produced this session's best Stage-2/HardAggregate result
by a wide margin. The likely reconciliation: `prefix_overlap`/`regret_t1`
measure agreement with the ORACLE's own trajectory, which on-policy
training was never targeting — T1 optimizes SELF-consistency (matching
each step's teacher-forced target to the model's OWN previously-chosen
state), not agreement with the Oracle's path. A model can become highly
self-consistent (and therefore behave coherently at Stage-2 inference,
which only ever sees the model's own free-running trajectory) while
diverging completely from the Oracle's specific choices, which is exactly
the pattern observed. This is a useful methodological note for interpreting
any future on-policy training diagnostics in this codebase: **agreement
with the Oracle's trajectory is not the right proxy for on-policy training
quality; downstream Stage-2 is.**
