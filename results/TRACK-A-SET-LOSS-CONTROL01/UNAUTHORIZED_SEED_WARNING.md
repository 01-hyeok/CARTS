```text
NOT DIRECTLY COMPARABLE TO FACTORIAL SEED-0 BASELINE
UNAUTHORIZED SEED CHANGE
PRESERVED AS A SEED-1 REPLICATE ONLY
```

Every result under `results/TRACK-A-SET-LOSS-CONTROL01/` (both `ETTh1_96/`
and `ETTh1_720/`) was produced with `--seed 1`, chosen without user
approval. The Factorial baseline this experiment's `A0_hard_choice` arm is
supposed to reproduce (`set_onpolicy_cosine`, Set Oracle + On-policy +
Cosine + Hard Choice CE, `checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth`)
was trained with `--seed 0`. These are therefore two different random
initializations/data orders, not a controlled reproduction, and any
comparison drawn between `A0_hard_choice` and the Factorial `set_onpolicy_cosine`
baseline in this directory's outputs is NOT a valid same-seed comparison.

Nothing under this directory has been deleted or overwritten. The
corrected, same-seed (seed=0) re-run lives at
`results/TRACK-A-SET-LOSS-CONTROL02/` -- a separate path, separate
checkpoints, never touching this one.
