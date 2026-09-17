```text
NOT DIRECTLY COMPARABLE TO FACTORIAL SEED-0 BASELINE
UNAUTHORIZED SEED CHANGE
PRESERVED AS A SEED-1 REPLICATE ONLY
```

Every result under `results/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/` was
built on top of `results/TRACK-A-SET-LOSS-CONTROL01/`'s `--seed 1`
Stage-1 checkpoints, chosen without user approval -- see
`results/TRACK-A-SET-LOSS-CONTROL01/UNAUTHORIZED_SEED_WARNING.md` for the
full explanation. The delta-space cache/Stage-2 correction this directory
represents (see `research/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED.md`) is
still valid AS A SEED-1 REPLICATE -- the value-space bug fix itself is
orthogonal to the seed issue -- but none of these numbers should be
compared directly against the Factorial seed=0 baseline or presented as
"the" TRACK-A loss-control result.

Nothing under this directory has been deleted or overwritten. The
corrected, same-seed (seed=0) re-run lives at
`results/EXP-SET-LOSS-STAGE2-RETRAIN02/` -- a separate path, separate
checkpoints, never touching this one.
