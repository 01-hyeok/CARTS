# results/

Raw artifacts for experiments run under the `research/` workflow.

Layout:

```
results/
  <EXP-ID>/            e.g. EXP-0001-recall-teacher-sweep/
    command.txt        exact command line(s) used
    env.txt            python/torch versions, GPU, git commit, `git diff` hash
    config.json        resolved args (or the printed arg dump)
    metrics.csv        copied/derived test metrics for this experiment
    <arm>.log          copied stdout for each arm
    notes.md           optional, factual only
```

Rules:

- Never overwrite an existing `<EXP-ID>/` directory. Baselines are never deleted.
- Copy (do not move) the relevant rows/files out of `metrics/` and `logs/` so the
  experiment stays readable after those directories are pruned.
- Record the git commit and whether the working tree was dirty. If dirty, save
  the diff (`git diff > results/<EXP-ID>/working_tree.diff`).

Note: `results/` and `metrics/` are listed in `.gitignore`, so these artifacts are
**not** version-controlled by default. See the setup notes in the session that
created this file.
