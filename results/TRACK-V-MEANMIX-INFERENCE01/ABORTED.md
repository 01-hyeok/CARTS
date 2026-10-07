# ABORTED

Inference-only RR vs Mean-Mixture experiment was intentionally
terminated because Stage1 checkpoints themselves had been selected
with Round-Robin validation RetMSE. Superseded by
TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01.

- Terminated: 2026-10-07 (UTC), by explicit user instruction.
- Terminated processes: PID 1070203 (`bash scripts/run_v_meanmix_all01.sh`),
  PID 1145545 (`python -u scripts/build_v_meanmix_cache01.py --arm V1
  ... --out_dir results/TRACK-V-MEANMIX-INFERENCE01/Weather/H336/seed0/V1`),
  both via SIGTERM, clean exit, no SIGKILL needed.
- Scope completed before termination: ETTh1 H96/H192/H336/H720 (all 4
  arms), Weather H96/H192/H336 (all 4 arms), Weather H336/V1 in-flight
  when killed. Weather H720 never started.
- Partial results below are preserved as-is (not deleted) for reference
  but are NOT the final comparison -- see
  `results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/` for the superseding
  experiment.
