# Calibration bridge02: preparation only

Code: `/data/pjh_workspace/CARTS/experiments/calibration_bridge02/`
Evidence: `/data/pjh_workspace/CARTS/results/TRACK-CALIBRATION-BRIDGE02/`
Version01 and all upstream/original sources and results remain unchanged.

`--check` checks sealed dependencies and the upstream readiness/resolver identity contract. It is NOT consumer integration validation.

`--preflight` is read-only, CPU-only, and calls readiness before resolver. It validates 40 exact identities, record hashes, data/pool/encoder provenance, selection, checkpoint contents on CPU, raw sidecars and arrays/coverage, native-to-cross provenance, Full selection and mixed original/recovered checkpoint routing, and actual Full module bindings. Selected Full D2/C2 models are loaded strictly into CPU Arm models. No GPU context constructor, training, lock, directory creation, receipt, or marker is invoked. Exit 2 means upstream incomplete; exit 1 means integrity/schema/runtime failure. Fixture completion does not constitute actual upstream completion or GPU validation.

Upstream readiness owns final PASS 16/8/16, full validation, reuse manifest and reports. Its code and dependencies are pinned. Existing original ETTh1 FULL96 raw symlinks are allowed only when explicitly manifest-declared, hash-verified, logically under the resolved result and resolving inside the approved original-results tree. This adapter creates no links and no recovery01 markers.

## Existing monitor procedure after upstream completion

Use the approved Windows OpenSSH command and the following remote commands; do not create another scheduler or waiter.

1. Verify adapter SOURCE_SEAL.json against the separately reported digest and its dependency checks. Run:

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge02/bridge.py --preflight
```

2. Only exit 0 with `PASS_CPU_INPUT_INTEGRATION_NOT_GPU` makes the stage eligible. Exit 2 means defer to a future monitor check. Exit 1 means stop and report; do not change seals or tolerate mismatches. Confirm no P100_STARTED.json, launch_p100, or prior Calibration P100 GPU artifacts exist. Then issue once:

```
PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge02/bridge.py --execute p100
```

Execution repeats preflight, GPU idle checks before locks, exclusive acquisition of both existing Calibration queue locks, legacy native/supplemental controller locks, recovery02 controller lock, and the original GPU lock. All are opened read-only, never recreated. GPU1 UUID, no compute processes, memory <128 MiB and utilization 0 are checked twice 30 seconds apart after locks, then before every child. A busy resource exits without waiting indefinitely.

3. The atomic `P100_STARTED.json` claim is created inside locks with stage/run ID, UTC, actual command, PID, boot ID, process start ticks, command hash, upstream final/validation/reuse/source seal hashes and adapter seal hash. `launch_p100/STATE.json` records child PID/identity/command/step and reaped exit status. A prior claim or launch directory blocks ALL automatic restart, even if its PID is dead/reused or JSON is incomplete. Never remove or overwrite a claim. Directory/state write failures also require review. Parent death cannot authorize another start; the GPU lock is inherited by children.

4. P100 executes canonical cache, exact GPU ordered-ID replay (zero for D2/C2 over all 16 identities), head diagnostics and P0 gate before P1/P2. After successful child reaping, it writes COMPLETE state and an exclusive P100_COMPLETE.json receipt binding the state and completion/selection/protocol/checkpoint/P0 hashes. Monitor must observe successful completion AND supervisor exit; a live supervisor, failed state, missing receipt, hash mismatch, or unresolved worker prevents Full.

5. Re-run actual preflight. When P100 receipt and all output hashes match the same upstream and adapter bindings, and all P100 P2 cells are complete, issue once:

```
PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge02/bridge.py --execute full
```

Full independently checks prior supervisor/worker identities and completion, resources/locks, and its own exclusive start claim; then performs Full P0, GPU integration gates, P1/P2, and frozen cross-support. It retains original GPU tolerances and exact replay requirements. Each stage has independent immutable start and completion records.

Any execution exit 1/2 or FAILED/incomplete lifecycle record is a stop for that attempt, not a retry instruction. Do not use --worker directly. No automatic process takeover, cancellation, Solar, seed1/2, P3/P4, tolerance relaxation, or source rewriting is provided. There is no background waiting process. Interrupted output requires separately reviewed recovery in another version.

GPU equivalence, memory/cost, and the actual complete-upstream preflight remain unexecuted until real upstream completion. Preparation tests use only CPU/isolated paths and never acquire production locks.
