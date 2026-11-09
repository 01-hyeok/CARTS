# Calibration bridge preparation

No controller, monitor, waiting process, GPU replay, or training was launched during preparation.

Code: `/data/pjh_workspace/CARTS/experiments/calibration_bridge01/`
Preparation evidence: `/data/pjh_workspace/CARTS/results/TRACK-CALIBRATION-BRIDGE01/`

The upstream API runs in a separate CPU process: `readiness.check()` must succeed before `resolver.resolve(mode, cell, training, pool=None)`. The adapter consumes the actual records returned by the owner; it does not implement the owner's final validator. It additionally checks the consumer's identity set and referenced hashes/provenance. A missing final PASS returns 2; integrity errors return 1. No fallback to recovery01 markers exists.

The original Calibration files remain unchanged. For Full workers, an explicitly scoped module-global `OLD` path adapter resolves native and cross-support artifacts and selected checkpoints. Features continue to read the original immutable feature directory. Checkpoints may legitimately resolve to the original Weather96 epoch <=8 even when other results resolve to recovery02. There are no filesystem links or synthetic completion markers.

Before execution, the upstream final contract must be READY, all pinned source/dependency hashes must match, and all original queue/controller/GPU locks must be exclusively available. GPU1 UUID, absence of compute processes, memory <128 MiB and utilization 0 are checked twice 30 seconds apart while locks are held, and again before every child. Busy GPUs cause an exit, never cancellation or a background wait. Lock paths are opened read-only and never replaced. Full requires the bridge P100 receipt bound to the same upstream final hash, all eight P100 P2 completions, and a hash-bound P0/GPU replay receipt. GPU ordered-ID mismatches must be exactly integer zero across all 16 cell/split identities.

Readiness-only command (no GPU work):

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge01/bridge.py --check
```

Future explicit execution, only after coordinated release and successful readiness:

```
PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge01/bridge.py --execute p100
PYTHONDONTWRITEBYTECODE=1 /data/pjh_workspace/ts-env/bin/python /data/pjh_workspace/CARTS/experiments/calibration_bridge01/bridge.py --execute full
```

Run these sequentially, never alongside old Calibration controllers. Do not invoke `--worker` directly: it requires the parent's inherited GPU lock descriptor. Use the approved Windows OpenSSH path to submit commands. No Solar, seed1/2, P3/P4, or automatic retry is included.

Launch directories are exclusive and prevent implicit recovery. An interrupted attempt requires a separately reviewed continuation; never delete a directory or weaken an assertion to rerun. Existing completed markers are not a substitute for reviewing a partial launch. GPU utilization can briefly remain nonzero after a child exits; this adapter fails closed and does not weaken that condition.

CPU tests validate control flow and isolated fixtures, not actual GPU numerical equivalence, Full memory requirements or a successful end-to-end run. Initial source-seal verification in the launcher also prevents Python optimization flags from bypassing consumer checks (consumer checks use explicit exceptions); upstream must continue to run normally without `-O`.
