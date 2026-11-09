# P100 Head Retrieval Quality — LEGACYPOOL02

Approved teacher: existing z-score. Stage1 only,144 fresh runs. Seed0 all48 -> seed1 all48 -> seed2 all48. Physical GPU1 only, one worker. No Stage2 or upload.

Pool correction: E96/E720/W96/W720 reuse immutable verified legacy ordered IDs. Eight new cells invoke original precompute_split with DataLoader batch32/query_chunk256 and full memory input layout. Complete ordered-ID equality, eligibility and hashes are required before training each cell. See reports/POOL_CORRECTION_AUDIT_KO.md and LEGACY_BASELINE_FULL_ORDERED_AUDIT.json.

Superseded results are preserved under results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01 and excluded. This revision's official output root is results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02; checkpoints use checkpoints/track_p100_head_retrieval_quality01_legacy_pool02.

Runtime gates and their actual status are reports/RUNTIME_GATES.json and runtime_gates.log. Live training status is the new results root/current.json and progress.json. Do not run preparation generators after launch.

The controller is single-worker and lock-protected. To request a safe pause, create PAUSE_REQUESTED.json in the new results root; the current run finishes, then the controller exits at the run boundary. Remove only that request after checking state to resume. Completed runs are verified and skipped; an interrupted epoch resumes from the last complete checkpoint with optimizer and RNG states. Unexpected errors stop the queue and preserve artifacts.


## Current execution scope (2026-10-30)
User cancelled seeds 1/2. Only 48 seed0 runs are active. The 96 cancelled rows are in configs/cancelled_runs.json and are not failures. Use orchestration_seed0/controller_seed0.py and orchestration_seed0/summarize_seed0.py. The original 144-row controller is historical and fails its inventory assertion against the new 48-row active manifest. Training runtime, source hashes and checkpoints are unchanged.
