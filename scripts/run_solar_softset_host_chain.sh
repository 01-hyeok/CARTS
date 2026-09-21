#!/usr/bin/env bash
# Chains Solar's canonical S0_wce Stage-1 -> Stage-2 host build (H96 then
# H720), for TRACK-A-TF-ORACLE-LEARNABILITY01's Solar baseline preparation.
set -euo pipefail
cd /data/pjh_workspace/CARTS
GPU="${GPU:-1}" bash scripts/run_solar_softset_host_stage1.sh
GPU="${GPU:-1}" bash scripts/run_solar_softset_host_stage2.sh
echo "[chain] Solar canonical host build complete $(date -Is)"
