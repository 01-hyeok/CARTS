#!/usr/bin/env bash
# Backward-compatible entry point. Experiment 2 now covers ETTh1 and ETTm1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_experiment2_diff1.sh" "$@"
