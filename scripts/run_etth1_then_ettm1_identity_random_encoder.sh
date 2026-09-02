#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deprecated wrapper: running the six-condition retrieval ablation suite."
exec "${SCRIPT_DIR}/run_etth1_then_ettm1_retrieval_ablation_suite.sh"
