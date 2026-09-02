#!/bin/bash
set -euo pipefail

exec "$(dirname "$0")/../run_oracle_mlp_repeat_seqeqpred.sh" ETTm1 mse
