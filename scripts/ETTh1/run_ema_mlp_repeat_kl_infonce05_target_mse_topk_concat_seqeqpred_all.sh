#!/bin/bash
set -euo pipefail

export INFONCE_POSITIVE_SOURCE=target_mse
exec bash "$(dirname "$0")/run_ema_mlp_repeat_kl_infonce05_concat_seqeqpred_all.sh" "$@"
