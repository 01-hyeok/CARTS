#!/bin/bash
set -euo pipefail

export INFONCE_POSITIVE_SOURCE=ema_cosine
exec bash "$(dirname "$0")/run_ema_mlp_repeat_kl_infonce05_concat_seqeqpred_all.sh" "$@"
