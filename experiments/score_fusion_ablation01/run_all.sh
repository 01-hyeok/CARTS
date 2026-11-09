#!/usr/bin/env bash
set -euo pipefail
source /data/pjh_workspace/ts-env/bin/activate
cd /data/pjh_workspace/CARTS
exec python -u experiments/score_fusion_ablation01/controller.py
