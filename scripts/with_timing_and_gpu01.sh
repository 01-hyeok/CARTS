#!/usr/bin/env bash
# Generic non-invasive wrapper: records wall-clock duration and GPU
# memory/utilization (sampled every 5s on the target GPU) around an
# arbitrary command, without modifying the command's own script.
# Writes <out_dir>/timing_gpu_summary.json and <out_dir>/gpu_usage.csv.
#
# Usage:
#   bash scripts/with_timing_and_gpu01.sh <out_dir> <gpu_index> -- <command...>
#
# Example:
#   bash scripts/with_timing_and_gpu01.sh results/MY-TRACK/cellA 1 -- \
#     python -u scripts/train_foo01.py --arg1 val1
set -uo pipefail
cd "$(dirname "$0")/.."

out_dir="$1"; gpu_index="$2"; shift 2
if [ "$1" != "--" ]; then
  echo "[with_timing_and_gpu01] expected '--' before the command" >&2
  exit 1
fi
shift

mkdir -p "$out_dir"
gpu_csv="$out_dir/gpu_usage.csv"
summary_json="$out_dir/timing_gpu_summary.json"

python3 -u scripts/record_gpu_usage01.py --out_csv "$gpu_csv" --gpu_index "$gpu_index" --interval 5 &
sampler_pid=$!

start_ts=$(date +%s.%N)
start_iso=$(date -Iseconds)
"$@"
cmd_status=$?
end_ts=$(date +%s.%N)
end_iso=$(date -Iseconds)

kill "$sampler_pid" 2>/dev/null
wait "$sampler_pid" 2>/dev/null

python3 - "$gpu_csv" "$summary_json" "$start_iso" "$end_iso" "$start_ts" "$end_ts" "$cmd_status" "$gpu_index" <<'PYEOF'
import csv
import json
import sys

gpu_csv, summary_json, start_iso, end_iso, start_ts, end_ts, cmd_status, gpu_index = sys.argv[1:9]
mems, utils = [], []
try:
    with open(gpu_csv) as f:
        for row in csv.DictReader(f):
            try:
                mems.append(int(row['memory_used_mib']))
                utils.append(int(row['utilization_pct']))
            except (ValueError, KeyError):
                pass
except FileNotFoundError:
    pass

summary = {
    'start_iso': start_iso,
    'end_iso': end_iso,
    'duration_sec': round(float(end_ts) - float(start_ts), 3),
    'command_exit_status': int(cmd_status),
    'gpu_index': int(gpu_index),
    'gpu_mem_used_mib_mean': round(sum(mems) / len(mems), 1) if mems else None,
    'gpu_mem_used_mib_max': max(mems) if mems else None,
    'gpu_util_pct_mean': round(sum(utils) / len(utils), 1) if utils else None,
    'gpu_util_pct_max': max(utils) if utils else None,
    'n_gpu_samples': len(mems),
}
with open(summary_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[with_timing_and_gpu01] wrote {summary_json}: {summary}")
PYEOF

exit "$cmd_status"
