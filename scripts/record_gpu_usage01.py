#!/usr/bin/env python3
"""Lightweight GPU usage sampler for experiment reports.

Polls `nvidia-smi` for one GPU index at a fixed interval and appends
(timestamp, memory.used, utilization.gpu) rows to a CSV. Meant to run as a
background sidecar alongside a training/eval process -- never imports or
touches any experiment code, so it is safe to attach to ANY existing run
(including ones whose own scripts must stay unmodified) without changing
their behavior.

Usage:
    python scripts/record_gpu_usage01.py --out_csv <path> --gpu_index 1 --interval 5
Stop with SIGTERM/SIGINT (e.g. `kill <pid>`); it flushes after every sample.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path


def sample(gpu_index):
    out = subprocess.run(
        ['nvidia-smi', '-i', str(gpu_index),
         '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    mem_str, util_str = [x.strip() for x in out.split(',')]
    return int(mem_str), int(util_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--gpu_index', type=int, required=True)
    ap.add_argument('--interval', type=float, default=5.0)
    args = ap.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with open(out_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['unix_ts', 'gpu_index', 'memory_used_mib', 'utilization_pct'])
        while True:
            try:
                mem, util = sample(args.gpu_index)
                writer.writerow([time.time(), args.gpu_index, mem, util])
                f.flush()
            except Exception as e:
                writer.writerow([time.time(), args.gpu_index, 'ERROR', str(e)])
                f.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
