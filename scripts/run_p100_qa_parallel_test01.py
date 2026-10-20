#!/usr/bin/env python3
"""TRACK-P100-QA-TEST-EVALUATION01 -- GPU1 parallel TEST-evaluation
scheduler (Python driver for `run_p100_qa_parallel_test01.sh`).

Reads `LOCKED_CONFIG.json`'s `combos` registry (built by
`lock_qa_test_eval_config01.py`) and runs `evaluate_locked_test01.py`
once per not-yet-evaluated combo. Evaluation jobs are independent
(read-only, separate output files) so up to `--max_parallel` (default
2) run concurrently on GPU1 -- but ONLY while enough GPU1 memory is
free; this script polls `nvidia-smi` before launching each new job and
waits rather than risking an OOM against whatever training job(s) may
already be running on GPU1. A failed job is recorded as FAILED and does
not affect any other job.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qa_test_eval_paths01 import reference_ckpt


def gpu1_free_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', '-i', '1'],
        capture_output=True, text=True, check=True).stdout.strip()
    used, total = (int(x.strip()) for x in out.split(','))
    return total - used


def build_eval_cmd(key, combo, out_dir, locked_config_path):
    ds, h = combo['dataset'], combo['horizon']
    out_dir_for_combo = Path(out_dir) / combo['protocol'] / combo['cell'] / combo['method'] / combo['arm']
    return [
        sys.executable, '-u', 'scripts/evaluate_locked_test01.py',
        '--locked_config', str(locked_config_path),
        '--protocol', combo['protocol'],
        '--reference_ckpt', reference_ckpt(ds, h),
        '--method', combo['method'], '--arm', combo['arm'], '--cell', combo['cell'],
        '--pred_len', str(h), '--seq_len', str(h),
        '--base_checkpoint', combo['base_checkpoint_path'],
        '--candidate_pool_cache', combo['candidate_pool_cache'],
        '--checkpoint_path', combo['checkpoint_path'],
        '--out_dir', str(out_dir),
    ], out_dir_for_combo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--locked_config', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_parallel', type=int, default=2)
    ap.add_argument('--min_free_mib_to_launch', type=int, default=8000,
                    help='refuse to launch a new eval job unless GPU1 has at least this much free memory')
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--status_json', required=True)
    ap.add_argument('--gpu_usage_json', required=True)
    cli = ap.parse_args()

    cfg = json.loads(Path(cli.locked_config).read_text())
    combos = cfg['combos']
    Path(cli.log_dir).mkdir(parents=True, exist_ok=True)
    out_dir = Path(cli.out_dir)

    status = {}
    gpu_samples = []
    running = {}  # key -> (Popen, out_dir_for_combo, log_fh)
    pending = []
    for key, combo in combos.items():
        done_marker = out_dir / combo['protocol'] / combo['cell'] / combo['method'] / combo['arm'] / 'test_metrics.json'
        if done_marker.exists():
            status[key] = 'ALREADY_DONE'
            continue
        pending.append(key)
    total = len(pending) + sum(1 for v in status.values() if v == 'ALREADY_DONE')
    print(f'[run_p100_qa_parallel_test] {len(status)} already done, {len(pending)} to evaluate (total {total}).')

    def launch(key):
        combo = combos[key]
        cmd, out_dir_for_combo = build_eval_cmd(key, combo, out_dir, cli.locked_config)
        log_path = Path(cli.log_dir) / f"{key.replace('/', '__')}.log"
        log_fh = open(log_path, 'w')
        print(f'[run_p100_qa_parallel_test] launching {key}')
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
                                env={'CUDA_VISIBLE_DEVICES': '1', **__import__('os').environ})
        running[key] = (proc, log_fh)
        status[key] = 'RUNNING'

    def reap_finished():
        finished = []
        for key, (proc, log_fh) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_fh.close()
                status[key] = 'DONE' if ret == 0 else f'FAILED (exit {ret})'
                finished.append(key)
                print(f'[run_p100_qa_parallel_test] {key}: {status[key]}')
        for key in finished:
            del running[key]

    while pending or running:
        reap_finished()
        free = gpu1_free_mib()
        gpu_samples.append({'t': time.time(), 'gpu1_free_mib': free, 'n_running': len(running)})
        while pending and len(running) < cli.max_parallel and free >= cli.min_free_mib_to_launch:
            launch(pending.pop(0))
            free = gpu1_free_mib()
        Path(cli.status_json).write_text(json.dumps(status, indent=2))
        if pending or running:
            time.sleep(10)
    reap_finished()

    Path(cli.status_json).write_text(json.dumps(status, indent=2))
    Path(cli.gpu_usage_json).write_text(json.dumps({
        'samples': gpu_samples,
        'max_parallel_configured': cli.max_parallel,
        'min_free_mib_to_launch': cli.min_free_mib_to_launch,
    }, indent=2))

    n_done = sum(1 for v in status.values() if v in ('DONE', 'ALREADY_DONE'))
    n_failed = sum(1 for v in status.values() if str(v).startswith('FAILED'))
    print(f'[run_p100_qa_parallel_test] COMPLETE. done={n_done} failed={n_failed} total_tracked={len(status)}')


if __name__ == '__main__':
    main()
