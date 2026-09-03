#!/usr/bin/env bash
# Runs the whole soft_set_mse experiment as one sequential chain on GPU 1,
# waiting for any currently-running GPU job to actually exit first (checked by
# PID, not by pattern-matching a command line -- that pattern match is what
# caused a duplicate run earlier in this session).
set -uo pipefail
cd /data/pjh_workspace/CARTS

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[chain] waiting for PID $WAIT_PID to exit $(date -Is)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[chain] PID $WAIT_PID gone $(date -Is)"
fi

exec 9>/tmp/carts_soft_set_mse_chain.lock
flock -n 9 || { echo "[chain] another instance holds the lock; exiting"; exit 1; }

echo "[chain] Stage-1 sweep starting $(date -Is)"
GPU="${GPU:-1}" bash scripts/run_soft_set_mse_stage1.sh
echo "[chain] Stage-1 sweep done $(date -Is)"

echo "[chain] Stage-2 sweep starting $(date -Is)"
GPU="${GPU:-1}" bash scripts/run_soft_set_mse_stage2.sh
echo "[chain] Stage-2 sweep done $(date -Is)"

echo "[chain] ALL DONE $(date -Is)"
