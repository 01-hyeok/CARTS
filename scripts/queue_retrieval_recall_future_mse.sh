#!/usr/bin/env bash
set -uo pipefail

# Waits for the running ema sweep to exit, then runs the future_mse sweep on the
# same GPU. Serialised rather than run alongside it: the CSV records a `seconds`
# column per configuration, and two jobs sharing one GPU would make those
# numbers a measure of contention instead of cost.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PID="${WAIT_PID:-}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[queue] waiting for pid ${WAIT_PID} (ema sweep) to finish"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[queue] pid ${WAIT_PID} gone at $(date -u '+%F %H:%M:%SZ'); starting future_mse sweep"
fi

TEACHER=future_mse exec "${SCRIPT_DIR}/run_retrieval_recall_teacher_sweep.sh"
