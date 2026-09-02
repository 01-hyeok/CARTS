#!/usr/bin/env bash
set -uo pipefail

DRIVER_PID="${1:?usage: monitor_experiment2_diff1.sh DRIVER_PID [INTERVAL_SECONDS]}"
INTERVAL_SECONDS="${2:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/logs/experiment2_diff1_self_only_seed0}"
MONITOR_LOG="${MONITOR_LOG:-${RESULT_ROOT}/monitor.log}"
DRIVER_OUTPUT="${DRIVER_OUTPUT:-${PROJECT_ROOT}/logs/exp2_diff1_driver.log}"

mkdir -p "${RESULT_ROOT}"

log_line() {
  printf '%s | monitor_pid=%s driver_pid=%s | %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "${DRIVER_PID}" "$*" \
    >> "${MONITOR_LOG}"
}

log_line "MONITOR_START host=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unavailable)"

while kill -0 "${DRIVER_PID}" 2>/dev/null; do
  active_python="$(
    ps -eo pid,ppid,stat,etime,args \
      | awk '/python -u run.py/ && /exp2_diff1/ {print; exit}'
  )"
  latest_log="$(
    find "${RESULT_ROOT}" -type f -name '*.log' \
      ! -name 'monitor.log' ! -name 'driver.log' ! -name 'launcher.log' \
      -printf '%T@ %s %p\n' 2>/dev/null \
      | sort -nr \
      | head -1
  )"
  gpu_state="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader 2>/dev/null \
      | sed -n '2p'
  )"
  log_line "HEARTBEAT active=${active_python:-none} latest_log=${latest_log:-none} gpu1=${gpu_state:-unavailable}"
  sleep "${INTERVAL_SECONDS}" || break
done

log_line "DRIVER_DISAPPEARED"
if [[ -r "${DRIVER_OUTPUT}" ]]; then
  if grep -Eq 'Traceback|CUDA out of memory|Killed|\[FAILED\]|SIGNAL=|EXIT status=' "${DRIVER_OUTPUT}"; then
    reason="$(grep -E 'Traceback|CUDA out of memory|Killed|\[FAILED\]|SIGNAL=|EXIT status=' "${DRIVER_OUTPUT}" | tail -5 | tr '\n' ';')"
    log_line "TERMINATION_EVIDENCE ${reason}"
  else
    log_line "TERMINATION_EVIDENCE none_in_driver_log"
  fi
  log_line "DRIVER_LOG_END_BEGIN"
  tail -20 "${DRIVER_OUTPUT}" >> "${MONITOR_LOG}"
  log_line "DRIVER_LOG_END_FINISH"
fi
