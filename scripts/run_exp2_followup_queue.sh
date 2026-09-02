#!/usr/bin/env bash
set -uo pipefail

# Follow-up queue for Experiment 2, started once run_experiment2_diff1.sh is
# done with ETTh1. Four steps, in order:
#
#   1. ETTh1 delta_last @ 720   fills the hole that makes the diff1 vs
#                               delta_last comparison stop at 336
#   2. ETTh1 joint  (diff1)     end-to-end + retrieval KL, 96..720
#   3. ETTm1 joint  (diff1)     same, on the second dataset
#   4. ETTm1 delta_last @ 720   same hole on ETTm1
#
# ETTm1 diff1 direct/future_mse/ema is NOT here: the main driver already walks
# 96..720 for those once it leaves ETTh1.
#
# Why 720 was missing at all: run_self_topk.sh pins PRED_LENS to 96/192/336
# because its chronos arm truncates to a 512-token context, which would make
# arm-vs-arm comparison invalid at seq_len 720. That reason does not apply to
# identity / 2stage_mse / e2e_mse, so this queue asks for exactly those three
# and leaves the chronos arms alone.
#
# run_self_topk.sh runs under `set -e`, so each arm is invoked separately here;
# otherwise one failing arm would abort every step after it.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_LOG="${DRIVER_LOG:-${PROJECT_ROOT}/logs/exp2_diff1_driver.log}"
DRIVER_PID="${DRIVER_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
BASELINE_ARMS=(${BASELINE_ARMS:-identity 2stage_mse e2e_mse})
SKIP_WAIT="${SKIP_WAIT:-0}"

cd "${PROJECT_ROOT}"
FAILLOG="${PROJECT_ROOT}/logs/exp2_followup_failures.txt"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

step() {
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# [$(stamp)] ${label}"
  echo "#   \$ $*"
  echo "############################################################"
  if "$@"; then
    echo "[$(stamp)] OK: ${label}"
  else
    local rc=$?
    echo "[$(stamp)] FAILED (rc=${rc}): ${label}" | tee -a "${FAILLOG}" >&2
  fi
}

baseline_720() {
  local dataset="$1"
  for arm in "${BASELINE_ARMS[@]}"; do
    step "${dataset} delta_last @720 :: ${arm}" \
      env PRED_LENS=720 bash scripts/run_self_topk.sh "${dataset}" "${arm}"
  done
}

if [[ "${SKIP_WAIT}" != "1" ]]; then
  echo "[queue] $(stamp) waiting for ETTh1 to finish (poll ${POLL_SECONDS}s, driver pid=${DRIVER_PID:-unknown})"
  while true; do
    if grep -q '^\[ETTm1\]' "${DRIVER_LOG}" 2>/dev/null; then
      echo "[queue] $(stamp) driver moved on to ETTm1 -> ETTh1 done"
      break
    fi
    if [[ -n "${DRIVER_PID}" ]] && ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
      echo "[queue] $(stamp) driver pid ${DRIVER_PID} exited"
      break
    fi
    sleep "${POLL_SECONDS}"
  done
fi

baseline_720 ETTh1
step "ETTh1 joint (diff1 end-to-end + KL)" \
  env DATASETS=ETTh1 bash scripts/run_experiment2_diff1_joint.sh
step "ETTm1 joint (diff1 end-to-end + KL)" \
  env DATASETS=ETTm1 bash scripts/run_experiment2_diff1_joint.sh
baseline_720 ETTm1

echo
echo "[queue] $(stamp) all steps attempted"
if [[ -s "${FAILLOG}" ]]; then
  echo "[queue] failures recorded in ${FAILLOG}:"
  cat "${FAILLOG}"
else
  echo "[queue] no failures"
fi
