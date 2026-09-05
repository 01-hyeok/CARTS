#!/usr/bin/env bash
# EXP-3 closure check: representative downstream evaluation.
#
# Not the full 5-arm x 4-cell sweep. Per cell, only S0 (WCE baseline) and the
# single non-baseline arm with the best *validation* hard_aggregate_mse10 from
# the already-completed Stage-1 sweep (never the arm's own training objective,
# never TEST). Selection was done by reading the "[stage1] new best on
# hard_aggregate_mse10" lines already in logs/soft_set_mse/<ds>/pred<h>/*.log --
# no new Stage-1 run was needed or performed for this choice.
#
#   ETTh1/96   -> S2_lam10    (val hard_agg -0.691330, best of S1-S4)
#   ETTh1/720  -> S3_lam30    (val hard_agg -1.592974, best of S1-S4)
#   Weather/96 -> S1_set_only (val hard_agg -0.472267, best of S1-S4)
#   Weather/720-> S3_lam30    (val hard_agg -0.700912, best of S1-S4)
#
# Delegates to run_soft_set_mse_stage2.sh per cell via SPECS_OVERRIDE/ARMS, so
# the underlying Stage-2 invocation, checkpoint lookup, and the FAIL guard it
# now carries are exactly the ones already reviewed -- this wrapper only
# narrows which (cell, arm) pairs get requested.
set -uo pipefail
cd /data/pjh_workspace/CARTS

run_cell () {
  local ds="$1" pred="$2" enc="$3" dkey="$4" root="$5" dpath="$6" arms="$7"
  SPECS_OVERRIDE="$ds $pred $enc $dkey $root $dpath" ARMS="S0_wce $arms" \
    bash scripts/run_soft_set_mse_stage2.sh
}

status=0
run_cell ETTh1   96  7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/ ETTh1.csv   S2_lam10    || status=1
run_cell ETTh1   720 7  ETTh1  ../Dataset/Time-Series-Library_dataset/ETT-small/ ETTh1.csv   S3_lam30    || status=1
run_cell Weather 96  21 custom ../Dataset/Time-Series-Library_dataset/weather/   weather.csv S1_set_only || status=1
run_cell Weather 720 21 custom ../Dataset/Time-Series-Library_dataset/weather/   weather.csv S3_lam30    || status=1

echo "[representative] all cells attempted, overall_status=$([ $status -eq 0 ] && echo OK || echo FAIL)"
exit $status
