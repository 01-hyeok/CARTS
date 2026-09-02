#!/bin/bash
set -uo pipefail

# Collect the self-only Top-K results into one table.
#
# Recall is the primary metric here, not MSE: alpha over the Top-K is close to
# uniform (topk_weight_entropy sits at ln(10) = 2.3026), so the model is
# effectively averaging the retrieved set and a better ranking does not
# necessarily move MSE. relation_oracle_mse is the ceiling for the arm.
#
# Usage
#   bash scripts/summarize_self_topk.sh            # both datasets
#   bash scripts/summarize_self_topk.sh ETTh1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASETS=("${@:-ETTh1 ETTm1}")
[ "$#" -gt 0 ] && DATASETS=("$@") || DATASETS=(ETTh1 ETTm1)

value_of() {  # value_of <log> <metric>
  grep -oP "^${2}: \K[-\d.]+" "$1" 2>/dev/null | tail -1
}

for DS in "${DATASETS[@]}"; do
  DIR="logs/${DS}/self_topk"
  [ -d "${DIR}" ] || continue
  echo "===== ${DS} ====="
  # cand_or = best of the k retrieved (the ceiling this arm's Top-K allows)
  # rel_or  = best relation branch, full_or = best over every candidate
  printf "%-12s %6s  %9s %9s  %8s %8s %8s  %9s %9s %9s\n" \
    arm pred final_mse final_mae R@1 R@5 R@10 cand_or rel_or full_or
  for PL in ${PRED_LENS:-96 192 336}; do
    for ARM in no_retrieval identity identity_l2 random random_l2 \
               chronos chronos_l2 chronos_eos chronos_tsrag \
               2stage_ema 2stage_ema_l2 2stage_mse 2stage_mse_l2 \
               e2e_ema e2e_ema_l2 e2e_mse e2e_mse_l2; do
      f="${DIR}/${ARM}_seq${PL}_pred${PL}.log"
      [ -f "$f" ] || continue
      grep -q 'Stage2 Test Final' "$f" || { printf "%-12s %6s  %9s\n" "${ARM}" "${PL}" "미완"; continue; }
      printf "%-12s %6s  %9s %9s  %8s %8s %8s  %9s %9s %9s\n" \
        "${ARM}" "${PL}" \
        "$(value_of "$f" final_mse)" "$(value_of "$f" final_mae)" \
        "$(value_of "$f" student_relation_oracle_recall_at_1)" \
        "$(value_of "$f" student_relation_oracle_recall_at_5)" \
        "$(value_of "$f" student_relation_oracle_recall_at_10)" \
        "$(value_of "$f" candidate_oracle_mse)" \
        "$(value_of "$f" relation_oracle_mse)" \
        "$(value_of "$f" full_oracle_mse)"
    done
  done
  if [ -s "${DIR}/_failures.txt" ]; then
    echo "--- failed ---"; cat "${DIR}/_failures.txt"
  fi
  echo
done
