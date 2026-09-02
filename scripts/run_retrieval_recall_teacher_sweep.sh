#!/usr/bin/env bash
set -uo pipefail

# Recall@K sweep for one MLP teacher, as a companion to
# run_retrieval_recall_sweep.sh.
#
# The teacher is what the encoder's KL target is built from:
#   ema_future -- sim(E'(query_future), E'(candidate_future)), E' = EMA(E)
#   future_mse -- -MSE(query_future, candidate_future), no encoder at all
#
# Only encoder=mlp is swept: identity trains nothing, so its rows are the same
# under either teacher and are already in the ema sweep's CSV.
#
#   4 representations x 1 similarity x 4 lengths x 2 datasets = 32 configurations
#
# Resumable: a configuration already present in the output CSV is skipped, so
# this can be re-run after an interrupt and it picks up where it stopped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/pjh_workspace/ts-env/bin/activate}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtual environment activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"

TEACHER="${TEACHER:-future_mse}"
DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
REPRESENTATIONS=(${REPRESENTATIONS:-raw delta_last arima_residual decomposition})
SIMILARITIES=(${SIMILARITIES:-cosine})
SEED="${SEED:-0}"
MOVING_AVG="${MOVING_AVG:-25}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/metrics/retrieval_recall}"
# Kept out of recall_seed${SEED}.csv: that file's header was fixed before the
# teacher column existed, and appending a wider row to it would misalign it.
OUT_CSV="${OUT_CSV:-${OUT_DIR}/recall_seed${SEED}_${TEACHER}.csv}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_DIR}"
FAILLOG="${OUT_DIR}/_failures_${TEACHER}.txt"
[[ "${FORCE}" == 1 ]] && rm -f "${OUT_CSV}"

already_done() {
  # dataset,seq,pred,representation,encoder,similarity,teacher identify a row.
  [[ -f "${OUT_CSV}" ]] || return 1
  python - "$@" <<'PY'
import csv, sys
path, data, seq, pred, rep, enc, sim, teacher = sys.argv[1:9]
try:
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            # A row written before the teacher column existed is an ema one.
            if (row['dataset'] == data and row['seq_len'] == seq
                    and row['pred_len'] == pred and row['representation'] == rep
                    and row['encoder'] == enc and row['similarity'] == sim
                    and row.get('teacher', 'ema_future') == teacher):
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

total=0
skipped=0
failed=0

echo "=== teacher=${TEACHER} encoder=mlp similarities=${SIMILARITIES[*]} ==="

for DATASET in "${DATASETS[@]}"; do
  case "${DATASET}" in
    ETTh1) DATA_PATH=ETTh1.csv; FREQ=h ;;
    ETTm1) DATA_PATH=ETTm1.csv; FREQ=t ;;
    *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
  esac

  for PRED_LEN in "${PRED_LENS[@]}"; do
    SEQ_LEN="${PRED_LEN}"
    for REPRESENTATION in "${REPRESENTATIONS[@]}"; do
      for SIMILARITY in "${SIMILARITIES[@]}"; do
        total=$((total + 1))
        if [[ "${FORCE}" != 1 ]] && already_done \
            "${OUT_CSV}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
            "${REPRESENTATION}" mlp "${SIMILARITY}" "${TEACHER}"; then
          skipped=$((skipped + 1))
          continue
        fi

        echo "[$(date -u '+%H:%M:%SZ')] ${TEACHER} ${DATASET} L=${SEQ_LEN} ${REPRESENTATION}/${SIMILARITY}"
        if ! python -u eval/retrieval_recall.py \
          --data "${DATASET}" \
          --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
          --data_path "${DATA_PATH}" \
          --freq "${FREQ}" \
          --seq_len "${SEQ_LEN}" \
          --pred_len "${PRED_LEN}" \
          --representation "${REPRESENTATION}" \
          --encoder mlp \
          --teacher "${TEACHER}" \
          --similarity "${SIMILARITY}" \
          --moving_avg "${MOVING_AVG}" \
          --seed "${SEED}" \
          --output "${OUT_CSV}"; then
          failed=$((failed + 1))
          echo "[FAILED] ${TEACHER} ${DATASET} seq${SEQ_LEN} ${REPRESENTATION} ${SIMILARITY}" \
            | tee -a "${FAILLOG}" >&2
        fi
      done
    done
  done
done

echo
echo "configurations: ${total}  skipped(done): ${skipped}  failed: ${failed}"
echo "results: ${OUT_CSV}"
