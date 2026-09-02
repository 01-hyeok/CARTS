#!/usr/bin/env bash
set -uo pipefail

# Retrieval-only Recall@K sweep.
#
# Order is encoder-major, then dataset: every identity configuration on ETTh1,
# then every identity configuration on ETTm1, and only then the MLP ones. The
# identity half needs no training at all, so this front-loads every result that
# can be produced without waiting on an encoder.
#
#   4 representations x 3 similarities x 4 lengths x 2 datasets x 2 encoders
#     = 192 configurations
#
# Fairness is not maintained by matching flags here: eval/retrieval_recall.py
# builds the query set, candidate pool, valid mask and the raw-future-MSE oracle
# in one place, identically for every row it writes.

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

ENCODERS=(${ENCODERS:-identity mlp})
DATASETS=(${DATASETS:-ETTh1 ETTm1})
PRED_LENS=(${PRED_LENS:-96 192 336 720})
REPRESENTATIONS=(${REPRESENTATIONS:-raw delta_last arima_residual decomposition})
# The similarity axis is only swept on raw inputs. Once an encoder is in the
# way, the embedding geometry is what the encoder was trained under, so the
# comparison there is held at cosine instead of re-sweeping the metric.
#   identity: 4 rep x 3 sim x 4 len x 2 data = 96
#   mlp     : 4 rep x 1 sim x 4 len x 2 data = 32
IDENTITY_SIMILARITIES=(${IDENTITY_SIMILARITIES:-cosine pearson mse})
MLP_SIMILARITIES=(${MLP_SIMILARITIES:-cosine})
SEED="${SEED:-0}"
MOVING_AVG="${MOVING_AVG:-25}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/metrics/retrieval_recall}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/recall_seed${SEED}.csv}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_DIR}"
FAILLOG="${OUT_DIR}/_failures.txt"
[[ "${FORCE}" == 1 ]] && rm -f "${OUT_CSV}"

already_done() {
  # dataset,seq,pred,representation,encoder,similarity uniquely identify a row.
  [[ -f "${OUT_CSV}" ]] || return 1
  python - "$@" <<'PY'
import csv, sys
path, data, seq, pred, rep, enc, sim = sys.argv[1:8]
try:
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if (row['dataset'] == data and row['seq_len'] == seq
                    and row['pred_len'] == pred and row['representation'] == rep
                    and row['encoder'] == enc and row['similarity'] == sim):
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

total=0
skipped=0
failed=0

for ENCODER in "${ENCODERS[@]}"; do
  if [[ "${ENCODER}" == identity ]]; then
    SIMILARITIES=("${IDENTITY_SIMILARITIES[@]}")
  else
    SIMILARITIES=("${MLP_SIMILARITIES[@]}")
  fi
  echo "=== encoder=${ENCODER} similarities=${SIMILARITIES[*]} ==="

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
              "${REPRESENTATION}" "${ENCODER}" "${SIMILARITY}"; then
            skipped=$((skipped + 1))
            continue
          fi

          echo "[$(date -u '+%H:%M:%SZ')] ${ENCODER} ${DATASET} L=${SEQ_LEN} ${REPRESENTATION}/${SIMILARITY}"
          if ! python -u eval/retrieval_recall.py \
            --data "${DATASET}" \
            --root_path ../Dataset/Time-Series-Library_dataset/ETT-small/ \
            --data_path "${DATA_PATH}" \
            --freq "${FREQ}" \
            --seq_len "${SEQ_LEN}" \
            --pred_len "${PRED_LEN}" \
            --representation "${REPRESENTATION}" \
            --encoder "${ENCODER}" \
            --similarity "${SIMILARITY}" \
            --moving_avg "${MOVING_AVG}" \
            --seed "${SEED}" \
            --output "${OUT_CSV}"; then
            failed=$((failed + 1))
            echo "[FAILED] ${ENCODER} ${DATASET} seq${SEQ_LEN} ${REPRESENTATION} ${SIMILARITY}" \
              | tee -a "${FAILLOG}" >&2
          fi
        done
      done
    done
  done
done

echo
echo "configurations: ${total}  skipped(done): ${skipped}  failed: ${failed}"
echo "results: ${OUT_CSV}"
