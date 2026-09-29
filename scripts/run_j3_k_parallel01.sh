#!/bin/bash
# TRACK-J3 + TRACK-K parallel execution driver.
#
# Topology (must not change):
#   J3 diagnostic  -- CPU only, no training
#         ||  (runs in parallel with the block below)
#   TRACK-K (GPU 1): K1 -> K2, SEQUENTIAL (never parallel on the same GPU)
set -uo pipefail

mkdir -p logs/j3_k_parallel01

REF="checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth"

# ------------------------------------------------------------
# Process A: J3 diagnostic -- CPU ONLY, no training
# ------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=8

    python -u scripts/diag_j3_error_complementarity01.py \
        > logs/j3_k_parallel01/j3_error_decomposition.log 2>&1

    echo $? > logs/j3_k_parallel01/j3.exit
) &
J3_PID=$!

# ------------------------------------------------------------
# Process B: TRACK-K -- GPU 1, K1 -> K2 sequentially
# ------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1

    python -u scripts/train_k_multislot_predictive_retrieval01.py \
        --reference_ckpt "${REF}" --arm K1_multislot_relevance --cell ETTh1_720 \
        > logs/j3_k_parallel01/k1_multislot.log 2>&1
    K1_STATUS=$?

    if [ "${K1_STATUS}" -ne 0 ]; then
        echo "${K1_STATUS}" > logs/j3_k_parallel01/k.exit
        exit "${K1_STATUS}"
    fi

    python -u scripts/train_k_multislot_predictive_retrieval01.py \
        --reference_ckpt "${REF}" --arm K2_multislot_aggregate --cell ETTh1_720 \
        > logs/j3_k_parallel01/k2_multislot.log 2>&1

    echo $? > logs/j3_k_parallel01/k.exit
) &
K_PID=$!

wait "${J3_PID}"
J3_STATUS=$(cat logs/j3_k_parallel01/j3.exit)

wait "${K_PID}"
K_STATUS=$(cat logs/j3_k_parallel01/k.exit)

echo "J3 status: ${J3_STATUS}"
echo "TRACK-K status: ${K_STATUS}"

if [ "${J3_STATUS}" -ne 0 ] || [ "${K_STATUS}" -ne 0 ]; then
    exit 1
fi
