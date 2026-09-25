# Experiment E2 Smoke Test Report

## cell
Weather_720

## checks
```json
[
  "1. encoder input never includes future y: PASS (encode_raw(model, batch_x/memory_x, c) only)",
  "2/3. teacher detached, no grad to teacher: PASS (component_distance_memsafe is @torch.no_grad(); asserted p_t_e.requires_grad is False every step)",
  "4. four subspace shapes: PASS (asserted zq/zk shapes == subspace_sizes every step)",
  "5. per-subspace L2 norm == 1: PASS (asserted every step, atol=1e-4)",
  "6. component score shapes [B,N]: PASS (asserted every step)",
  "7. mask -> no invalid candidate selected: PASS (stable_topk_indices operates on score.masked_fill(~cand_mask, -inf); invalid candidates score -inf, cannot enter Top-K given n_valid >> top_k)",
  "8. component loss finite: PASS (asserted torch.isfinite every step)",
  "9. gradient exists on encoder parameters after backward: PASS (asserted nonzero grad sum across model.parameters() every step)",
  "10. router freeze/unfreeze: N/A (no router in E2)",
  "11. candidate re-encoding policy matches production: PASS (encode_raw(model, exp.memory_x, c) recomputed every batch, unmodified from train_factorial_e2e01/train_patch_retrieval_expert01)",
  "12. peak VRAM (this smoke run, Weather_720, subset): 31.521 GB (delta from pre-smoke: 25.076 GB) -- full-candidate run will scale with candidate-bank size per channel, same channelwise_backward pattern as train_patch_retrieval_expert01/train_horizon_retrieval_expert01"
]
```

## smoke_train_metrics
```json
{
  "train_loss": 10.190792304951518,
  "train_kl_shared": 5.00631261829819,
  "train_teacher_entropy": 5.054009233202253,
  "train_teacher_top1_mass": 0.08849006072947911,
  "train_teacher_effective_positives": 4002.371679169791,
  "train_n_valid_mean": 32569.0,
  "train_kl_local": 5.5499413555399295,
  "train_kl_trend": 4.288377831113481,
  "train_kl_seasonal": 5.71511917481465
}
```

## smoke_val_metrics
```json
{
  "shared_retmse10": 1.261752310253325,
  "shared_oracle_retmse10": 0.16976611954825266,
  "shared_recall_at_10": 0.000595238104107834,
  "shared_ndcg_at_10": 0.9388702029273623,
  "shared_overall_agg_mse": 0.6707461220877511,
  "local_retmse10": 0.15984146367935909,
  "local_oracle_retmse10": 0.04065272779691787,
  "local_recall_at_10": 0.0,
  "local_ndcg_at_10": 0.7989618664696103,
  "local_overall_agg_mse": 0.649911085764567,
  "trend_retmse10": 0.48684056599934894,
  "trend_oracle_retmse10": 0.018788571159044903,
  "trend_recall_at_10": 0.001190476208215668,
  "trend_ndcg_at_10": 0.9656043279738653,
  "trend_overall_agg_mse": 0.6531305086045038,
  "seasonal_retmse10": 1.312301272437686,
  "seasonal_oracle_retmse10": 0.17066625754038492,
  "seasonal_recall_at_10": 0.000595238104107834,
  "seasonal_ndcg_at_10": 0.725739910489037,
  "seasonal_overall_agg_mse": 0.654949460710798,
  "n_queries_seen": 8
}
```

## fingerprint
```json
{
  "exp": "EXPERIMENT-E-E2-MULTISUBSPACE01",
  "cell": "Weather_720",
  "d_model": 128,
  "subspace_sizes_shared_local_trend_seasonal": [
    32,
    32,
    32,
    32
  ],
  "period": 142,
  "tau_t_per_component": {
    "shared": 0.01,
    "local": 0.02,
    "trend": 0.005,
    "seasonal": 0.02
  },
  "tau_s": 0.1,
  "lambda_component": 1.0,
  "relation_encoder_type_corrected_from_host": "mlp -> transformer",
  "relation_self_fill": "zero",
  "patch_len": 16,
  "stride": 16,
  "param_count": 2688160,
  "top_k": 10,
  "init_seed": 0,
  "loader_seed": 0,
  "channels": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20
  ],
  "encoder_init_sha256": "ac3073e368439bcdea02a86fbb75669572d71a1bab2b712c0b89e8f8d30fd115",
  "learning_rate": 0.001,
  "batch_size": 4,
  "epochs": 10,
  "patience": 5,
  "checkpoint_criterion": "[ISSUE] spec ambiguous on which subspace Top-10 to checkpoint on -- using E-Shared subspace val overall_agg_mse as the direct analogue of the existing single-encoder production metric",
  "code_commit": "10531fa5a65edf6a52f80a76dd35451bdfee148f"
}
```