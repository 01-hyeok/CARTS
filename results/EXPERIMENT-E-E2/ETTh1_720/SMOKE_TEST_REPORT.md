# Experiment E2 Smoke Test Report

## cell
ETTh1_720

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
  "12. peak VRAM (this smoke run, ETTh1_720, subset): 5.948 GB (delta from pre-smoke: 5.501 GB) -- full-candidate run will scale with candidate-bank size per channel, same channelwise_backward pattern as train_patch_retrieval_expert01/train_horizon_retrieval_expert01"
]
```

## smoke_train_metrics
```json
{
  "train_loss": 9.492733410426547,
  "train_kl_shared": 4.9185667378561835,
  "train_teacher_entropy": 3.3780154216857183,
  "train_teacher_top1_mass": 0.24135823139832133,
  "train_teacher_effective_positives": 43.30818403334845,
  "train_n_valid_mean": 4502.416666666667,
  "train_kl_local": 5.149635212762015,
  "train_kl_trend": 4.150474150975545,
  "train_kl_seasonal": 4.422390052250454
}
```

## smoke_val_metrics
```json
{
  "shared_retmse10": 2.2927840096609935,
  "shared_oracle_retmse10": 0.872711317879813,
  "shared_recall_at_10": 0.005357143070016589,
  "shared_ndcg_at_10": 0.9154040132250104,
  "shared_overall_agg_mse": 1.7025175775800432,
  "local_retmse10": 0.30500151429857525,
  "local_oracle_retmse10": 0.2027753165790013,
  "local_recall_at_10": 0.007142857382340091,
  "local_ndcg_at_10": 0.6302615574428013,
  "local_overall_agg_mse": 1.6490771429879325,
  "trend_retmse10": 1.4918786457606725,
  "trend_oracle_retmse10": 0.3798761027199881,
  "trend_recall_at_10": 0.0017857143123235022,
  "trend_ndcg_at_10": 0.9271027020045689,
  "trend_overall_agg_mse": 1.6158969742911202,
  "seasonal_retmse10": 0.9225633144378662,
  "seasonal_oracle_retmse10": 0.45180303709847586,
  "seasonal_recall_at_10": 0.0,
  "seasonal_ndcg_at_10": 0.5609030893870762,
  "seasonal_overall_agg_mse": 1.6258070128304618,
  "n_queries_seen": 8
}
```

## fingerprint
```json
{
  "exp": "EXPERIMENT-E-E2-MULTISUBSPACE01",
  "cell": "ETTh1_720",
  "d_model": 128,
  "subspace_sizes_shared_local_trend_seasonal": [
    32,
    32,
    32,
    32
  ],
  "period": 24,
  "tau_t_per_component": {
    "shared": 0.02,
    "local": 0.1,
    "trend": 0.01,
    "seasonal": 0.1
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
    6
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