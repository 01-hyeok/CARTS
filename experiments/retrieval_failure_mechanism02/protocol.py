"""Test-independent analysis definitions, fixed before Mechanism02 inference."""
from pathlib import Path

ROOT = Path('/data/pjh_workspace/CARTS')
CELLS = [f'{d}_{h}' for d in ('ETTh1', 'Weather') for h in (96, 192, 336, 720)]
MODELS = ('V0', 'V1', 'V5')
SUPPORTS = ('P100', 'FULL')
GPU_UUID = 'GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
PROTOCOL = {
    'version': 'MECHANISM02_v2_parallel_authorized_20261102_064812',
    'source_library_id': 'libfile_c73041716eac8191a0faae25ecdb7209',
    'source_lines': 1311,
    'authorization': '2026-11-02: existing checkpoints frozen; train V1 ONLY if valid independent V1 unavailable',
    'cells': CELLS, 'models': MODELS, 'seed': 0, 'k': 10,
    'physical_gpu': 1, 'gpu_uuid': GPU_UUID,
    'execution_order': '2026-11-02 06:48:12 user supersedes Solar-completion wait: one new diagnostic may coexist with existing Solar on physical GPU1 after CPU tests, actual free-memory and coexistence smoke; each cell P100+FULL before next cell',
    'concurrency': {'maximum_new_gpu_workers':1,'allowed_other_worker':'existing authorized Solar Full8 or its model-gap diagnostic','minimum_free_before_start_gib':8,'new_worker_allocated_cap_fraction':.18,'timing':'shared GPU observations; not standalone throughput'},
    'channel_batch': 'ALL channels in each encoder/score/distance GPU call; no serial fallback',
    'chunking': 'query/candidate axes only; global masked softmax across entire support',
    'core_population': 'all validation/test queries and all channels, including tail batches',
    'global_correlation_sampling': {'max_query_timestamps': 256, 'rule': 'unique linspace integer row indices across entire split', 'all_methods_same': True, 'all_channels': True, 'selection_uses_future': False},
    'sampling_exclusions': 'No sampling for RetMSE/AggMSE/AggMAE/Recall/regret/exchange/IDs/head utility/coverage/near-optimal/multimodality',
    'ndcg': {'name': 'NDCG_inverse_MSE_v1', 'gain': '1/(1+d)', 'discount': '1/log2(rank+1)', 'exponentiation': False, 'idcg': 'best k over ENTIRE valid support', 'legacy': 'separately named; never substituted'},
    'recall': {'Recall@k_vs_Oracle@k': 'intersection(S_k,O_k)/k, k=10,50,100', 'Oracle10Coverage@k': 'intersection(S_k,O_10)/10; separate name'},
    'tie_policy': 'stable inherited support order; P100 stored ordered global IDs, Full ascending canonical global ID',
    'aggregation': 'horizon within error; equal query-channel weight; float64 accumulated sums/counts',
    'normalization': 'unchanged train-fitted StandardScaler; delta-last reconstruction; no inverse transform',
    'regret': {'identity_unit': 'each query-channel', 'negative_clamp': False, 'share': 'ratio of aggregate sums; abs(total_sum)<=1e-12 => undefined', 'less_than_10_valid': 'INVALID, stop; no silent smaller K', 'worst_query': 'mean channels per timestamp, rank timestamps by regret; ceil(fraction*Q), fractions=.01,.05,.10'},
    'support_intervention': 'same frozen weights, exact same Full logits; P100 is verified ordered subset of masked Full; native canonical scores separately reproduced',
    'normalization_only': {'identity': 'pFull_restricted=mean_h(exp(logZp-logZf)*pP_h)', 'implicit_head_weight': 'm_h/sum_h(m_h), m_h=Zp/Zf', 'arithmetic': 'logsumexp; also retain log probability for single-head rank invariance', 'single_head': 'same P100 relative order MUST remain identical'},
    'union': {'deployable': 'deduplicate global IDs of 5 head Top10; choose K10 by original ENTIRE-support mixture score; no union re-softmax', 'oracle': 'future-selected union Top10 is diagnostic only', 'budget_note': 'equal K does not imply equal compute'},
    'best_single': 'one head per cell/training-support/evaluation-support selected by aggregate validation RetMSE; tie lower head ID; fixed for test',
    'posthoc_single': 'POSTHOC_ORACLE_HEAD, separate, never official',
    'near_optimal': {'epsilon_per_channel': 'max(1e-8, 0.01*median_validation_oracle_kth_mse))', 'threshold': 'test d_i <= test d_oracle10 + frozen validation epsilon_c', 'unique_relevant': 'head Top10 candidate not in other heads Top10 and passes this near-optimal threshold', 'exclusive_oracle': 'head-exclusive candidate in exact Oracle10; separate from near-optimal'},
    'geometry': {'positive_sizes': [10, 30], 'spaces': ['raw', 'delta_last', 'pearson_normalized', 'V0_candidate', 'V1_candidate', 'V5_shared_candidate'], 'cluster_grid': [1, 2, 3], 'minimum_cluster_size': 3, 'k1': 'null/no-cluster', 'validation_choice': 'per cell/support/space/set-size, choose k among 2/3 only if median silhouette>=.25, median subsample ARI>=.8, median silhouette exceeds matched-size random control by>=.10; otherwise k1; freeze for test', 'stability': '20 deterministic 80% candidate subsamples, compare shared-ID assignments; seed0', 'controls': ['same_size_random', 'same_size_past_nearest', 'future_mse_matched_if_feasible'], 'shared_ids': 'report set overlap; never call overlap-induced similarity independent evidence', 'verdict': 'test failure/degenerate geometry => descriptive/INCONCLUSIVE, no forced multimodality'},
    'statistics': {'seed': 0, 'repeats': 2000, 'unit': 'time blocks with all channels jointly', 'block_lengths': 'primary L+H, sensitivities H and 2*(L+H); fixed for all comparisons', 'insufficient': 'fewer than four effective primary blocks => descriptive CI only, no significance claim', 'multiple_comparisons': 'exploratory, no unadjusted significance claims', 'training_seed_uncertainty': 'not estimated'},
    'probe': {'reuse': 'existing frozen Ridge only when all provenance/feature/label gates match', 'fit': 'train-only labels with complete future within train boundary', 'choose': 'validation only', 'features': 'past only; no query/candidate IDs'},
    'new_v1_if_needed': {'architecture': 'same independently trained single SlotHeads projection; no V5-head extraction', 'seed': 0, 'epochs': 10, 'batch': 32, 'adam_lr': .001, 'teacher': 'existing valid-support z-score FutureMSE, negative temperature .1', 'student_temperature': .1, 'selection': 'min validation RetMSE@10, earlier epoch on tie; test never used', 'note': 'fresh runs separately identified; do not claim fixed-budget causal parity with historical early-stopped Full models'},
    'forbidden': ['V0/V5 retraining', 'Stage2', 'new loss', 'new architecture', 'seed1/2', 'GPU0', 'channel reduction', 'more than one new Mechanism02 GPU worker', 'push/upload', 'original result overwrite'],
}
