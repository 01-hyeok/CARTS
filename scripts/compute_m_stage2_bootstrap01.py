#!/usr/bin/env python3
"""TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 Stage2 -- PART 20 paired
bootstrap significance testing. NO TRAINING: loads each arm's already
-selected best checkpoint (`checkpoints/track_m_relevance_constrained_multislot01/
stage2/ETTh1_720/<arm>/checkpoint.pth`, chosen by min val final_mse,
PART 19), re-evaluates on the test split with that arm's own cache,
collecting PER-QUERY mean-squared-error (averaged over channels and
horizon) so queries can be paired across arms by `query_start_idx` --
the same resampling unit (all channels resampled jointly) used
throughout TRACK-J3/L's bootstraps this session.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, freeze_retrieval_submodules, load_cache_as_lookup

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
ARMS = ('S0_base', 'S1_J1', 'S2_K2', 'S3_Mstar')
CKPT_DIR = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720'
CACHE_DIR = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720'
OUT_DIR = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2'
N_BOOT = 10000
SEED = 0


@torch.no_grad()
def per_query_mse(arm, device):
    bl = torch.load(CKPT_DIR / arm / 'checkpoint.pth', map_location=device)
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    freeze_retrieval_submodules(model)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()

    cache, lut = load_cache_as_lookup(CACHE_DIR / arm / 'test.pt')
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    starts, per_q = [], []
    for batch_x, batch_y, batch_start_idx in test_loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(batch_x.device) > 0
        rows = [lut[int(s)] for s in batch_start_idx.tolist()]
        idx = torch.tensor(rows, dtype=torch.long)
        rcache = {'relation_outputs': cache['relation_outputs'][idx].to(device),
                  'relation_query_embs': cache['relation_query_embs'][idx].to(device),
                  'query_offset': cache['query_offset'][idx].to(device)}
        y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
            rcache['relation_outputs'], batch_x=batch_x, retrieval_cache=rcache,
            memory_y=exp.memory_y, valid_mask=cand_mask, key_bank=exp.key_bank,
            memory_x_last=exp.memory_x_last, target_y=batch_y)
        se_per_q = ((y_final - batch_y) ** 2).mean(dim=(1, 2))  # [B], averaged over horizon+channels
        for b in range(batch_x.size(0)):
            if bool(valid_query[b]):
                starts.append(int(batch_start_idx[b]))
                per_q.append(float(se_per_q[b]))
    return dict(zip(starts, per_q))


def paired_bootstrap(diffs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    diffs = np.asarray(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {'mean_diff': float(diffs.mean()), 'ci_lo_2.5': float(lo), 'ci_hi_97.5': float(hi),
           'ci_excludes_zero': bool(lo > 0 or hi < 0), 'n_queries': n}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = per_query_mse(arm, device)
        print(f'[bootstrap] {arm}: n_queries={len(per_arm[arm])} mean_mse={np.mean(list(per_arm[arm].values())):.6f}')

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    assert len(common_ids) > 0
    print(f'[bootstrap] common query_start_idx across all 4 arms: {len(common_ids)}')

    arrs = {arm: np.array([per_arm[arm][q] for q in common_ids]) for arm in ARMS}

    comparisons = [('S1_J1', 'S0_base'), ('S2_K2', 'S0_base'), ('S3_Mstar', 'S0_base'),
                  ('S3_Mstar', 'S1_J1'), ('S3_Mstar', 'S2_K2')]
    results = {}
    for a, b in comparisons:
        diffs = arrs[a] - arrs[b]  # MSE_a - MSE_b
        res = paired_bootstrap(diffs, N_BOOT, SEED)
        results[f'{a}_minus_{b}'] = res
        print(f'[bootstrap] {a} - {b}: mean_diff={res["mean_diff"]:.6f} '
             f'CI=[{res["ci_lo_2.5"]:.6f}, {res["ci_hi_97.5"]:.6f}] excludes_zero={res["ci_excludes_zero"]}')

    out = {'n_boot': N_BOOT, 'seed': SEED, 'resampling_unit': 'query_start_idx (paired across arms)',
          'n_common_queries': len(common_ids), 'per_arm_mean_mse': {a: float(arrs[a].mean()) for a in ARMS},
          'comparisons': results}
    (OUT_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))
    print(f'[bootstrap] wrote {OUT_DIR / "bootstrap.json"}')


if __name__ == '__main__':
    main()
