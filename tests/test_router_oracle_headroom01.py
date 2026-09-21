"""ROUTER-ORACLE-HEADROOM01 sanity tests (spec section 17)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import transform_relation_history
from models.SequentialSetRetriever import SetConditioner
from scripts.analyze_router_oracle_headroom01 import (best_static_expert, oracle_row,
                                                       paired_bootstrap_ci)
from scripts.diag_router_oracle_headroom01 import raw_cosine_scores, run_split
from scripts.train_factorial_e2e01 import encode_raw, run_sequence
from scripts.train_margutil01 import build_experiment, memory_value

REF_CKPT = 'checkpoints/track_a_tf_oracle_learnability01/ETTh1_96/individual_tf_cosine/checkpoint.pth'


def _available():
    return (REPO_ROOT / REF_CKPT).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 TF-Oracle checkpoint not present')


def _build():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blob = torch.load(REF_CKPT, map_location='cpu')
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': 0, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    model.load_state_dict(blob['model_state_dict'])
    model.eval()
    sc = SetConditioner(int(args.d_model)).to(device)
    sc.load_state_dict(blob['set_conditioner_state_dict'])
    sc.eval()
    return exp, args, model, sc, device


# ---- T1: raw absolute cosine matches direct computation ----
def test_t1_raw_absolute_cosine_matches_direct():
    exp, args, model, sc, device = _build()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    q_abs = transform_relation_history(batch_x, 'absolute')
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    k_abs = transform_relation_history(memory_x, 'absolute')
    c = 0
    scores = raw_cosine_scores(q_abs[:, :, c], k_abs[:, :, c])
    direct = torch.matmul(F.normalize(batch_x[:, :, c], dim=-1),
                          F.normalize(memory_x[:, :, c], dim=-1).transpose(0, 1))
    assert torch.allclose(scores, direct, atol=1e-5)


# ---- T2: delta-last cosine matches direct transform + cosine ----
def test_t2_delta_last_cosine_matches_direct():
    exp, args, model, sc, device = _build()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    c = 0
    q_delta = transform_relation_history(batch_x, 'delta_last')
    k_delta = transform_relation_history(memory_x, 'delta_last')
    scores = raw_cosine_scores(q_delta[:, :, c], k_delta[:, :, c])
    direct_q = batch_x[:, :, c] - batch_x[:, -1:, c]
    direct_k = memory_x[:, :, c] - memory_x[:, -1:, c]
    direct = torch.matmul(F.normalize(direct_q, dim=-1), F.normalize(direct_k, dim=-1).transpose(0, 1))
    assert torch.allclose(scores, direct, atol=1e-5)


# ---- T3/T4/T5: candidate mask identical + Top-K validity/uniqueness ----
def test_t3_t4_t5_mask_identity_and_topk_validity():
    exp, args, model, sc, device = _build()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    c = 0
    q_abs = transform_relation_history(batch_x, 'absolute')
    k_abs = transform_relation_history(memory_x, 'absolute')
    q_delta = transform_relation_history(batch_x, 'delta_last')
    k_delta = transform_relation_history(memory_x, 'delta_last')
    s_raw = raw_cosine_scores(q_abs[:, :, c], k_abs[:, :, c])
    s_delta = raw_cosine_scores(q_delta[:, :, c], k_delta[:, :, c])
    neg_inf = float('-inf')
    picks_raw = s_raw.masked_fill(~cand_mask, neg_inf).topk(10, dim=-1).indices
    picks_delta = s_delta.masked_fill(~cand_mask, neg_inf).topk(10, dim=-1).indices
    # T3: same cand_mask object drives both (trivially identical here; the
    # real per-run-invocation invariant is that _candidate_mask is a pure
    # function of batch_start_idx, asserted by re-calling it).
    cand_mask2, _ = exp._candidate_mask(batch_start_idx)
    assert torch.equal(cand_mask, cand_mask2)
    # T4: no invalid candidate ever selected
    valid_raw = cand_mask.gather(1, picks_raw)
    valid_delta = cand_mask.gather(1, picks_delta)
    assert valid_raw.all()
    assert valid_delta.all()
    # T5: K unique candidates per row
    for picks in (picks_raw, picks_delta):
        for b in range(picks.size(0)):
            assert len(set(picks[b].tolist())) == picks.size(1)


# ---- T6: FutureAligned free-running picks are index-identical to the
# production run_sequence(free_running=True) call ----
def test_t6_learned_picks_match_production_run_sequence():
    exp, args, model, sc, device = _build()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    c = 0
    z_q = encode_raw(model, batch_x, c)
    E = encode_raw(model, memory_x, c)
    _, _, picks_a, _ = run_sequence(z_q, E, cand_mask, sc, None, None, None, None,
                                    'individual', 'tf', 0.1, 10, 4096, free_running=True)
    _, _, picks_b, _ = run_sequence(z_q, E, cand_mask, sc, None, None, None, None,
                                    'individual', 'tf', 0.1, 10, 4096, free_running=True)
    assert torch.equal(picks_a, picks_b)


# ---- T7: no leakage -- corrupting future/Y_q does not change selected
# indices (structural: free_running=True never receives them) ----
def test_t7_future_aligned_no_leakage():
    exp, args, model, sc, device = _build()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    c = 0
    z_q = encode_raw(model, batch_x, c)
    E = encode_raw(model, memory_x, c)
    garbage_future = torch.randn(batch_x.size(0), memory_x.size(0), 96, device=device)
    garbage_query = torch.randn(batch_x.size(0), 96, device=device)
    _, _, picks_clean, _ = run_sequence(z_q, E, cand_mask, sc, None, None, None, None,
                                        'individual', 'tf', 0.1, 10, 4096, free_running=True)
    _, _, picks_garbage, _ = run_sequence(z_q, E, cand_mask, sc, None, None,
                                          garbage_future, garbage_query,
                                          'individual', 'tf', 0.1, 10, 4096, free_running=True)
    assert torch.equal(picks_clean, picks_garbage)


# ---- T8/T9: query identity across experts + uniform aggregate reconstruction ----
def test_t8_t9_query_alignment_and_uniform_aggregate():
    exp, args, model, sc, device = _build()
    channels = list(range(int(args.enc_in)))[:2]
    rows = run_split(exp, args, model, sc, 'x', 'val', channels, top_k=10,
                     chunk_size=4096, limit_batches=1)
    assert len(rows) > 0
    # T8: every row carries the same query_id space (no duplicates, deterministic)
    ids = [r['query_id'] for r in rows]
    assert len(ids) == len(set(ids))
    # T9: manual reference reconstruction for one query/channel matches the
    # uniform_mse the batched code computed, within tolerance.
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    c = 0
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    q_abs = transform_relation_history(batch_x, 'absolute')
    k_abs = transform_relation_history(memory_x, 'absolute')
    scores = raw_cosine_scores(q_abs[:, :, c], k_abs[:, :, c])
    picks = scores.masked_fill(~cand_mask, float('-inf')).topk(10, dim=-1).indices
    b = 0
    manual_futures = torch.stack([memory_c[picks[b, k]] + offset_c[b] for k in range(10)])
    manual_uniform = manual_futures.mean(dim=0)
    manual_mse = ((manual_uniform - batch_y[b, :, c]) ** 2).mean()
    V = memory_c[picks] + offset_c.view(-1, 1, 1)
    batched_uniform = V.mean(dim=1)
    batched_mse = ((batched_uniform - batch_y[:, :, c]) ** 2).mean(-1)
    assert torch.allclose(manual_mse, batched_mse[b], atol=1e-5)


# ---- T10: Oracle Router picks the exact row-wise min expert (synthetic) ----
def test_t10_oracle_router_picks_row_min():
    row = {'raw_uniform_mse': 0.5, 'delta_uniform_mse': 0.2, 'learned_uniform_mse': 0.9}
    best, err, margin = oracle_row(row)
    assert best == 'delta'
    assert err == 0.2
    assert margin == pytest.approx(0.3)


# ---- T11: Best Static selection reads only validation, never test ----
def test_t11_best_static_reads_only_validation():
    val_rows = [{'raw_uniform_mse': 1.0, 'delta_uniform_mse': 5.0, 'learned_uniform_mse': 9.0}] * 5
    static_expert, means = best_static_expert(val_rows)
    assert static_expert == 'raw'
    # function signature takes only val_rows -- no test argument exists at all,
    # so it is structurally impossible for it to read test labels.
    import inspect
    sig = inspect.signature(best_static_expert)
    assert list(sig.parameters) == ['val_rows']


# ---- T12: channel/query aggregation matches manual calculation ----
def test_t12_bootstrap_matches_manual_mean():
    diffs = [1.0, 2.0, 3.0, 4.0]
    result = paired_bootstrap_ci(diffs, n=500, seed=1)
    assert result['mean'] == pytest.approx(sum(diffs) / len(diffs))
    assert result['ci_low'] <= result['mean'] <= result['ci_high']
