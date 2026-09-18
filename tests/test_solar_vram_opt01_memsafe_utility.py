"""TRACK-A-SOLAR-VRAM-OPT01 P0.2 -- equivalence tests for the memory-safe
Oracle utility functions (`utils.dense_utility.dense_utility_memsafe`,
`scripts.train_factorial_e2e01.greedy_set_utility_memsafe`/
`individual_utility_memsafe`) against their reference (full-`futures`-
materializing) counterparts. Real ETTh1_96 data, real memory bank.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import (greedy_set_utility, greedy_set_utility_memsafe,
                                           individual_utility, individual_utility_memsafe)
from scripts.train_margutil01 import build_experiment, memory_value
from utils.dense_utility import candidate_weights

REF_CKPT = 'checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth'


def _available():
    return (REPO_ROOT / REF_CKPT).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 checkpoint not present')


def _real_data(channel=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 16, 'seed': 0, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
    futures = memory_c + offset_c.view(-1, 1, 1)
    query_future = batch_y[:, :, channel]
    device = cand_mask.device
    raw_scores = torch.randn(batch_x.size(0), memory_c.size(0), device=device)
    w_host = candidate_weights(raw_scores.masked_fill(~cand_mask, torch.finfo(torch.float32).min / 4),
                               cand_mask, 0.1)
    return memory_c, offset_c, futures, query_future, w_host, cand_mask


def test_individual_utility_memsafe_matches_reference_various_chunk_sizes():
    memory_c, offset_c, futures, query_future, w_host, cand_mask = _real_data()
    ref = individual_utility(futures, query_future)
    for chunk_size in (500, 4096, None):
        opt = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=chunk_size)
        assert torch.allclose(ref, opt, atol=1e-5, rtol=1e-5), f'chunk_size={chunk_size}'


def test_greedy_set_utility_memsafe_matches_reference_empty_prefix():
    memory_c, offset_c, futures, query_future, w_host, cand_mask = _real_data()
    bsz = futures.size(0)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long, device=futures.device)
    ref = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    for chunk_size in (500, 4096, None):
        opt = greedy_set_utility_memsafe(empty_prefix, w_host, memory_c, offset_c, query_future,
                                         chunk_size=chunk_size)
        assert torch.allclose(ref, opt, atol=1e-5, rtol=1e-5), f'chunk_size={chunk_size}'


def test_greedy_set_utility_memsafe_matches_reference_nonempty_prefix():
    memory_c, offset_c, futures, query_future, w_host, cand_mask = _real_data()
    bsz = futures.size(0)
    n_valid = int(cand_mask.sum(dim=-1).min())
    k = min(3, n_valid)
    valid_idx = cand_mask.float().multinomial(k, replacement=False)
    ref = greedy_set_utility(valid_idx, w_host, futures, query_future, chunk_size=4096)
    for chunk_size in (500, 4096, None):
        opt = greedy_set_utility_memsafe(valid_idx, w_host, memory_c, offset_c, query_future,
                                         chunk_size=chunk_size)
        assert torch.allclose(ref, opt, atol=1e-4, rtol=1e-4), f'chunk_size={chunk_size}'


def test_greedy_set_utility_memsafe_gpu_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip('no GPU available')
    memory_c, offset_c, futures, query_future, w_host, cand_mask = _real_data()
    device = torch.device('cuda')
    bsz = futures.size(0)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long, device=device)
    ref = greedy_set_utility(empty_prefix, w_host.to(device), futures.to(device),
                             query_future.to(device), chunk_size=4096)
    opt = greedy_set_utility_memsafe(empty_prefix, w_host.to(device), memory_c.to(device),
                                     offset_c.to(device), query_future.to(device), chunk_size=4096)
    assert torch.allclose(ref.cpu(), opt.cpu(), atol=1e-5, rtol=1e-5)


def test_greedy_set_utility_memsafe_small_synthetic_deterministic():
    torch.manual_seed(0)
    bsz, n_cand, pred_len = 4, 20, 6
    memory_c = torch.randn(n_cand, pred_len)
    offset_c = torch.randn(bsz)
    futures = memory_c + offset_c.view(-1, 1, 1)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    prefix = torch.tensor([[2, 5], [1, 3], [0, 7], [4, 6]])
    ref = greedy_set_utility(prefix, w_host, futures, query_future, chunk_size=7)
    opt = greedy_set_utility_memsafe(prefix, w_host, memory_c, offset_c, query_future, chunk_size=7)
    assert torch.allclose(ref, opt, atol=1e-6, rtol=1e-6)
