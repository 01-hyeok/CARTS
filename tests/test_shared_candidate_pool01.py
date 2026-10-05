"""Tests for one-build/many-consumer shared candidate-pool support."""
import json

import pytest
import torch

from scripts.shared_candidate_pool01 import SharedCandidatePool, TOP100_SIZE, tensor_sha256


def _write_cache(tmp_path, n_candidates=140, n_channels=3):
    starts = torch.tensor([10, 20, 30], dtype=torch.long)
    idx = torch.empty(3, n_channels, TOP100_SIZE, dtype=torch.long)
    for q in range(3):
        for c in range(n_channels):
            idx[q, c] = torch.arange(TOP100_SIZE) + ((q + c) % 5)
    fps = {}
    for split in ("train", "val", "test"):
        payload = {
            "query_start_idx": starts,
            "candidate_indices": idx.clone(),
            "split": split,
            "pool_size": TOP100_SIZE,
        }
        torch.save(payload, tmp_path / f"{split}.pt")
        fps[split] = tensor_sha256(payload["candidate_indices"])
    meta = {
        "mode": "top100",
        "pool_size": TOP100_SIZE,
        "source": "unit_test",
        "reference_model_state_sha256": "abc",
        "n_candidates": n_candidates,
        "channels": list(range(n_channels)),
        "split_fingerprints": fps,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return starts, idx


def test_full_mode_is_identity():
    base = torch.ones(2, 140, dtype=torch.bool)
    pool = SharedCandidatePool(mode="full", n_candidates=140, channels=[0, 1], top_k=10)
    assert pool.apply(base, torch.tensor([1, 2]), 0, "train") is base


def test_top100_same_cache_can_be_reused_by_multiple_consumers(tmp_path):
    starts, idx = _write_cache(tmp_path)
    a = SharedCandidatePool("top100", tmp_path, 140, [0, 1, 2], top_k=10)
    b = SharedCandidatePool("top100", tmp_path, 140, [0, 1, 2], top_k=10)
    base = torch.ones(3, 140, dtype=torch.bool)
    ma = a.apply(base, starts, 1, "train")
    mb = b.apply(base, starts, 1, "train")
    assert torch.equal(ma, mb)
    assert torch.equal(ma.sum(dim=1), torch.full((3,), TOP100_SIZE))
    for q in range(3):
        assert set(ma[q].nonzero().flatten().tolist()) == set(idx[q, 1].tolist())


def test_top100_rejects_missing_query(tmp_path):
    _write_cache(tmp_path)
    pool = SharedCandidatePool("top100", tmp_path, 140, [0, 1, 2], top_k=10)
    with pytest.raises(KeyError):
        pool.apply(torch.ones(1, 140, dtype=torch.bool), torch.tensor([999]), 0, "train")


def test_top100_rejects_runtime_mask_mismatch(tmp_path):
    starts, idx = _write_cache(tmp_path)
    pool = SharedCandidatePool("top100", tmp_path, 140, [0, 1, 2], top_k=10)
    base = torch.ones(3, 140, dtype=torch.bool)
    base[0, int(idx[0, 0, 0])] = False
    with pytest.raises(ValueError, match="invalid under runtime mask"):
        pool.apply(base, starts, 0, "train")


def test_fingerprint_mismatch_is_detected(tmp_path):
    _write_cache(tmp_path)
    payload = torch.load(tmp_path / "val.pt")
    payload["candidate_indices"][0, 0, 0] += 1
    torch.save(payload, tmp_path / "val.pt")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        SharedCandidatePool("top100", tmp_path, 140, [0, 1, 2], top_k=10)
