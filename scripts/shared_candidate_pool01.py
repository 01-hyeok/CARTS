#!/usr/bin/env python3
"""Shared candidate support for TRACK-V and downstream routers.

Two modes are intentionally supported:
- full: use the experiment's native candidate mask unchanged.
- top100: intersect the native mask with one precomputed, query/channel-specific
  Top-100 pool. The pool is built once from a frozen reference retriever and
  reused by every V arm and any router so candidate support is identical.

The cache stores only candidate indices. Each consumer still recomputes its own
scores; only the support is shared.
"""
import hashlib
import json
from pathlib import Path

import torch

VALID_POOL_MODES = ("full", "top100")
TOP100_SIZE = 100


def tensor_sha256(tensor):
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


class SharedCandidatePool:
    def __init__(self, mode="full", cache_dir=None, n_candidates=None, channels=None, top_k=10):
        if mode not in VALID_POOL_MODES:
            raise ValueError(f"candidate_pool_mode must be one of {VALID_POOL_MODES}, got {mode!r}")
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n_candidates = int(n_candidates) if n_candidates is not None else None
        self.channels = list(channels) if channels is not None else None
        self.top_k = int(top_k)
        self.metadata = {"mode": mode}
        self._split_cache = {}
        self._lut = {}
        self._channel_to_col = {}

        if self.mode == "full":
            return
        if self.cache_dir is None:
            raise ValueError("top100 mode requires --candidate_pool_cache_dir")

        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"shared candidate-pool metadata not found: {meta_path}")
        self.metadata = json.loads(meta_path.read_text())
        if self.metadata.get("mode") != "top100":
            raise ValueError(f"cache mode mismatch: expected top100, got {self.metadata.get('mode')!r}")
        if int(self.metadata.get("pool_size", -1)) != TOP100_SIZE:
            raise ValueError(f"top100 cache must contain exactly {TOP100_SIZE} candidates")
        if self.top_k > TOP100_SIZE:
            raise ValueError(f"top_k={self.top_k} exceeds shared Top-{TOP100_SIZE} support")

        cached_n = int(self.metadata["n_candidates"])
        if self.n_candidates is not None and cached_n != self.n_candidates:
            raise ValueError(f"candidate-bank mismatch: cache={cached_n}, runtime={self.n_candidates}")
        cached_channels = [int(c) for c in self.metadata["channels"]]
        if self.channels is not None and cached_channels != [int(c) for c in self.channels]:
            raise ValueError(f"channel mismatch: cache={cached_channels}, runtime={self.channels}")
        self.n_candidates = cached_n
        self.channels = cached_channels
        self._channel_to_col = {c: i for i, c in enumerate(cached_channels)}

        for split in ("train", "val", "test"):
            path = self.cache_dir / f"{split}.pt"
            if not path.exists():
                raise FileNotFoundError(f"shared candidate-pool split cache not found: {path}")
            payload = torch.load(path, map_location="cpu")
            starts = payload["query_start_idx"].long().contiguous()
            indices = payload["candidate_indices"].long().contiguous()
            if starts.ndim != 1:
                raise ValueError(f"{split}: query_start_idx must be 1-D")
            if indices.ndim != 3:
                raise ValueError(f"{split}: candidate_indices must be [Q,C,100]")
            if indices.shape != (starts.numel(), len(cached_channels), TOP100_SIZE):
                raise ValueError(
                    f"{split}: candidate_indices shape {tuple(indices.shape)} does not match "
                    f"({starts.numel()}, {len(cached_channels)}, {TOP100_SIZE})"
                )
            if starts.unique().numel() != starts.numel():
                raise ValueError(f"{split}: query_start_idx contains duplicates")
            expected = self.metadata.get("split_fingerprints", {}).get(split)
            actual = tensor_sha256(indices)
            if expected and expected != actual:
                raise ValueError(f"{split}: candidate-pool fingerprint mismatch")
            self._split_cache[split] = {"query_start_idx": starts, "candidate_indices": indices}
            self._lut[split] = {int(s): i for i, s in enumerate(starts.tolist())}

    def apply(self, base_mask, query_start_idx, channel, split):
        """Return the effective candidate mask for one channel.

        In full mode this is the original object unchanged. In top100 mode the
        cached support is intersected with the runtime native mask, and cache
        validity is asserted instead of silently accepting stale/foreign pools.
        """
        if self.mode == "full":
            return base_mask
        if split not in self._split_cache:
            raise ValueError(f"unknown split {split!r}")
        channel = int(channel)
        if channel not in self._channel_to_col:
            raise ValueError(f"channel {channel} not present in shared candidate pool")
        starts = query_start_idx.tolist() if torch.is_tensor(query_start_idx) else list(query_start_idx)
        lut = self._lut[split]
        try:
            rows = [lut[int(s)] for s in starts]
        except KeyError as exc:
            raise KeyError(f"{split}: query_start_idx {exc.args[0]} missing from shared candidate pool") from exc

        col = self._channel_to_col[channel]
        idx = self._split_cache[split]["candidate_indices"][rows, col, :].to(base_mask.device)
        if int(idx.min()) < 0 or int(idx.max()) >= int(base_mask.size(1)):
            raise ValueError(f"{split}/channel{channel}: cached candidate index out of range")

        valid_cached = base_mask.gather(1, idx)
        if not bool(valid_cached.all()):
            bad = int((~valid_cached).sum().item())
            raise ValueError(
                f"{split}/channel{channel}: shared pool contains {bad} candidates invalid under runtime mask"
            )

        pool_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        pool_mask.scatter_(1, idx, True)
        effective = base_mask & pool_mask
        counts = effective.sum(dim=1)
        if bool((counts < self.top_k).any()):
            raise ValueError(
                f"{split}/channel{channel}: effective pool smaller than top_k={self.top_k}; "
                f"min={int(counts.min())}"
            )
        return effective

    def describe(self):
        if self.mode == "full":
            return {"candidate_pool_mode": "full", "candidate_pool_size": self.n_candidates}
        return {
            "candidate_pool_mode": "top100",
            "candidate_pool_size": TOP100_SIZE,
            "candidate_pool_cache_dir": str(self.cache_dir),
            "candidate_pool_source": self.metadata.get("source"),
            "candidate_pool_score_definition": self.metadata.get("score_definition"),
            "candidate_pool_reference_ckpt_for_data_config": self.metadata.get("reference_ckpt_for_data_config"),
            "candidate_pool_split_fingerprints": self.metadata.get("split_fingerprints", {}),
        }
