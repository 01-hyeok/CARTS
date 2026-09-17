"""TRACK-A-SET-LOSS-CONTROL02 -- shared RNG-control utilities.

Separates three RNG concerns that a single `torch.manual_seed(seed)` call
right before `DataLoader` construction conflates:

    init_seed    model (encoder/SetConditioner) weight initialization, and
                 CUDA RNG generally
    loader_seed  DataLoader shuffle order, via an EXPLICIT torch.Generator
                 object passed to DataLoader(generator=...) -- decoupled
                 from whatever else has consumed the global RNG stream
                 between model construction and loader creation.

Not a new concept, just made explicit and independently reproducible:
previously, `torch.manual_seed(seed)` was called once before both model
init and (implicitly, via the global generator) DataLoader shuffling, so
the loader's actual shuffle order silently depended on how many random
draws model init happened to consume -- fragile, and impossible to audit
after the fact. Passing an explicit generator removes that coupling.
"""
import hashlib
import random

import numpy as np
import torch


def set_global_seeds(seed):
    """Seeds Python's `random`, NumPy, and PyTorch CPU/CUDA RNGs. Intended
    to be called once, right before constructing anything whose
    initialization should be controlled by `seed` (e.g. model weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader_generator(loader_seed):
    """A fresh, independently-seeded torch.Generator for DataLoader(shuffle
    =True, generator=...). Independent of the global RNG state -- callers
    should create ONE of these per DataLoader and never share it across
    loaders/arms/processes (each arm is its own process here, so this is
    naturally per-arm; the seed value, not the object, is what must match
    across arms)."""
    g = torch.Generator()
    g.manual_seed(loader_seed)
    return g


def batch_order_sha256(batch_start_idx_per_batch):
    """SHA256 over the exact sequence of `batch_start_idx` tensors observed
    across one epoch (or one full run), in encounter order. Order-sensitive
    and value-sensitive: any difference in which indices land in which
    batch, or in what order batches are yielded, changes the hash."""
    h = hashlib.sha256()
    for t in batch_start_idx_per_batch:
        arr = t.detach().cpu().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
        h.update(arr.tobytes())
    return h.hexdigest()
