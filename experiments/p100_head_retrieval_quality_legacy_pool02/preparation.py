"""CPU preparation helpers. No trainer, GPU dispatch, or default teacher."""
from contextlib import contextmanager
import hashlib
import math
import random
from pathlib import Path
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader

TEACHER_MODES = ('zscore_future_mse', 'raw_future_mse')
GPU_UUID = 'GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def state_hash(state):
    if hasattr(state, 'state_dict'):
        state = state.state_dict()
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@contextmanager
def isolated_cpu_rng(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

def initialize_common_encoder(factory, seed):
    with isolated_cpu_rng(seed):
        model = factory()
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Common initialization must be constructed on CPU')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, state, state_hash(state)

def seeded_slot_heads(d_model, n_heads, seed, std=1e-3):
    # Reuse canonical architecture/forward. Only the explicit initialization
    # seed policy differs; seed0 is exactly the historical 1000+h policy.
    from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
    if n_heads not in (1, 2, 5):
        raise ValueError('Additional heads must be 1, 2, or 5')
    if seed not in (0, 1, 2):
        raise ValueError('Only the authorized seeds are supported')
    heads = SlotHeads(d_model, n_slots=n_heads, std=std)
    with torch.no_grad():
        for h in range(n_heads):
            gen = torch.Generator(device='cpu').manual_seed(1000 + 10000 * seed + h)
            heads.W[h].copy_(torch.eye(d_model) + std * torch.randn(d_model, d_model, generator=gen))
    return heads

def make_training_loader(dataset, batch_size, seed):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=False, generator=gen)

def batch_order_hash(batches):
    # Include boundaries, unlike a plain concatenated-ID hash.
    h = hashlib.sha256()
    for batch in batches:
        arr = np.asarray(batch, dtype=np.int64).reshape(-1)
        h.update(struct.pack('<Q', len(arr)))
        h.update(arr.astype('<i8', copy=False).tobytes())
    return h.hexdigest()

class WeightedMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add_values(self, values):
        x = np.asarray(values, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite evaluation values')
        self.total += float(x.sum())
        self.count += x.size

    def add_batch_mean(self, mean, n_observations):
        if n_observations <= 0 or not math.isfinite(float(mean)):
            raise ValueError('Invalid batch mean/count')
        self.total += float(mean) * n_observations
        self.count += n_observations

    @property
    def mean(self):
        if self.count == 0:
            raise ValueError('No observations')
        return self.total / self.count

def aligned_futures(candidate_future, candidate_last, query_last):
    return candidate_future - candidate_last[..., None] + query_last[:, None, None]

def future_distances(candidate_future, candidate_last, query_last, target):
    from utils.candidate_pool import pooled_future_mse
    return pooled_future_mse(candidate_future - candidate_last[..., None], query_last, target)

def oracle_ids(distances, k):
    from models.RelationStage1 import stable_topk_indices
    return stable_topk_indices(distances, k, largest=False)

def mixture_ids(scores, mask, tau_s, k=10):
    from utils.mean_mixture_selection import mean_mixture_topk_selection
    return mean_mixture_topk_selection(scores, mask, tau_s, k)

def retrieval_metrics(candidate_future, candidate_last, query_last, target, ids):
    d = future_distances(candidate_future, candidate_last, query_last, target)
    aligned = aligned_futures(candidate_future, candidate_last, query_last)
    if ids.ndim != 2 or ids.shape[0] != len(target):
        raise ValueError('Bad candidate selection shape')
    if ids.min() < 0 or ids.max() >= aligned.shape[1]:
        raise ValueError('Candidate ID out of range')
    if not (ids.sort(-1).values.diff(dim=-1) > 0).all():
        raise ValueError('Duplicate candidate ID')
    picked = aligned.gather(1, ids[..., None].expand(-1, -1, target.shape[-1]))
    error = picked.mean(1) - target
    oracle = oracle_ids(d, ids.shape[1])
    ret = d.gather(1, ids).mean(-1)
    oracle_ret = d.gather(1, oracle).mean(-1)
    return dict(retmse=ret, agg_mse=error.square().mean(-1),
                agg_mae=error.abs().mean(-1),
                recall=(ids[..., None] == oracle[:, None, :]).any(-1).float().mean(-1),
                regret=ret - oracle_ret, oracle_retmse=oracle_ret)

def last_repeat_metrics(query_history, target):
    if query_history.ndim != 2 or target.ndim != 2:
        raise ValueError('Flatten query-channel axes before calling')
    prediction = query_history[:, -1:].expand_as(target)
    error = prediction - target
    return dict(prediction=prediction, mse=error.square().mean(-1), mae=error.abs().mean(-1))

def last_skill(model_mse, last_mse, eps=1e-12):
    a, b = np.asarray(model_mse, dtype=np.float64), np.asarray(last_mse, dtype=np.float64)
    if a.shape != b.shape or a.size == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Skill requires the same finite evaluation units')
    denominator = float(b.mean())
    return None if denominator <= eps else 1.0 - float(a.mean()) / denominator

def random_topk(n_queries, pool_size, k, draw_seed):
    if not 0 < k <= pool_size:
        raise ValueError('Bad K')
    gen = torch.Generator(device='cpu').manual_seed(draw_seed)
    return torch.stack([torch.randperm(pool_size, generator=gen)[:k] for _ in range(n_queries)])

def validate_pool(ids, query_starts, candidate_starts, seq_len, pred_len, split, pool_size=100):
    ids = np.asarray(ids)
    q = np.asarray(query_starts, dtype=np.int64)
    starts = np.asarray(candidate_starts, dtype=np.int64)
    if ids.ndim != 3 or ids.shape[0] != len(q) or ids.shape[-1] != pool_size:
        raise ValueError('Pool shape mismatch')
    if not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() >= len(starts):
        raise ValueError('Invalid global candidate ID')
    if not (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all():
        raise ValueError('Duplicate pool ID')
    c = starts[ids]
    end = c + seq_len + pred_len
    q = q[:, None, None]
    if split == 'train':
        allowed = (end <= q) | (c >= q + seq_len + pred_len)
    elif split in ('val', 'test'):
        allowed = end <= q + seq_len  # candidate future is known at forecast origin
    else:
        raise ValueError('Unknown split')
    if not allowed.all():
        raise ValueError('Temporal eligibility violation')
    return True

def require_teacher_resolution(config):
    teacher = config['teacher']
    if teacher.get('status') != 'USER_CONFIRMED' or teacher.get('mode') not in TEACHER_MODES:
        raise RuntimeError('BLOCKED: teacher is unresolved; no training may start')
    if not teacher.get('decision_reference'):
        raise RuntimeError('BLOCKED: missing explicit teacher decision reference')
    return teacher['mode']
