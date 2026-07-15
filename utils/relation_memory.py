import numpy as np
import torch
from torch.utils.data import Dataset


class Stage1WindowDataset(Dataset):
    """Return batch_x, future-only batch_y, and a start index for Stage-1."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.seq_len = base_dataset.seq_len
        self.pred_len = base_dataset.pred_len
        self.label_len = base_dataset.label_len
        self.flag = getattr(base_dataset, 'flag', None)
        self.teacher_mse_space = getattr(base_dataset.args, 'teacher_mse_space', 'normalized')
        self.data_x = base_dataset.data_x
        self.data_y = base_dataset.data_y
        self.data = self.data_x
        self.channel_names = getattr(base_dataset, 'channel_names', None)
        self.starts = np.arange(len(base_dataset), dtype=np.int64) + int(getattr(base_dataset, 'border1', 0))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        _, seq_x, seq_y, _, _ = item
        future = seq_y[-self.pred_len:]
        if self.teacher_mse_space == 'raw' and getattr(self.base_dataset, 'scale', False):
            future = self.base_dataset.inverse_transform(future)
        return seq_x, future, np.int64(self.starts[index])

    def get_all_valid_starts(self):
        return self.starts.copy()

    def get_window_by_start(self, start_idx):
        row = int(np.searchsorted(self.starts, int(start_idx)))
        if row < 0 or row >= len(self.starts) or int(self.starts[row]) != int(start_idx):
            raise IndexError(f'start_idx={start_idx} is not in this split')
        return self[row]


class RelationMemorySampler:
    def __init__(self, train_dataset, seq_len, pred_len, mask_mode='raft'):
        if mask_mode not in ('raft', 'strict_causal', 'overlap_only', 'none'):
            raise ValueError(f'Unsupported candidate_mask: {mask_mode}')
        self.dataset = train_dataset
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mask_mode = mask_mode
        self.starts = train_dataset.get_all_valid_starts().astype(np.int64)

        data_x = np.asarray(train_dataset.data_x, dtype=np.float32)
        data_y = np.asarray(train_dataset.data_y, dtype=np.float32)
        num_windows = len(train_dataset)
        memory_x = np.lib.stride_tricks.sliding_window_view(
            data_x, self.seq_len, axis=0
        ).transpose(0, 2, 1)
        memory_y = np.lib.stride_tricks.sliding_window_view(
            data_y[self.seq_len:], self.pred_len, axis=0
        ).transpose(0, 2, 1)
        self.memory_x = memory_x[:num_windows]
        self.memory_y = memory_y[:num_windows]

    def valid_indices(self, query_start):
        query_start = int(query_start)
        candidate_end = self.starts + self.seq_len + self.pred_len
        if self.mask_mode == 'raft':
            if np.any(self.starts == query_start):
                window = 2 * (self.seq_len + self.pred_len) - 1
                mask_starts = (
                    np.arange(window, dtype=np.int64)
                    + query_start
                    - self.seq_len
                    - self.pred_len
                    + 1
                )
                mask_starts = np.clip(mask_starts, self.starts[0], self.starts[-1])
                valid = ~np.isin(self.starts, mask_starts)
            else:
                valid = np.ones_like(self.starts, dtype=bool)
        elif self.mask_mode == 'strict_causal':
            valid = candidate_end <= query_start
        elif self.mask_mode == 'overlap_only':
            query_end = query_start + self.seq_len + self.pred_len
            valid = (self.starts < query_end) & (candidate_end > query_start)
            valid &= self.starts != query_start
        else:
            valid = self.starts != query_start
        return np.flatnonzero(valid)

    def valid_mask_batch(self, query_starts):
        query_starts = np.asarray(query_starts, dtype=np.int64)
        bsz = len(query_starts)
        mask = np.zeros((bsz, len(self.starts)), dtype=bool)
        counts = np.zeros((bsz,), dtype=np.int64)

        for row, start in enumerate(query_starts):
            valid = self.valid_indices(start)
            counts[row] = len(valid)
            mask[row, valid] = True

        return (
            torch.from_numpy(mask),
            torch.from_numpy(counts),
        )


class RelationMemoryBank:
    """Train-split memory used by Stage-2 retrieval."""

    def __init__(self, train_dataset, seq_len, pred_len, mask_mode='raft'):
        self.sampler = RelationMemorySampler(
            train_dataset,
            seq_len=seq_len,
            pred_len=pred_len,
            mask_mode=mask_mode,
        )
        self.memory_x = self.sampler.memory_x
        self.memory_y = self.sampler.memory_y
        self.memory_starts = self.sampler.starts
        self.key_bank = None

    def valid_mask_batch(self, query_starts):
        return self.sampler.valid_mask_batch(query_starts)

    def to_tensors(self, device):
        return {
            'memory_x': torch.from_numpy(self.memory_x).float().to(device),
            'memory_y': torch.from_numpy(self.memory_y).float().to(device),
            'memory_starts': torch.from_numpy(self.memory_starts).long().to(device),
            'key_bank': None if self.key_bank is None else self.key_bank.to(device),
        }


def build_memory_index(model, train_dataset, args):
    """Stage-2 TODO: materialize relation embeddings/value cache for retrieval."""
    raise NotImplementedError('Stage-2 memory index building is intentionally left as a TODO stub.')
