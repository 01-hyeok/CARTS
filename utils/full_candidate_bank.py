"""TRACK-V-R100-EFFICIENCY01 -- shared infra for the two approved
efficiency changes (channel-first preprocessing; periodic full-candidate
embedding bank refresh). Candidate support is ALWAYS the full memory
bank (N candidates) -- nothing here subsets/prunes/prefilters
candidates. This is a pure implementation-optimization layer, not a
new retrieval architecture.

`encode_raw_channel_first` is a drop-in replacement for
`train_factorial_e2e01.encode_raw(model, x, c)` on the SELF-RETRIEVAL
path only (target_channel == source_channel == c, which is the only
path V0/V1/V2/V5 ever use): `build_relation_encoder_input` returns
early with `target = stack([view[..., c] for view in features], dim=1)`
for self relations, where `features = transform_relation_features(x,
relation_input_space)` is computed over ALL C channels of `x` and only
channel `c` is kept. Every entry in `RELATION_INPUT_FEATURES`
(absolute / delta_last / diff1) is a per-channel elementwise transform
with no cross-channel mixing (`_transform_relation_feature` only ever
indexes/subtracts/diffs along the last-but-one (length) axis, never
touching the channel axis) -- so selecting channel `c` BEFORE the
transform instead of after produces bit-identical values, at O(L) work
per channel instead of O(L*C). Proven in `tests/test_full_candidate_bank01.py`
and in the age=0 parity tests against the real legacy path.
"""
import torch

from models.RelationStage1 import transform_relation_features


def encode_raw_channel_first(model, x, c):
    """Channel-first self-relation encode. `x`: [B, L, C]. Mathematically
    identical to `train_factorial_e2e01.encode_raw(model, x, c)` for the
    self-retrieval path; never valid for target_channel != source_channel
    (cross-relation), which this function does not implement."""
    x_c = x[..., c:c + 1]  # [B, L, 1] -- select channel BEFORE transform
    features = transform_relation_features(x_c, model.relation_input_space)
    target = torch.stack([view[..., 0] for view in features], dim=1)  # [B, F, L']
    return model.encoder(target)


class FullCandidateBank:
    """Per-channel cached FULL-N candidate embedding bank (raw encoder
    output, un-normalized -- callers normalize exactly where the legacy
    code did, so this is a pure drop-in for the candidate-side
    `encode_raw(...)` call, nothing else changes). Candidate support is
    always every one of the N candidates; this class only controls WHEN
    the embeddings are recomputed, never WHICH ones are considered.

    Built under `torch.no_grad()` so every cached tensor already has
    `requires_grad=False` (asserted on every refresh, spec test 3)."""

    def __init__(self, channels):
        self.channels = list(channels)
        self.bank = {}
        self.age = 0
        self.n_refreshes = 0
        self.n_candidate_encoder_calls = 0  # one increment per (refresh x channel)
        self.n_candidate_sequences_encoded = 0  # += N per (refresh x channel)

    @torch.no_grad()
    def refresh(self, model, memory_x):
        was_training = model.training
        model.eval()
        n = memory_x.size(0)
        for c in self.channels:
            emb = encode_raw_channel_first(model, memory_x, c)
            assert not emb.requires_grad, '[ISSUE][ABORT] cached candidate bank must have requires_grad=False'
            self.bank[c] = emb
            self.n_candidate_encoder_calls += 1
            self.n_candidate_sequences_encoded += n
        if was_training:
            model.train()
        self.age = 0
        self.n_refreshes += 1

    def get(self, c):
        return self.bank[c]

    def tick(self):
        self.age += 1

    def counters(self):
        return {
            'n_refreshes': self.n_refreshes,
            'n_candidate_encoder_calls': self.n_candidate_encoder_calls,
            'n_candidate_sequences_encoded': self.n_candidate_sequences_encoded,
        }
