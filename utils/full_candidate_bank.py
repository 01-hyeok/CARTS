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
import torch.nn.functional as F

from models.RelationStage1 import transform_relation_features
from utils.candidate_pool import encode_pooled_candidates, gather_candidate_histories


def encode_raw_channel_first(model, x, c):
    """Channel-first self-relation encode. `x`: [B, L, C]. Mathematically
    identical to `train_factorial_e2e01.encode_raw(model, x, c)` for the
    self-retrieval path; never valid for target_channel != source_channel
    (cross-relation), which this function does not implement."""
    x_c = x[..., c:c + 1]  # [B, L, 1] -- select channel BEFORE transform
    features = transform_relation_features(x_c, model.relation_input_space)
    target = torch.stack([view[..., 0] for view in features], dim=1)  # [B, F, L']
    return model.encoder(target)


def compute_scores_full_grad_channel_first(model, slot_heads, batch_x, memory_x, c):
    """A1 (channel-first) drop-in for `train_t_pure_multislot01.
    compute_scores_full_grad` -- bit-exact equivalent (proven by
    `tests/test_v_meanmix_checkpoint_correction01.py` and this module's
    own age=0 parity tests), candidate-side gradient stays ON (no
    `.detach()`), full-N candidate support unchanged. The ONLY
    difference from the legacy function is encode order. Shared by every
    new V0-V5-family trainer per the standing A1-default policy."""
    z_q = encode_raw_channel_first(model, batch_x, c)
    q = slot_heads(z_q)
    k_full = F.normalize(encode_raw_channel_first(model, memory_x, c), dim=-1)
    return torch.einsum('bsd,nd->bsn', q, k_full)


def compute_scores_pool_channel_first_grad(model, slot_heads, batch_x, exp, channel, pool_cache, batch_start_idx,
                                           device):
    """TRACK-HARD-EXPERT-V5-P100-ALLH01 -- A1 (channel-first) Shared-Top-100
    (P100) scorer, GRADIENT-ENABLED (no `torch.no_grad()` anywhere in
    this function body -- callers that want an eval-only pass wrap the
    call site themselves, exactly as `scripts/train_v_sharedtop100_01.py`
    already does for its own `compute_scores`). Bit-identical math to
    the `@torch.no_grad()`-wrapped `compute_scores_pool_channel_first`
    helper duplicated in `scripts/eval_v5_meanmix_checkpoint_selection_pool01.py`
    and `scripts/build_v_meanmix_cache_pool01.py` -- this is the single
    shared, differentiable definition those eval-only call sites could
    have imported from instead of inlining their own no-grad copy.
    Returns (scores [B,S,M], pool_valid_mask [B,M] all-True, pool_idx_global [B,M])."""
    z_q = encode_raw_channel_first(model, batch_x, channel)
    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
    z_k = encode_pooled_candidates(lambda x, c: encode_raw_channel_first(model, x, c), pooled_x, channel)
    z_k = F.normalize(z_k, dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global


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
