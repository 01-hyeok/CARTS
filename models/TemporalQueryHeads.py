"""TRACK-TEMPORAL-QUERY-HEADS01 -- Temporal Query Heads with Masked MLP
Tokens (TQH-5-Masked).

V5's relation encoder is `relation_encoder_type='mlp'`
(`scripts/train_j_shared_encoder_drift01.build_model`): a flat
`Linear(rows*seq_len, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)`
over the WHOLE flattened `[B, F, L]` relation input at once
(`models/RelationStage1.RelationEncoder._prepare_rows` + `forward`).
There is no patch/token structure inside this encoder --
`RelationEncoder.forward(..., return_tokens=True)` only works for
`encoder_type='transformer'` (raises `ValueError` otherwise), which is
NOT what V5 uses. So the existing Late-Interaction token path
(TRACK-F) cannot be reused here -- confirmed and reported before this
file was written (see `research/EXPERIMENT_LOG.md`).

Per explicit user design (2026-10-21): build a temporal "token" per
query WITHOUT adding any new encoder or learnable projection, by
re-running the SAME MLP encoder E on P time-masked copies of the exact
relation input `r_q = model._relation_tensor(x, c, c)` ([B, F, L]) that
V5 already feeds it, each copy zeroing every time position outside one
non-overlapping patch. A zero-input baseline `z_0 = E(zeros_like(r_q))`
is subtracted before normalizing, so the (possibly large) encoder bias/
mean-output component common to every patch does not dominate the
token bank. This adds ZERO new trainable parameters: masking is a
fixed buffer, and the only parameters in the whole head-generation
mechanism remain V5's own `SlotHeads.W`.
"""
import math

import torch
import torch.nn.functional as F


def build_patch_time_masks(seq_len, patch_len, device=None):
    """Deterministic non-overlapping patches along the time axis.
    `P = ceil(seq_len / patch_len)`; the last patch covers the
    remainder (shorter than `patch_len` when `seq_len` doesn't divide
    evenly) -- no padding, no randomness. Returns a bool tensor
    `[P, seq_len]`, broadcastable over `[B, F, L]` relation inputs."""
    if patch_len <= 0:
        raise ValueError(f'patch_len must be positive, got {patch_len}')
    n_patches = math.ceil(seq_len / patch_len)
    masks = torch.zeros(n_patches, seq_len, dtype=torch.bool, device=device)
    for p in range(n_patches):
        start = p * patch_len
        end = min(start + patch_len, seq_len)
        masks[p, start:end] = True
    return masks


def compute_temporal_token_bank(encoder, relation_x, patch_masks):
    """`relation_x`: `[B, F, L]` (the exact input V5's `encode_raw` sends
    to `model.encoder`). `patch_masks`: `[P, L]` bool, from
    `build_patch_time_masks`. Runs the SAME `encoder` module once on a
    `[B*(P+1), F, L]` batch (P masked patches + 1 all-zero baseline,
    concatenated so a single forward call covers everything -- no
    per-patch Python loop over the encoder). Returns:

        t_p      [B, P, D]  L2-normalized token bank, baseline-subtracted
        z_p_raw  [B, P, D]  pre-subtraction encoder output per patch (diagnostic)
        z_0      [B, D]     the zero-input baseline encoder output (diagnostic)

    `encoder` is called directly (not through `encode_raw`) because
    `encode_raw` only accepts `(model, x, c)` and rebuilds the relation
    tensor itself; here the SAME already-built `relation_x` is reused
    for both the patches and the (separately computed) pooled `z_q`,
    so there is exactly one `_relation_tensor` call per channel, matching
    V5's own call count."""
    bsz, feat, seq_len = relation_x.shape
    n_patches = patch_masks.size(0)
    mask_f = patch_masks.to(relation_x.dtype)  # [P, L]

    patches = relation_x.unsqueeze(1) * mask_f.view(1, n_patches, 1, seq_len)  # [B, P, F, L]
    zero_baseline = relation_x.new_zeros(bsz, 1, feat, seq_len)
    stacked = torch.cat([patches, zero_baseline], dim=1)  # [B, P+1, F, L]

    flat = stacked.reshape(bsz * (n_patches + 1), feat, seq_len)
    z_flat = encoder(flat)  # [B*(P+1), D]
    d_model = z_flat.size(-1)
    z_all = z_flat.reshape(bsz, n_patches + 1, d_model)
    z_p_raw = z_all[:, :n_patches, :]
    z_0 = z_all[:, n_patches, :]

    t_p = F.normalize(z_p_raw - z_0.unsqueeze(1), dim=-1)
    return t_p, z_p_raw, z_0


def temporal_attention_heads(g, t_p):
    """`g`: `[B, S, D]` (V5's own `SlotHeads(z_q)` output, unchanged).
    `t_p`: `[B, P, D]`, from `compute_temporal_token_bank`. No learnable
    parameters here -- pure dot-product attention, `g` as query,
    `t_p` as both key and value (per the user's explicit spec: the
    SAME token bank is read for both the attention logits and the
    weighted sum).

        e_{m,p} = g_m . t_p          (NO 1/sqrt(D) scaling)
        a_{m,p} = softmax_p(e_{m,p})
        q_m     = normalize(sum_p a_{m,p} t_p)

    2026-10-22 correction: `g` and `t_p` are BOTH already L2-normalized,
    so `g_m . t_p` is already a cosine similarity bounded in [-1, 1] --
    the standard Transformer 1/sqrt(D) correction exists to offset the
    variance growth of an UNnormalized dot product as D grows, which
    does not apply here. Measured on ETTh1_96 (D=128): with the
    division, logit std=0.0143 and softmax max-prob=0.170 (uniform=
    0.167, i.e. no real discrimination); without it, the same batch
    gives std=0.161 and the same scale as the token bank's own pairwise
    cosine diversity (0.003-0.49). No temperature parameter (fixed or
    learnable) is introduced -- this one, un-scaled, pre-defined formula
    is used everywhere in this track.

    Returns `q` `[B, S, D]` (L2-normalized, ready to score against
    candidates exactly like V5's `q = slot_heads(z_q)`), and `a`
    `[B, S, P]` (attention weights, for diagnostics)."""
    e = torch.einsum('bsd,bpd->bsp', g, t_p)
    a = torch.softmax(e, dim=-1)
    q = torch.einsum('bsp,bpd->bsd', a, t_p)
    q = F.normalize(q, dim=-1)
    return q, a


def attention_entropy(a, eps=1e-8):
    """`a`: `[B, S, P]`. Returns `[B, S]` Shannon entropy over the P axis."""
    a_safe = a.clamp_min(eps)
    return -(a_safe * a_safe.log()).sum(-1)


def normalized_attention_entropy(a, eps=1e-8):
    """`a`: `[B, S, P]`. `attention_entropy` divided by `log(P)` (the
    uniform-distribution entropy), so 1.0 == exactly uniform and 0.0 ==
    a one-hot pick -- scale-free, comparable across different P."""
    p = a.size(-1)
    return attention_entropy(a, eps) / math.log(p)


def pairwise_head_attention_cosine(a):
    """`a`: `[B, S, P]`. Mean pairwise cosine similarity between
    DIFFERENT heads' attention distributions (off-diagonal, S>=2),
    averaged over the batch -- a value near 1.0 means the heads have
    collapsed to (near-)identical temporal attention (spec item 14)."""
    bsz, s, p = a.shape
    if s < 2:
        return float('nan')
    a_norm = F.normalize(a, dim=-1)
    cos = torch.einsum('bsp,btp->bst', a_norm, a_norm)  # [B, S, S]
    iu = torch.triu_indices(s, s, offset=1)
    return float(cos[:, iu[0], iu[1]].mean())


def query_attention_diversity(a):
    """`a`: `[B, S, P]`. For EACH head `m`, mean pairwise cosine
    similarity between DIFFERENT queries' (batch rows') attention
    vectors for that head, then averaged over heads. A value near 1.0
    means that head's attention is (near-)identical across different
    queries, i.e. not actually query-adaptive (spec item 13)."""
    bsz, s, p = a.shape
    if bsz < 2:
        return float('nan')
    a_norm = F.normalize(a, dim=-1)  # [B, S, P]
    per_head = []
    for m in range(s):
        am = a_norm[:, m, :]  # [B, P]
        cos = am @ am.T  # [B, B]
        iu = torch.triu_indices(bsz, bsz, offset=1)
        per_head.append(cos[iu[0], iu[1]].mean())
    return float(torch.stack(per_head).mean())
