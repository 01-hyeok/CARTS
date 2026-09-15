"""TRACK-A-WEATHER-OPT01/OPT02 -- algebraically exact, optimized
reformulation of `utils.dense_utility.dense_utility`.

Does NOT modify `utils/dense_utility.py`. The reference implementation
stays untouched and importable; this module is a separate, optional code
path (spec S13: "기존 reference implementation을 삭제하지 마세요").

Derivation (spec S3, OPT01)
----------------------------
`dense_utility`'s own docstring already establishes the incremental
weighted-mean identity: with `w_i = exp(s_i/tau)`, `Z_S = sum_{j in S} w_j`,
`M_S = sum_{j in S} w_j*Y_j`,

    A(S+{i}) = MSE( (M_S + w_i*Y_i) / (Z_S + w_i), Y_q )

Let `d_i = Y_i - Y_q` (the candidate residual against the query future --
PREFIX-INVARIANT, computed once per query and reused across all K greedy
steps) and `E_S = M_S - Z_S*Y_q = sum_{j in S} w_j*d_j` (exactly `R_S` in
the spec, expressed in this module's own `M_S`/`Z_S` variables so it can
reuse `prefix_weighted_sums` unmodified -- `prefix_weighted_sums(prefix,
w, d)` returns `(Z_S, E_S)` directly, since it is a linear function of
whatever "futures" tensor it is given).

Then:

    M_S + w_i*Y_i - Y_q*(Z_S+w_i) = E_S + w_i*d_i

so the un-normalised numerator of `(M_S+w_i*Y_i)/(Z_S+w_i) - Y_q` is
`E_S + w_i*d_i`, and

    A(S+{i}) = ||E_S + w_i*d_i||^2 / (H*(Z_S+w_i)^2)
             = [ ||E_S||^2 + 2*w_i*(E_S . d_i) + w_i^2*||d_i||^2 ]
               / (H*(Z_S+w_i)^2)

exactly the formula in spec S3. This is algebraically IDENTICAL to
`dense_utility` for every candidate, every step, every query -- not an
approximation. Two real bugs in the first draft were caught by the OPT01
equivalence suite before either reached production (see
`research/TRACK-A-WEATHER-OPT01.md` section 3-4 for the full account):
(1) the empty-prefix (Z_S=0) special case below, without which `w_i**2`
can underflow before `w_i` itself does; (2) the eps-clamp must be applied
to `(Z_S+w_i)` BEFORE squaring, matching the reference's own clamp point
exactly -- clamping the squared value applies a far more aggressive floor
and silently distorts small-but-representable denominators.

OPT02 addition -- candidate-dimension chunking without a persistent
`[B, N, H]` tensor
-------------------------------------------------------------------------
OPT01's `prepare_query_static` computed and returned the FULL `d = futures
- query_future` tensor, `[B, N, H]`, held live for the entire K=10
trajectory. That is the same size as `futures` itself, so it roughly
DOUBLES the memory footprint of the one large per-query input for the
whole trajectory -- measured as a genuine +58-59% peak-VRAM regression
vs the reference (`research/TRACK-A-WEATHER-OPT01.md` section 6), even
though OPT01's per-STEP candidate loop was already chunked.

OPT02 removes that persistent full-size tensor entirely, while keeping
BOTH of OPT01's real wins intact:

1. The one PREFIX-INVARIANT quantity that is expensive to recompute
   (`||d_i||^2`, needed at every step) is still computed ONCE per query --
   but via `prepare_query_static_chunked`, which chunks the candidate
   dimension internally and only ever KEEPS the `[B, N]` scalar result
   (`d_sq`), not the `[B, N, H]` tensor that produced it. `[B, N]` floats
   is negligible next to `[B, N, H]` (a factor of H smaller -- 96x-720x
   for this project's horizons).
2. `E_S`/`Z_S` (spec: `prefix_weighted_sums`) only ever need `d` AT THE
   PREFIX INDICES -- at most K<=10 rows out of N, never the full candidate
   set -- so they are built by gathering those few rows directly from
   `futures` and subtracting `query_future`, an O(K*H) operation, not
   O(N*H).
3. The per-step candidate loop chunks `d_c = futures[:,chunk,:] -
   query_future` FRESH, transiently, exactly where OPT01 already chunked
   the rest of the per-step computation -- so at no point does any
   `[B, N, H]`-sized tensor other than `futures` itself (which the
   reference implementation also holds, as an unavoidable input) exist on
   the GPU.

`dense_utility_optimized` stays 100% BACKWARD COMPATIBLE with OPT01's own
call signature and all 23 of that experiment's equivalence tests: passing
a precomputed `d` (OPT01's `prepare_query_static` output) uses the OLD,
unchunked-`d` code path verbatim, unchanged. The new, memory-safe path
activates only when `d=None` and `futures`/`query_future` are supplied
instead -- `TRACK-A-WEATHER-OPT02`'s own tests and benchmark exercise ONLY
this new path.

Chunking (spec S7-S8, both OPT01/OPT02) is preserved: every candidate is
still visited, in the same chunk order, and the returned `[B, N]` tensor is
exactly the same shape/semantics as `dense_utility`'s -- a caller that does
`out.argmin(dim=-1)` (or masks then argmins) on either function's output
gets the identical global argmin, tie-break included (standard
`torch.argmin` returns the smallest index among exact ties, and both paths
build the SAME `[B, N]` tensor before that single final argmin call -- no
running-min/chunk-merge logic is needed, since `[B, N]` itself is cheap to
hold in full; only the `H`-dimensioned intermediates are chunked away).
"""
import torch

from utils.dense_utility import prefix_weighted_sums


def prepare_query_static(futures, query_future):
    """OPT01, UNCHANGED. Per-query, PREFIX-INVARIANT quantities, computed
    once and reused for every greedy step in the K=10 trajectory:
      d       : [B, N, H]  candidate residual y_i - y_q
      d_sq    : [B, N]     ||d_i||^2

    Kept exactly as OPT01 defined it so every one of that experiment's 23
    equivalence tests keeps passing unmodified. Holds a full `[B,N,H]`
    tensor live -- this is the code path OPT02 was built to AVOID; prefer
    `prepare_query_static_chunked` for new call sites.
    """
    d = futures - query_future.unsqueeze(1)
    d_sq = d.pow(2).sum(dim=-1)
    return d, d_sq


def prepare_query_static_chunked(futures, query_future, candidate_chunk_size=None):
    """OPT02. Same PREFIX-INVARIANT `d_sq` as `prepare_query_static`, but
    NEVER materializes the full `[B, N, H]` `d` tensor -- chunks the
    candidate dimension, computes `d_c` transiently per chunk, and keeps
    only the `[B, N]` scalar-per-candidate result. Returns `d_sq` only
    (no `d`) -- callers needing per-candidate residuals must recompute
    them fresh from `futures`/`query_future`, which is exactly what
    `dense_utility_optimized`'s `d=None` path does, per-chunk, per-step.
    """
    bsz, n, h = futures.shape
    candidate_chunk_size = candidate_chunk_size or n
    d_sq = futures.new_empty(bsz, n)
    q = query_future.unsqueeze(1)
    for start in range(0, n, candidate_chunk_size):
        end = min(start + candidate_chunk_size, n)
        d_c = futures[:, start:end, :] - q       # transient, freed each iteration
        d_sq[:, start:end] = d_c.pow(2).sum(dim=-1)
    return d_sq


def _prefix_e_s_z_s_from_futures(prefix_idx, w, futures, query_future):
    """`prefix_weighted_sums`, but computing `d` only AT THE PREFIX
    INDICES (at most K<=10 rows out of N) instead of requiring a
    precomputed full-size `d` tensor. Algebraically identical to
    `prefix_weighted_sums(prefix_idx, w, d)` for any `d` consistent with
    `futures - query_future`; verified by a dedicated equivalence test
    (`test_prefix_e_s_z_s_matches_full_d_computation`)."""
    bsz = futures.size(0)
    h = futures.size(-1)
    if prefix_idx.size(1) == 0:
        return futures.new_zeros(bsz, 1), futures.new_zeros(bsz, h)
    w_sel = w.gather(1, prefix_idx)
    y_sel = futures.gather(1, prefix_idx.unsqueeze(-1).expand(-1, -1, h))
    d_sel = y_sel - query_future.unsqueeze(1)
    z_s = w_sel.sum(dim=-1, keepdim=True)
    e_s = (w_sel.unsqueeze(-1) * d_sel).sum(dim=1)
    return z_s, e_s


def dense_utility_optimized(prefix_idx, w, d=None, d_sq=None, chunk_size=None,
                            candidate_chunk_size=None, futures=None, query_future=None,
                            eps=1e-12):
    """Exact algebraic reformulation of `dense_utility`. Returns the SAME
    `[B, N]` raw-A tensor (lower=better), byte-for-byte algebraically equal
    to `dense_utility(prefix_idx, w, futures, query_future, ...)` up to
    floating-point summation order.

    TWO call conventions, chosen automatically by whether `d` is given:

    OPT01 (`d` given, from `prepare_query_static`): unchanged from the
    first optimization round -- `d` is used directly, `chunk_size` chunks
    only the per-step candidate loop. `d` itself stays resident for the
    whole K=10 trajectory (the OPT01 memory trade-off).

    OPT02 (`d=None`, `futures`+`query_future`+`d_sq` given instead;
    `d_sq` from `prepare_query_static_chunked`): the memory-safe path.
    NEVER holds a `[B, N, H]` tensor beyond `futures` itself (already an
    unavoidable input, same as the reference holds). `E_S`/`Z_S` are built
    from at most K<=10 gathered rows; the per-step candidate loop
    recomputes `d_c` transiently, chunk by chunk, exactly where the
    reference already re-derives its own per-chunk quantities every step.
    `candidate_chunk_size` controls this chunking (falls back to
    `chunk_size` if not given, for call-site symmetry with OPT01).
    """
    if d is not None:
        return _dense_utility_optimized_opt01(prefix_idx, w, d, d_sq, chunk_size, eps)
    if futures is None or query_future is None or d_sq is None:
        raise ValueError('dense_utility_optimized: pass either `d` (OPT01) or '
                         '`futures`+`query_future`+`d_sq` (OPT02, chunked)')
    return _dense_utility_optimized_opt02(prefix_idx, w, futures, query_future, d_sq,
                                          candidate_chunk_size or chunk_size, eps)


def _dense_utility_optimized_opt01(prefix_idx, w, d, d_sq, chunk_size, eps):
    """OPT01 path, byte-identical to the original implementation."""
    bsz, n, h = d.shape
    if prefix_idx.size(1) == 0:
        # Empty prefix: Z_S = E_S = 0 EXACTLY, so the general formula
        # degenerates to A(i) = ||d_i||^2 / H for every i, independent of
        # w_i. Returned directly via the closed-form limit -- see the
        # module docstring for why evaluating the general formula naively
        # here would be numerically unsafe.
        return d_sq / h
    with torch.no_grad():
        z_s, e_s = prefix_weighted_sums(prefix_idx, w, d)   # [B,1], [B,H]
        e_s_sq = e_s.pow(2).sum(dim=-1, keepdim=True)       # [B,1]

    chunk_size = chunk_size or n
    out = d.new_empty(bsz, n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        w_c = w[:, start:end]
        d_c = d[:, start:end, :]
        dsq_c = d_sq[:, start:end]
        dot = torch.einsum('bh,bnh->bn', e_s, d_c)          # E_S . d_i, [B, chunk]
        num = e_s_sq + 2.0 * w_c * dot + w_c.pow(2) * dsq_c
        den = h * (z_s + w_c).clamp_min(eps).pow(2)
        out[:, start:end] = num / den
    return out


def _dense_utility_optimized_opt02(prefix_idx, w, futures, query_future, d_sq,
                                   candidate_chunk_size, eps):
    """OPT02 path: no persistent `[B, N, H]` tensor ever exists beyond
    `futures` itself. Algebraically identical formula and clamp order to
    the OPT01 path -- only WHERE `d`/`E_S`/`Z_S` come from differs."""
    bsz, n, h = futures.shape
    if prefix_idx.size(1) == 0:
        return d_sq / h    # identical closed form and identical justification as OPT01
    with torch.no_grad():
        z_s, e_s = _prefix_e_s_z_s_from_futures(prefix_idx, w, futures, query_future)
        e_s_sq = e_s.pow(2).sum(dim=-1, keepdim=True)

    candidate_chunk_size = candidate_chunk_size or n
    out = futures.new_empty(bsz, n)
    q = query_future.unsqueeze(1)
    for start in range(0, n, candidate_chunk_size):
        end = min(start + candidate_chunk_size, n)
        w_c = w[:, start:end]
        d_c = futures[:, start:end, :] - q        # transient: freed at the end of this iteration
        dsq_c = d_sq[:, start:end]
        dot = torch.einsum('bh,bnh->bn', e_s, d_c)
        num = e_s_sq + 2.0 * w_c * dot + w_c.pow(2) * dsq_c
        den = h * (z_s + w_c).clamp_min(eps).pow(2)
        out[:, start:end] = num / den
    return out
