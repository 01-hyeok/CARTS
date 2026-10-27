# PHASE A -- Bug Impact Audit

## Bug confirmed at the exact reported location

`scripts/diag_v5_head_objective01.py` lines 233-239 (pre-fix):

```python
e_m = R_m - query_future.unsqueeze(1)                              # [B,5,H]
dot_mn = torch.einsum('bmh,bnh->bmn', e_m, e_m) / e_m.size(-1)      # [B,5,5] -- already /H
norm_m = e_m.norm(dim=-1)                                            # [B,5] -- RAW L2 norm, NOT /H
...
d_ = dot_mn[:, m, n]                                                 # = raw_dot/H
cos_ = d_ / (norm_m[:, m] * norm_m[:, n] + EPS)                      # = (raw_dot/H) / (||e_m||*||e_n||)
```

True cosine `= raw_dot / (||e_m|| * ||e_n||)`. The code's `cos_ = true_cosine / H` (exactly, up to the negligible `EPS=1e-8` term, which is many orders of magnitude smaller than any real norm product encountered in this data -- confirmed numerically in PHASE B/C below). **Root cause confirmed: `dot_mn` was divided by `H` for the Error-Dot definition, but `norm_m` (used only for the cosine denominator) was never supposed to carry that same `/H` scaling, and the code reused the already-H-divided `dot_mn` as the cosine numerator instead of the raw (undivided) dot product.**

## Files audited

| File | Role | Cosine-dependent? |
|---|---|---|
| `scripts/diag_v5_head_objective01.py` | Computes `cos_`/`dot_`/`overlap_` columns per query-channel row (full_pass mode) | YES -- bug source |
| `scripts/analyze_v5_head_objective01.py` | CPU aggregation; `phase_d()` reads `cos_*`/`dot_*` columns; `phase_c()`/`phase_e()`/`phase_g()` never reference them | PARTIAL -- only `phase_d()` |
| `scripts/diag_v5_head_gradient01.py` | Frozen-checkpoint gradient diagnostic (Phase F) -- computes `KL`/gradients only, never constructs `e_m`, `R_m`, or any cosine | NO |
| `tests/test_v5_head_objective_audit01.py` | `test_item7_head_error_dot_cosine` uses a hand-written, CORRECT cosine formula (`e1.norm(dim=-1)` properly paired with the SAME un-scaled dot) on synthetic 3-dim vectors -- never imports or exercises the buggy in-script `cos_` line at all | NO (this specific test was never exposed to the bug) |

## Classification of every PHASE D/E/F/G metric

| # | Metric | Classification | Reasoning |
|---|---|---|---|
| 1 | Error Cosine (`cos_{pair}`, `mean_cos`, `cross_head_mean_cosine_all_pairs`) | **AFFECTED** | Direct consumer of the buggy formula; true value ≈ H x reported value |
| 2 | Error Dot (`dot_{pair}`, `mean_dot`) | **UNAFFECTED** | `dot_mn = raw_dot/H` is exactly the spec's own Error-Dot definition (`D_mn = (1/H)<e_m,e_n>`) -- this was never buggy |
| 3 | Negative-Cosine Fraction | **UNAFFECTED** | `cos_ = true_cosine / H` with `H>0` is a POSITIVE scalar multiple of the true cosine -- `sign(cos_) = sign(true_cosine)` always, so `(cos_ < 0)` and `(true_cosine < 0)` select the identical set of rows |
| 4 | Overlap-vs-Cosine Pearson correlation | **UNAFFECTED** | Pearson correlation is invariant to any positive affine rescaling of one variable: `corr(X, c*Y) = corr(X, Y)` for constant `c > 0` (here `c = 1/H`, fixed per cell, same for every row) |
| 5 | Overlap-vs-Cosine Spearman correlation | **UNAFFECTED** | Spearman is invariant to ANY strictly monotonic transform, which a positive constant rescaling trivially is |
| 6 | Cross-Head complementarity INTERPRETATION ("near-zero", "near-orthogonal") | **AFFECTED** | This is a narrative conclusion drawn directly from the (wrong) absolute magnitude of metric #1 |
| 7 | Within-Mean-Mixture-Set D/C (`D_mix`, `C_mix`, `aggmse_mix`) | **UNAFFECTED** | Computed entirely from `pooled_memory_c`/`offset_c`/`query_future`/`mean_idx` (the ACTUAL selected Top-10 candidates) -- this code path never constructs `e_m`, `R_m`, `dot_mn`, or `cos_` at all; it is a structurally separate computation (confirmed by direct re-read of `run_full_pass`) |
| 8 | Head Contribution / 31-Subset Oracle (PHASE E) | **UNAFFECTED** | `subset_agg_mse()` operates on `p_heads` (softmax probabilities) and `pooled_memory_c` directly -- never touches `e_m`/`cos_`/`dot_mn` |
| 9 | Soft/Hard Gap (PHASE G) | **UNAFFECTED** | `aggmse_soft`/`aggmse_mix` derive from `p_mix` and the pooled candidate futures -- no dependency on the cosine computation |
| 10 | Gradient Norm / Gradient Cosine (PHASE F) | **UNAFFECTED** | Computed by an entirely separate script (`diag_v5_head_gradient01.py`) that never constructs an error vector or a cosine between error vectors -- its "gradient cosine" is between two LOSS GRADIENTS, a different object altogether, unrelated to this bug |

**Per the spec's own warning, items 3/4/5 are explicitly confirmed UNAFFECTED by direct mathematical argument (not assumed) -- the bug is a pure positive rescaling, and rank/sign-based quantities are invariant to it.** Only items 1 and 6 require correction; items 7-10 require NO correction and are expected to reproduce bit-identically when the forward pass is rerun with the fixed script (verified in PHASE D below).
