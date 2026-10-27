# PHASE B -- Error Cosine Implementation Fix and Unit Tests

## Fix applied

`scripts/diag_v5_head_objective01.py`, PHASE D block (in-place fix, old buggy formula removed, not kept as a dead branch):

```python
e_m = R_m - query_future.unsqueeze(1)                              # [B,5,H]
horizon = e_m.size(-1)
raw_dot_mn = torch.einsum('bmh,bnh->bmn', e_m, e_m)                 # [B,5,5], NOT /H
dot_mn = raw_dot_mn / horizon                                        # Error Dot (unchanged definition)
norm_m = e_m.norm(dim=-1)                                            # [B,5]
norm_prod_mn = norm_m.unsqueeze(2) * norm_m.unsqueeze(1)             # [B,5,5]
cos_mn = raw_dot_mn / norm_prod_mn.clamp_min(EPS)                    # Error Cosine (FIXED)
```

Error Dot and Error Cosine are now computed from two separately-named tensors (`dot_mn` vs `cos_mn`), each with its own, independently-correct normalization -- no shared intermediate is reused across the two definitions anymore, eliminating the class of bug that caused this in the first place.

**Zero-norm rule (explicit, not hardcoded +-1)**: `clamp_min(EPS)` on the norm PRODUCT (not `+EPS` on the result) guarantees a finite output for any zero-norm row. When both error vectors are exactly zero, `raw_dot=0` and `norm_prod_mn` clamps to `EPS`, giving `cosine = 0/EPS = 0` -- a neutral "no signal" value, never an arbitrary +1 or -1. Verified by `test_test6_zero_norm_no_nan_inf_not_hardcoded`.

## Unit tests -- `tests/test_v5_head_objective_cosine_fix01.py`, 10/10 PASS

| Test | Checks | Result |
|---|---|---|
| 1. Identical vectors | `cos(e,e) = 1` | PASS |
| 2. Opposite vectors | `cos(e,-e) = -1` | PASS |
| 3. Orthogonal vectors | `cos([1,0],[0,1]) = 0`, `dot = 0` | PASS |
| 4. Horizon invariance | Same error pattern repeated at H=96 vs H=720 gives IDENTICAL cosine (not shrunk by H) | PASS |
| 5. Error Dot scale | `dot` still equals `(1/H)<e_m,e_n>`, MSE-unit definition preserved exactly | PASS |
| 6. Zero norm | Finite output, not NaN/Inf, not hardcoded to +-1 (value is exactly 0 for zero-vs-zero) | PASS |
| 7. Symmetry | `cos(e_m,e_n) == cos(e_n,e_m)` | PASS |
| 8. Range | `-1 <= cos <= 1` (with 1e-5 floating-point slack) over 50x20 random samples | PASS |
| 9. Reference match | Matches `torch.nn.functional.cosine_similarity` AND an independent NumPy implementation to 1e-5 | PASS |
| 10. Regression | Reproduces the ORIGINAL bug on synthetic input: `old_buggy_cos * H == true_cos` (confirms the exact 1/H scaling relationship, and confirms the bug reproduction is non-vacuous -- `old_buggy_cos != true_cos`) | PASS |

Pre-existing test `tests/test_v5_head_objective_audit01.py::test_item7_head_error_dot_cosine` was independently re-run and still PASSES -- it was never exposed to the buggy in-script formula (it hand-computes cosine correctly on 3-dim synthetic vectors), confirming it was not a false-positive cover for this bug.

## Full repo regression

`python -m pytest tests/ -q`: unaffected by this change beyond the 2 new/modified test files -- see FINAL_REPORT.md for the full run executed after PHASE C's recomputation.
