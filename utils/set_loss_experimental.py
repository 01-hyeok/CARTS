"""EXP-SET-LOSS01 -- two experimental 1-term losses for the Greedy Set
Oracle: Soft Regret Mass (SRM) and Set-Utility Soft Cross-Entropy (SoftCE).

Both losses consume the SAME reference Set-Oracle utility tensor
(`scripts.train_factorial_e2e01.greedy_set_utility`, UNCHANGED, UNMODIFIED)
and the SAME student score/mask/temperature convention as the existing
Hard Choice CE (`scripts.train_oracle_choice01.oracle_choice_step_loss`):

    logits = u_hat.masked_fill(~valid_mask, neg_inf) / tau
    log_probs = F.log_softmax(logits, dim=-1)

This module does not call, import, or modify any reference Oracle
implementation -- it only consumes an already-computed `u_target` tensor
(utility, LARGER = BETTER, confirmed by code trace: `greedy_set_utility`
returns `-dense_utility(...)`, `dense_utility` returns MSE, and every
existing call site picks `u_target.argmax(...)` as the best action).

Epsilon design (see EXP-SET-LOSS01 report for the derivation and the two
invariance unit tests that motivate it):

    spread_t   = u_t* - min_{i in V_t} u_t(i)         (shift-invariant)
    eps_t      = finfo(dtype).eps * (|spread_t| + finfo(dtype).eps)
    d_t        = max(u_t* - u_t^(M), eps_t)

`eps_t` is proportional to the row's OWN utility spread (not an arbitrary
constant, and not proportional to the utilities' absolute magnitude) so
that adding a constant to every utility in a row leaves `d_t` and every
downstream quantity EXACTLY unchanged (spread is shift-invariant), and
multiplying every utility by a positive constant leaves them
APPROXIMATELY unchanged (exact in the regime where `spread >>
finfo(dtype).eps`, which is every non-degenerate row).
"""
import torch
import torch.nn.functional as F


def _student_log_probs(u_hat, valid_mask, tau):
    """Byte-identical convention to
    `scripts.train_oracle_choice01.oracle_choice_step_loss`'s own
    `logits`/`log_probs` lines -- same mask, same temperature location."""
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    logits = u_hat.masked_fill(~valid_mask, neg_inf) / float(tau)
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def compute_oracle_teacher_signal(u_target, valid_mask, m=10):
    """Oracle Top-M positive support `P_t`, normalized regret
    `regret_norm`, and teacher quality weight `weight` (already 0 outside
    `P_t`), plus `informative_row` (False = no oracle preference at all in
    that row -> both losses must return exactly 0 for it, per spec section
    10 item 3). Every returned tensor is created under `no_grad` (the
    caller never needs to `.detach()` again) -- gradient never flows into
    `u_target` or these teacher quantities.

    `u_target` should be the RAW utility (already masked or not -- this
    function applies its own masking, matching the Hard-CE convention of
    masking with `neg_inf` at invalid positions before any reduction).
    """
    neg_inf = torch.finfo(u_target.dtype).min / 4
    pos_inf = torch.finfo(u_target.dtype).max / 4
    u = u_target.masked_fill(~valid_mask, neg_inf)
    bsz, n = u.shape
    row_has_valid = valid_mask.any(dim=-1)
    n_valid = valid_mask.sum(dim=-1)

    u_star = u.max(dim=-1).values                                    # [B]
    u_min_valid = u_target.masked_fill(~valid_mask, pos_inf).min(dim=-1).values  # [B]
    spread = (u_star - u_min_valid)                                  # [B], shift-invariant, >=0

    # ---- per-row M-th-largest VALID utility (ties at the cutoff are
    # included automatically by the >= comparison below; this is NOT an
    # approximation of "top M", it is exactly "top >= M, more if tied") ----
    sorted_u, _ = u.sort(dim=-1, descending=True)
    k_idx = (torch.clamp(n_valid, max=m) - 1).clamp_min(0)            # [B], 0-indexed rank
    u_m = sorted_u.gather(1, k_idx.unsqueeze(-1)).squeeze(-1)         # [B]

    eps = torch.finfo(u.dtype).eps
    eps_row = eps * (spread.abs() + eps)                              # [B], see module docstring
    d = torch.clamp_min(u_star - u_m, eps_row)                        # [B]

    u_star_col = u_star.unsqueeze(-1)
    d_col = d.unsqueeze(-1)
    regret_norm = (u_star_col - u) / d_col                            # [B, N]; +inf-ish at invalid
                                                                       # (never read outside P_mask)
    p_mask = valid_mask & (u >= u_m.unsqueeze(-1))                    # [B, N], P_t

    weight = torch.where(p_mask, torch.exp(-regret_norm.clamp(min=0.0, max=80.0)),
                         torch.zeros_like(u))
    weight = torch.where(p_mask, weight, torch.zeros_like(weight))    # belt-and-suspenders

    informative_row = row_has_valid & (spread > eps_row)

    diag = {
        'spread': spread, 'd': d, 'eps_row': eps_row, 'u_star': u_star, 'u_m': u_m,
        'n_valid': n_valid, 'row_has_valid': row_has_valid,
    }
    return {'p_mask': p_mask, 'regret_norm': regret_norm, 'weight': weight,
           'informative_row': informative_row, 'row_has_valid': row_has_valid, 'diag': diag}


def soft_regret_mass_loss(u_hat, u_target, valid_mask, tau, m=10, teacher=None):
    """L_SRM,t = -logsumexp_{i in P_t} [log p_theta(i) - regret_norm(i)],
    averaged over rows with any valid candidate (rows with zero oracle
    preference contribute exactly 0, per spec).

    `teacher`: optional precomputed `compute_oracle_teacher_signal(...)`
    output (so TF/on-policy call sites that already computed it once for
    diagnostics don't pay for it twice). Computed fresh if omitted.
    Returns (loss, diag).
    """
    if teacher is None:
        teacher = compute_oracle_teacher_signal(u_target, valid_mask, m=m)
    p_mask, regret_norm = teacher['p_mask'], teacher['regret_norm']
    informative_row, row_has_valid = teacher['informative_row'], teacher['row_has_valid']

    log_probs = _student_log_probs(u_hat, valid_mask, tau)
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    terms = (log_probs - regret_norm).masked_fill(~p_mask, neg_inf)
    row_loss = -torch.logsumexp(terms, dim=-1)                        # [B]
    row_loss = row_loss * informative_row.float()

    sel = row_loss[row_has_valid]
    loss = sel.mean() if sel.numel() > 0 else u_hat.sum() * 0.0

    with torch.no_grad():
        diag = _shared_diagnostics(u_hat, valid_mask, teacher, log_probs)
    return loss, diag


def set_utility_soft_ce_loss(u_hat, u_target, valid_mask, tau, m=10, teacher=None):
    """L_SoftCE,t = -sum_{i in P_t} pi(i) log p_theta(i), pi = weight /
    sum_{P_t} weight. Same informative-row zeroing as SRM."""
    if teacher is None:
        teacher = compute_oracle_teacher_signal(u_target, valid_mask, m=m)
    p_mask, weight = teacher['p_mask'], teacher['weight']
    informative_row, row_has_valid = teacher['informative_row'], teacher['row_has_valid']

    log_probs = _student_log_probs(u_hat, valid_mask, tau)
    weight_sum = weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
    pi = weight / weight_sum                                          # 0 outside P_t
    safe_log_probs = log_probs.masked_fill(~p_mask, 0.0)              # avoid 0 * (-inf) = NaN
    row_loss = -(pi * safe_log_probs).sum(dim=-1)                     # [B]
    row_loss = row_loss * informative_row.float()

    sel = row_loss[row_has_valid]
    loss = sel.mean() if sel.numel() > 0 else u_hat.sum() * 0.0

    with torch.no_grad():
        diag = _shared_diagnostics(u_hat, valid_mask, teacher, log_probs)
        diag['teacher_entropy'] = float(
            (-(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(-1))[row_has_valid].mean()
        ) if bool(row_has_valid.any()) else float('nan')
    return loss, diag


@torch.no_grad()
def _shared_diagnostics(u_hat, valid_mask, teacher, log_probs):
    p_mask = teacher['p_mask']
    row_has_valid = teacher['row_has_valid']
    probs = log_probs.exp()
    topm_mass = (probs * p_mask.float()).sum(dim=-1)                  # student mass on P_t
    student_entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    eff_pos = p_mask.float().sum(dim=-1)
    rows = row_has_valid
    return {
        'topm_probability_mass_mean': float(topm_mass[rows].mean()) if bool(rows.any()) else float('nan'),
        'student_entropy_mean': float(student_entropy[rows].mean()) if bool(rows.any()) else float('nan'),
        'effective_positive_count_mean': float(eff_pos[rows].mean()) if bool(rows.any()) else float('nan'),
        'informative_row_fraction': float(teacher['informative_row'][rows].float().mean())
                                    if bool(rows.any()) else float('nan'),
    }
