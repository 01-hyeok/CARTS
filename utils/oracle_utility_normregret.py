"""TRACK-A-SET-NORMREGRET-CONTROL01 -- normalized-regret Soft CE (A2) and
normalized Soft Regret Mass (A3), with an EXPLICIT, separately-controlled
teacher temperature `tau_T` (spec section 8) -- the gap this experiment's
own audit found in `utils/set_loss_experimental.py`'s existing
`soft_regret_mass_loss`/`set_utility_soft_ce_loss`: their `weight =
exp(-regret_norm)` never divides by any temperature (equivalent to an
implicit, unexposed `tau_T=1`), so TRACK-A-SET-LOSS-CONTROL01/EXP-SET-LOSS01
did not run under this spec's `tau_T=0.1`.

REUSES, UNMODIFIED: `utils.set_loss_experimental.compute_oracle_teacher_signal`
for the affine-invariant normalized regret itself (`regret_norm`), the
tie-inclusive Top-M' positive set (`p_mask`), and the all-tied/no-preference
row zeroing (`informative_row`) -- exactly spec sections 6-7's math, already
implemented and unit-tested there; NOT reimplemented here. This module only
adds the missing `tau_T` division when turning `regret_norm` into a
teacher weight, and the two loss reductions spec sections 10-11 specify
(which structurally differ from the existing module's own SRM formula --
see `normregret_srm_loss` docstring).

Both A2 and A3 call `normregret_weight()` -- ONE shared weight builder
(spec: "두 normalized loss가 동일한 weight builder를 공유").
"""
import torch
import torch.nn.functional as F

from utils.set_loss_experimental import compute_oracle_teacher_signal

_MAX_EXPONENT = 80.0


def _check_finite(u_target, valid_mask):
    bad = valid_mask & ~torch.isfinite(u_target)
    if bool(bad.any()):
        raise ValueError(
            f'[ISSUE] NaN/Inf utility at a VALID candidate position (spec section 12 item 5) -- '
            f'{int(bad.sum())} offending entries. Refusing to silently include them.')


def _student_log_probs(u_hat, valid_mask, tau_S):
    """Byte-identical convention to oracle_choice_step_loss's own logits/
    log_probs lines -- same mask, same temperature location."""
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    logits = u_hat.masked_fill(~valid_mask, neg_inf) / float(tau_S)
    return F.log_softmax(logits, dim=-1)


def normregret_weight(u_target, valid_mask, tau_T, m=10, teacher=None):
    """w_t(i) = exp(-regret_norm(i) / tau_T) for i in P_t, else 0.

    `regret_norm` (from `compute_oracle_teacher_signal`, reused unmodified)
    is already affine-invariant under `u' = a*u + b, a>0` (shift via the
    numerator/denominator both being differences of `u`, scale via both
    being proportional to `u`'s scale) -- dividing by the FIXED `tau_T`
    afterward does not break that invariance (tau_T is not a function of
    u). No gradient reaches `u_target` (`compute_oracle_teacher_signal`
    runs entirely under `torch.no_grad()`).
    """
    _check_finite(u_target, valid_mask)
    if teacher is None:
        teacher = compute_oracle_teacher_signal(u_target, valid_mask, m=m)
    p_mask, regret_norm = teacher['p_mask'], teacher['regret_norm']
    scaled = torch.clamp(regret_norm / float(tau_T), min=0.0, max=_MAX_EXPONENT)
    weight = torch.where(p_mask, torch.exp(-scaled), torch.zeros_like(regret_norm))
    return weight, teacher


@torch.no_grad()
def _shared_diag(u_hat, valid_mask, teacher, weight, log_probs):
    p_mask = teacher['p_mask']
    row_has_valid = teacher['row_has_valid']
    probs = log_probs.exp()
    topm_mass = (probs * p_mask.float()).sum(dim=-1)
    student_entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    eff_pos = p_mask.float().sum(dim=-1)
    w_sum = weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
    pi = weight / w_sum
    pi_c = pi.clamp_min(1e-12)
    teacher_entropy = -(pi_c * pi_c.log()).sum(-1)
    teacher_top1_weight = weight.max(dim=-1).values / w_sum.squeeze(-1)
    rows = row_has_valid
    m = lambda x: float(x[rows].mean()) if bool(rows.any()) else float('nan')
    return {
        'topm_probability_mass': m(topm_mass), 'student_entropy': m(student_entropy),
        'effective_positive_count': m(eff_pos), 'teacher_entropy': m(teacher_entropy),
        'teacher_top1_weight': m(teacher_top1_weight),
        'informative_row_fraction': m(teacher['informative_row'].float()),
    }


def normregret_softce_loss(u_hat, u_target, valid_mask, tau_S, tau_T, m=10, teacher=None):
    """spec section 10: L_SoftCE,t = -sum_{i in P_t} pi(i) log p_theta(i),
    pi(i) = w_t(i) / sum_{P_t} w_t."""
    weight, teacher = normregret_weight(u_target, valid_mask, tau_T, m=m, teacher=teacher)
    p_mask = teacher['p_mask']
    informative_row, row_has_valid = teacher['informative_row'], teacher['row_has_valid']

    log_probs = _student_log_probs(u_hat, valid_mask, tau_S)
    weight_sum = weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
    pi = weight / weight_sum
    safe_log_probs = log_probs.masked_fill(~p_mask, 0.0)  # avoid 0 * (-inf) = NaN
    row_loss = -(pi * safe_log_probs).sum(dim=-1)
    row_loss = row_loss * informative_row.float()

    sel = row_loss[row_has_valid]
    loss = sel.mean() if sel.numel() > 0 else u_hat.sum() * 0.0
    diag = _shared_diag(u_hat, valid_mask, teacher, weight, log_probs)
    return loss, diag


def normregret_srm_loss(u_hat, u_target, valid_mask, tau_S, tau_T, m=10, teacher=None, eps=1e-8):
    """spec section 11: L_SRM,t = -log(sum_{i in P_t} p_theta(i) w_t(i) + eps).

    Deliberately the DIRECT-sum form the spec writes, not the older
    `utils.set_loss_experimental.soft_regret_mass_loss`'s logsumexp
    reformulation (which is only exactly equivalent to this when
    `tau_T=1` and omits the `+eps`) -- kept as literally specified so this
    experiment's math matches what the spec defines, not a reformulation
    someone must re-derive is equivalent."""
    weight, teacher = normregret_weight(u_target, valid_mask, tau_T, m=m, teacher=teacher)
    informative_row, row_has_valid = teacher['informative_row'], teacher['row_has_valid']

    log_probs = _student_log_probs(u_hat, valid_mask, tau_S)
    probs = log_probs.exp()
    mass = (probs * weight).sum(dim=-1)
    row_loss = -torch.log(mass + eps)
    row_loss = row_loss * informative_row.float()

    sel = row_loss[row_has_valid]
    loss = sel.mean() if sel.numel() > 0 else u_hat.sum() * 0.0
    diag = _shared_diag(u_hat, valid_mask, teacher, weight, log_probs)
    return loss, diag
