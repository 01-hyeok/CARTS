"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- Router architecture, losses, and
routed-selection primitives. The Router consumes ONLY the frozen
backbone's own past-representation z_qc (`encode_raw_channel_first`
output for channel c, pre-computed once since encoder/SlotHeads/
retriever are frozen throughout this phase -- no future information,
no calendar-only feature) and outputs a 5-way distribution over heads,
per (query, channel), per NEXT_ROUTER_MODE=CHANNEL_AWARE
(TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01's Phase 1 decision).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SLOTS = 5


class AggregateRouter(nn.Module):
    """Linear(d, d/2) -> GELU -> Linear(d/2, 5). No capacity sweep per
    spec section 4 -- this is the ONLY architecture used."""

    def __init__(self, d_model, num_slots=NUM_SLOTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_slots),
        )

    def forward(self, z_qc):
        return self.net(z_qc)  # logits [..., 5]


def aggregate_soft_teacher(U_agg, tau_r):
    """R2: z-score U_agg across heads (last dim), then softmax(-z/tau_R).
    U_agg: [..., 5], lower=better."""
    mu = U_agg.mean(dim=-1, keepdim=True)
    sigma = U_agg.std(dim=-1, keepdim=True, unbiased=False)
    z = (U_agg - mu) / (sigma + 1e-8)
    return torch.softmax(-z / tau_r, dim=-1)


def r1_ce_loss(router_logits, h_star_agg):
    """R1: CE(router_logits, argmin_h U_agg)."""
    return F.cross_entropy(router_logits, h_star_agg)


def r2_kl_loss(router_logits, soft_teacher):
    """R2: KL(teacher || router_pi)."""
    log_pi = F.log_softmax(router_logits, dim=-1)
    teacher = soft_teacher.detach()
    return (teacher * (teacher.clamp_min(1e-12).log() - log_pi)).sum(dim=-1).mean()


def hard_routed_topk(router_logits, p_h, k=10):
    """A. Hard routing: h_hat = argmax(pi); use that head's own
    standalone Top-K distribution. p_h: [B,5,M] per-head softmax
    probabilities over the pool (already computed by the scoring
    pipeline). Returns picks [B,k]."""
    h_hat = router_logits.argmax(dim=-1)  # [B]
    bsz = p_h.size(0)
    p_selected = p_h[torch.arange(bsz, device=p_h.device), h_hat]  # [B,M]
    picks = p_selected.topk(k, dim=-1).indices
    return picks, h_hat


def soft_routed_topk(router_logits, p_h, k=10):
    """B. Soft routing: p_route(i) = sum_h pi_h * p_h(i). Returns
    picks [B,k] and the routed mixture probability p_route [B,M]."""
    pi = torch.softmax(router_logits, dim=-1)  # [B,5]
    p_route = (pi.unsqueeze(-1) * p_h).sum(dim=1)  # [B,M]
    picks = p_route.topk(k, dim=-1).indices
    return picks, p_route


def router_entropy(router_logits):
    pi = torch.softmax(router_logits, dim=-1)
    p = pi.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    return ent / float(torch.log(torch.tensor(float(pi.size(-1)))))


def winner_accuracy(router_logits, h_star_agg):
    pred = router_logits.argmax(dim=-1)
    return float((pred == h_star_agg).float().mean())


def top2_accuracy(router_logits, h_star_agg):
    top2 = router_logits.topk(2, dim=-1).indices
    hit = (top2 == h_star_agg.unsqueeze(-1)).any(dim=-1)
    return float(hit.float().mean())


def oracle_gap_recovered_pct(fixed_mse, router_mse, oracle_mse):
    """(Fixed - Router) / (Fixed - Oracle) * 100. Clamped to avoid
    division blowups when Fixed ~= Oracle."""
    denom = fixed_mse - oracle_mse
    if abs(denom) < 1e-12:
        return float('nan')
    return (fixed_mse - router_mse) / denom * 100.0
