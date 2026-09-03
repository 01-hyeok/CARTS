"""Selection-rule intervention over one fixed candidate support.

The question this exists to answer is causal: holding the Stage-2 model, the
base forecaster, the gate, the weighting rule and the candidate support all
fixed, does swapping *which ten candidates* go in change the forecast?

Everything here selects indices. Nothing here weights, aggregates, or forecasts
-- that stays in Stage-2 exactly as it already is, which is what keeps the arms
comparable. The selectors themselves are the ones already used by the Stage-1
oracle diagnostic; they are imported rather than reimplemented so that a change
to one cannot silently make the two disagree.

Support definition: `P100` is the top-`pool_m` of the *cosine* score under the
candidate validity mask. It is not a neutral pool -- it is cosine-induced -- so
a result inside it bounds what a selector could do given what cosine offered,
and must never be reported as a full-memory oracle.
"""

import torch

from models.RelationStage1 import (
    select_good_diverse,
    select_greedy_set,
    select_individual_oracle,
    set_utility_metrics,
)

# Arms whose selection reads the query future. R0 does not.
#   R2-U  uniform-mean set oracle       (the aggregate an unweighted mean gives)
#   R2-W  weighted set oracle           (the aggregate Stage-2's softmax gives)
#   R2-target / R2-relation             kept as aliases for the space the greedy
#                                       runs in; identical under a self-only graph
ORACLE_ARMS = ('R1', 'R2-U', 'R2-W', 'R2-target', 'R2-relation', 'R3')
ALL_ARMS = ('R0',) + ORACLE_ARMS

# Arms run on GPU. R2-relation is excluded when the relation space is the target
# space duplicated -- there it is the same experiment as R2-target, and is kept
# only as a unit-test invariant.
DEFAULT_ARMS = ('R0', 'R1', 'R2-U', 'R2-W', 'R3')

_GOOD_N = 30


def build_common_support(cosine_scores, valid_mask, pool_m, k):
    """The fixed candidate support every arm selects inside.

    Returns [B, P] global candidate indices and [B, P] validity. Width is
    clamped to the bank size and floored at `k` so a selector can always fill
    its Top-K.
    """
    if cosine_scores.shape != valid_mask.shape:
        raise ValueError(
            f'cosine_scores {tuple(cosine_scores.shape)} and valid_mask '
            f'{tuple(valid_mask.shape)} must agree'
        )
    neg = torch.finfo(cosine_scores.dtype).min / 4
    masked = cosine_scores.masked_fill(~valid_mask, neg)
    width = max(int(k), min(int(pool_m), masked.size(-1)))
    pool_idx = masked.topk(width, dim=-1).indices
    pool_valid = valid_mask.gather(1, pool_idx)
    return pool_idx, pool_valid


def _mask_invalid(distance, pool_valid):
    """Invalid support slots must never be selectable."""
    return distance.masked_fill(~pool_valid, float('inf'))


def select_within_support(arm, pool_idx, pool_valid, learned_scores,
                          target_futures, relation_futures,
                          query_target, query_relation, k, tau=0.1):
    """Local (within-support) Top-K indices for one arm.

    Args:
        pool_idx: [B, P] global candidate ids forming the support.
        pool_valid: [B, P] which support slots are real.
        learned_scores: [B, P] the live Stage-2 scorer's score on the support.
        target_futures: [B, P, H] candidate target-channel futures.
        relation_futures: [B, P, 2H] candidate [target||source] futures.
        query_target: [B, H]; query_relation: [B, 2H].
    Returns [B, k] indices *into the support*, not global ids.
    """
    if arm not in ALL_ARMS:
        raise ValueError(f'unknown arm {arm!r}; expected one of {ALL_ARMS}')

    if arm == 'R0':
        neg = torch.finfo(learned_scores.dtype).min / 4
        scores = learned_scores.masked_fill(~pool_valid, neg)
        return scores.topk(k, dim=-1).indices

    if arm in ('R2-U', 'R2-target'):
        return select_greedy_set(
            _sentinel_invalid(target_futures, pool_valid), query_target, k)
    if arm == 'R2-relation':
        return select_greedy_set(
            _sentinel_invalid(relation_futures, pool_valid), query_relation, k)
    if arm == 'R2-W':
        return select_greedy_weighted_set(
            target_futures, query_target, learned_scores, pool_valid, k, tau)

    # R1 and R3 rank by individual distance, in the space Stage-2's own oracle
    # uses, so R1 reproduces the existing `_relation_oracle_topk_candidates`
    # ordering restricted to the support rather than defining a second oracle.
    distance = _mask_invalid(
        (relation_futures - query_relation.unsqueeze(1)).pow(2).mean(-1), pool_valid)
    if arm == 'R1':
        return select_individual_oracle(distance, k)
    # An infinite distance keeps invalid rows out of the "good" shortlist only
    # while the shortlist is smaller than the valid count. Past that they enter
    # it, and the diversity step actively *seeks* whatever is furthest away --
    # so an invalid row is exactly what it would reach for. Centring them makes
    # them the least diverse thing in the pool instead.
    return select_good_diverse(
        _centre_invalid(relation_futures, pool_valid), distance, k, _GOOD_N)


def select_greedy_weighted_set(futures, query_future, scores, pool_valid, k, tau,
                               eps=1e-12):
    """Greedy set selection against the aggregate Stage-2 actually forms.

    `select_greedy_set` minimises the error of the *uniform* mean, but Stage-2
    never takes a uniform mean -- it weights the selected Top-K by
    softmax(score/tau). A set that is good on average can be poor once those
    weights land, so this arm optimises the quantity the forecast is built from.

    The weights are re-derived over the whole selected set at every step, which
    is what makes this different from uniform greedy: adding a candidate with a
    high score does not merely add a term, it *dilutes* every weight already
    assigned. Keeping the earlier weights fixed and appending a new one would
    optimise a softmax that Stage-2 will never compute.

    Selection reads the query future, so this is a diagnostic oracle and an
    upper bound -- never an inference rule.

    The softmax over a trial set S+{c} has a closed form that avoids ever
    materialising one weight vector per candidate. With e_i = exp(s_i/tau),
    Z_S = sum_{i in S} e_i and N_S = sum_{i in S} e_i y_i,

        aggregate(S + {c}) = (N_S + e_c y_c) / (Z_S + e_c)

    so a step costs O(P*D) instead of O(P^2*D). That is what makes the same
    selector usable over a full memory bank of tens of thousands of candidates
    rather than only inside a hundred-wide support.
    """
    bsz, pool, dim = futures.shape
    y = futures.float()
    q = query_future.float()
    s = scores.float().masked_fill(~pool_valid, float('-inf'))
    # Shift by the row max before exponentiating; the shift cancels in the ratio
    # and keeps exp() away from overflow when tau is small.
    e = torch.exp((s - s.max(dim=-1, keepdim=True).values) / float(tau))
    e = torch.where(pool_valid, e, torch.zeros_like(e))

    taken = torch.zeros(bsz, pool, dtype=torch.bool, device=y.device)
    numerator = torch.zeros(bsz, dim, device=y.device)      # N_S
    denominator = torch.zeros(bsz, 1, device=y.device)      # Z_S
    picks = []
    for _ in range(min(k, pool)):
        trial_num = numerator.unsqueeze(1) + e.unsqueeze(-1) * y   # [B, P, D]
        trial_den = (denominator + e).unsqueeze(-1).clamp_min(eps)
        err = (trial_num / trial_den - q.unsqueeze(1)).pow(2).mean(-1)
        err = err.masked_fill(taken | ~pool_valid, float('inf'))
        nxt = err.argmin(dim=-1, keepdim=True)
        taken.scatter_(1, nxt, True)
        picks.append(nxt)
        e_sel = e.gather(1, nxt)
        numerator = numerator + e_sel * torch.gather(
            y, 1, nxt.unsqueeze(-1).expand(-1, -1, dim)).squeeze(1)
        denominator = denominator + e_sel
    return torch.cat(picks, dim=-1)


_SENTINEL = 1.0e6


def _sentinel_invalid(futures, pool_valid):
    """Push invalid rows far away so greedy selection can never take one.

    Greedy set selection has no natural -inf: it minimises a distance between a
    running mean and the target, so an invalid row has to be neutralised in the
    value space rather than the score space. Zeroing is not enough -- a zero row
    is a perfectly ordinary candidate that drags the mean toward the origin -- so
    invalid rows are moved somewhere no running mean would ever want to go.
    """
    return torch.where(
        pool_valid.unsqueeze(-1), futures, futures.new_full((), _SENTINEL))


def _centre_invalid(futures, pool_valid, eps=1e-12):
    """Replace invalid rows with the mean of the valid ones.

    The opposite treatment to `_sentinel_invalid`, for the opposite selector:
    greedy set selection is repelled by distance, so invalid rows are pushed
    far away; diversity selection is *attracted* to it, so they are pulled to
    the centre where nothing will ever pick them as the most distant candidate.
    """
    w = pool_valid.unsqueeze(-1).to(futures.dtype)
    centre = (futures * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(eps)
    return torch.where(pool_valid.unsqueeze(-1), futures, centre.expand_as(futures))


def to_global(pool_idx, local_idx):
    """Support-local indices -> global candidate ids."""
    return pool_idx.gather(1, local_idx)


def uniform_metrics(selected_futures, query_future):
    """(I, A_uniform, V_uniform, residual) with I = A + V by construction.

    Computed in float64. The identity is exact in real arithmetic, so the
    residual is a pure numerical-error readout and is used as a correctness
    gate -- which means it must not be dominated by the precision of the check
    itself. `set_utility_metrics` casts to float32, which is ample at Stage-1
    scale but leaves ~2e-4 of residual once a full-memory arm selects
    candidates whose individual error is an order of magnitude larger, and that
    would trip the gate on rounding rather than on a real defect.

    `test_uniform_metrics_matches_the_stage1_helper` pins these against the
    shared Stage-1 implementation so the two cannot drift apart.
    """
    y = selected_futures.double()
    q = query_future.double()
    mean = y.mean(dim=1)
    individual = ((y - q.unsqueeze(1)) ** 2).mean(-1).mean(-1)
    aggregate = ((mean - q) ** 2).mean(-1)
    variance = ((y - mean.unsqueeze(1)) ** 2).mean(-1).mean(-1)
    return individual, aggregate, variance, individual - aggregate - variance


def weighted_metrics(selected_futures, query_future, alpha, eps=1e-12):
    """The same decomposition under Stage-2's own weights.

    A_uniform is not what Stage-2 sees: it weights the Top-K by
    softmax(score/tau). Reported alongside the uniform figures because a
    selection can improve the uniform mean and leave the weighted mean -- the
    quantity that actually reaches the forecast -- unchanged.

    The identity generalises: with weights w summing to one,
        I_w = sum_i w_i ||y_i - q||^2
        A_w = ||sum_i w_i y_i - q||^2
        V_w = sum_i w_i ||y_i - mean_w||^2
    still gives I_w = A_w + V_w, so the same residual check applies.
    """
    alpha = alpha.double()
    selected_futures = selected_futures.double()
    query_future = query_future.double()
    w = alpha / alpha.sum(-1, keepdim=True).clamp_min(eps)
    w3 = w.unsqueeze(-1)
    mean_w = (w3 * selected_futures).sum(1)
    diff = selected_futures - query_future.unsqueeze(1)
    i_w = (w * diff.pow(2).mean(-1)).sum(-1)
    a_w = (mean_w - query_future).pow(2).mean(-1)
    spread = selected_futures - mean_w.unsqueeze(1)
    v_w = (w * spread.pow(2).mean(-1)).sum(-1)
    return i_w, a_w, v_w, i_w - a_w - v_w


def overlap(idx_a, idx_b):
    """Fraction of shared candidates per query, [B]."""
    if idx_a.shape != idx_b.shape:
        raise ValueError(
            f'overlap needs equal shapes, got {tuple(idx_a.shape)} and {tuple(idx_b.shape)}')
    match = (idx_a.unsqueeze(-1) == idx_b.unsqueeze(-2)).any(-1)
    return match.float().mean(-1)


def relation_equals_target(source_channels_per_target):
    """Is [target||source] provably the target space duplicated?

    True when every target's only source is itself, which makes R2-relation and
    R2-target the same experiment. Reported rather than silently accepted.
    """
    return all(list(sources) == [target]
               for target, sources in enumerate(source_channels_per_target))
