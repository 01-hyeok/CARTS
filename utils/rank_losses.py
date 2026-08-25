"""Pairwise ranking losses that ask for score *separation*, not just order.

The bottleneck study left one link unexplained: Stage-1 ranking improved, the
forecast did not. The direct evidence was that Stage-2's Top-K weights are
uniform -- `topk_weight_entropy` sits at ln(10) because the top candidates'
cosine scores differ by ~0.004 and `tau_topk=0.10` shrinks that to a 0.04 logit
gap. A retriever can therefore get the order right and still hand Stage-2 a flat
distribution it cannot act on.

KL distillation does not fix this: matching a teacher distribution constrains
relative probabilities, not the absolute score gap that survives division by tau.
A margin does constrain it, which is why the margin arm is the main one here.

Everything is defined on a mined candidate subset rather than all N^2 pairs, and
all of it is pure tensor arithmetic over scores -- no forecast is reconstructed
anywhere in this file.
"""

import torch

EPS = 1e-8
MODES = ('none', 'ranknet', 'weighted_ranknet', 'margin', 'adaptive_margin')


def mine_ranking_candidates(teacher_scores, student_scores, valid_mask,
                            top_p=10, hard_negatives=30, random_negatives=10,
                            generator=None):
    """Candidates worth forming pairs from, per query.

    Three groups, because a loss trained only on the model's own Top-K can never
    learn that something outside it should have been in:

      positives       the teacher's best -- what should be retrieved
      hard negatives  what the student ranks highly and the teacher does not
      random          a weak background so the geometry does not collapse onto
                      the hard cases alone

    Returns (indices [B, M], counts dict). Padding, when a query has too few
    valid candidates, repeats its best valid candidate; such pairs are ties and
    the losses drop ties.
    """
    if teacher_scores.shape != student_scores.shape or teacher_scores.shape != valid_mask.shape:
        raise ValueError(
            f'teacher {tuple(teacher_scores.shape)}, student {tuple(student_scores.shape)} '
            f'and mask {tuple(valid_mask.shape)} must share shape'
        )
    floor = torch.finfo(student_scores.dtype).min
    num_candidates = student_scores.size(-1)
    width_p = max(0, min(top_p, num_candidates))
    width_h = max(0, min(hard_negatives, num_candidates))

    teacher_masked = teacher_scores.detach().masked_fill(~valid_mask, floor)
    student_masked = student_scores.detach().masked_fill(~valid_mask, floor)

    positives = teacher_masked.topk(width_p, dim=-1).indices if width_p else None
    picked = torch.zeros_like(valid_mask)
    if positives is not None:
        picked.scatter_(1, positives, True)

    # Hard negatives: high student score, not already a teacher positive.
    hard = None
    if width_h:
        hard_pool = student_masked.masked_fill(picked, floor)
        hard = hard_pool.topk(width_h, dim=-1).indices
        picked.scatter_(1, hard, True)

    parts = [part for part in (positives, hard) if part is not None]
    if random_negatives > 0:
        weights = (valid_mask & ~picked).float()
        # A query whose valid pool is exhausted falls back to any valid candidate;
        # duplicates become ties, which the losses skip.
        weights = torch.where(weights.sum(-1, keepdim=True) > 0, weights, valid_mask.float())
        count = min(random_negatives, num_candidates)
        parts.append(torch.multinomial(weights + EPS, count, replacement=True,
                                       generator=generator))

    indices = torch.cat(parts, dim=-1)
    counts = {
        'rank_positives': float(width_p),
        'rank_hard_negatives': float(width_h),
        'rank_random_negatives': float(random_negatives if random_negatives > 0 else 0),
        'rank_candidates': float(indices.size(-1)),
    }
    return indices, counts


def _pair_terms(teacher, student, valid, tie_epsilon):
    """Upper-triangular pairs with a definite teacher preference."""
    teacher = teacher.detach().float()
    gap_teacher = teacher.unsqueeze(-1) - teacher.unsqueeze(-2)       # [B, M, M]
    gap_student = student.unsqueeze(-1) - student.unsqueeze(-2)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    upper = torch.triu(torch.ones_like(pair_valid[0]), diagonal=1).bool()
    keep = pair_valid & upper.unsqueeze(0) & (gap_teacher.abs() > tie_epsilon)
    return gap_teacher, gap_student, keep


def topk_spread(student_scores, topk_mask, eps=EPS):
    """Per-query spread of the scores Stage-2 actually weights. [B, 1].

    The audit measured this at 0.0088 while the mined set spanned 0.79, so a
    single absolute margin cannot be right for both. Everything scaled by this
    is detached: it sets how much separation to ask for, and must not itself
    become something the loss can optimise by inflating.
    """
    floor = torch.finfo(student_scores.dtype).min
    scores = student_scores.detach()
    high = scores.masked_fill(~topk_mask, floor).max(-1, keepdim=True).values
    low = scores.masked_fill(~topk_mask, -floor).min(-1, keepdim=True).values
    return (high - low).clamp_min(eps)


def ranking_loss(teacher_scores, student_scores, valid_mask, mode='margin',
                 margin=0.05, tie_epsilon=1e-6, topk_mask=None, gamma=None,
                 margin_mode='absolute', margin_cap=0.2, sigma_mode='fixed'):
    """Pairwise loss over a mined candidate subset. Shapes are [B, M].

    `margin` and `adaptive_margin` are the arms that target score separation:
    they are only satisfied once the student's gap reaches an absolute size, so
    what they buy survives the division by tau that flattens everything else.

    The optional arguments exist because the v1 design missed its own target.
    The audit found only 3.7% of mined pairs sit inside the Top-K that Stage-2
    weights, and they carried 1.9% of the margin loss -- the other 98% went into
    widening pairs already 24x further apart, which a bounded cosine space can
    only answer by collapsing.

      topk_mask    which mined slots are in the model's Top-K
      gamma        L = gamma * mean(inside) + (1 - gamma) * mean(outside),
                   normalised separately so the split does not depend on how
                   many pairs each region happens to contain
      margin_mode  'topk_relative' asks for `margin` times the current Top-K
                   spread instead of an absolute number, capped so the demand
                   cannot run away
      sigma_mode   'topk_relative' scales RankNet's logit by 1 / spread. Without
                   it sigmoid(-a) sits at 0.50 for both a 0.009 gap and a 0.21
                   gap, which is exactly the scale blindness that left RankNet
                   unable to separate anything.
    """
    if mode == 'none':
        return None, {}
    if mode not in MODES:
        raise ValueError(f'Unsupported ranking mode: {mode}; expected one of {MODES}')
    if margin_mode not in ('absolute', 'topk_relative'):
        raise ValueError(f'Unsupported margin mode: {margin_mode}')
    if sigma_mode not in ('fixed', 'topk_relative'):
        raise ValueError(f'Unsupported sigma mode: {sigma_mode}')
    for name, tensor in (('student_scores', student_scores), ('valid_mask', valid_mask)):
        if tensor.shape != teacher_scores.shape:
            raise ValueError(
                f'{name} shape {tuple(tensor.shape)} does not match teacher '
                f'{tuple(teacher_scores.shape)}'
            )
    scoped = topk_mask is not None and gamma is not None
    if (margin_mode == 'topk_relative' or sigma_mode == 'topk_relative') and topk_mask is None:
        raise ValueError('topk_relative scaling needs topk_mask')
    if topk_mask is not None and topk_mask.shape != teacher_scores.shape:
        raise ValueError(
            f'topk_mask shape {tuple(topk_mask.shape)} does not match '
            f'{tuple(teacher_scores.shape)}'
        )

    gap_teacher, gap_student, keep = _pair_terms(
        teacher_scores, student_scores, valid_mask, tie_epsilon)
    if not keep.any():
        return None, {}

    sign = torch.sign(gap_teacher)
    agreement = sign * gap_student                    # > 0 when the order is right
    scale = gap_teacher.abs()

    spread = (
        topk_spread(student_scores, topk_mask & valid_mask)
        if topk_mask is not None else None
    )
    effective_margin = (
        (margin * spread).clamp(max=margin_cap).unsqueeze(-1)
        if margin_mode == 'topk_relative' else
        torch.full_like(agreement[:, :1, :1], float(margin))
    )
    sigma = (1.0 / spread).unsqueeze(-1) if sigma_mode == 'topk_relative' else 1.0

    if mode in ('weighted_ranknet', 'adaptive_margin'):
        # Normalised inside the query so one outlier pair cannot dominate, and so
        # the weight is scale-free across datasets. With a scope split the
        # normaliser is taken per region, or the outside pairs -- 24x wider --
        # would set the scale for the inside ones too.
        regions = [keep]
        if scoped:
            inside = topk_mask.unsqueeze(-1) & topk_mask.unsqueeze(-2) & keep
            regions = [inside, keep & ~inside]
        normalized = torch.zeros_like(scale)
        for region in regions:
            count = region.sum(dim=(-2, -1), keepdim=True).clamp_min(1).float()
            mean_gap = (scale * region).sum(dim=(-2, -1), keepdim=True) / count
            normalized = torch.where(
                region, (scale / mean_gap.clamp_min(EPS)).clamp(max=10.0), normalized)
    else:
        normalized = torch.ones_like(scale)

    if mode == 'ranknet':
        per_pair = torch.nn.functional.softplus(-sigma * agreement)
    elif mode == 'weighted_ranknet':
        per_pair = normalized * torch.nn.functional.softplus(-sigma * agreement)
    elif mode == 'margin':
        per_pair = (effective_margin - agreement).clamp_min(0.0)
    else:
        per_pair = (effective_margin * normalized - agreement).clamp_min(0.0)

    def region_mean(region):
        total = region.float().sum().clamp_min(1.0)
        return (per_pair * region.float()).sum() / total

    if scoped:
        inside = topk_mask.unsqueeze(-1) & topk_mask.unsqueeze(-2) & keep
        outside = keep & ~inside
        loss = gamma * region_mean(inside) + (1.0 - gamma) * region_mean(outside)
    else:
        loss = region_mean(keep)
    if not torch.isfinite(loss):
        return None, {}

    active = keep.float()
    total = active.sum().clamp_min(1.0)
    satisfied = ((agreement >= effective_margin) & keep).float().sum() / total
    metrics = {
        'rank_loss': loss.detach(),
        'rank_pairs': total.detach(),
        'rank_order_accuracy': (((agreement > 0) & keep).float().sum() / total).detach(),
        'rank_margin_satisfied': satisfied.detach(),
        'rank_mean_student_gap': ((gap_student.abs() * active).sum() / total).detach(),
        'rank_mean_teacher_gap': ((scale * active).sum() / total).detach(),
        'rank_effective_margin': effective_margin.detach().mean(),
    }
    if topk_mask is not None:
        inside = topk_mask.unsqueeze(-1) & topk_mask.unsqueeze(-2) & keep
        outside = keep & ~inside
        metrics.update({
            'rank_topk_spread': spread.detach().mean(),
            'rank_pairs_inside_topk': inside.float().sum().detach(),
            'rank_fraction_inside_topk': (inside.float().sum() / total).detach(),
            'rank_loss_inside_topk': region_mean(inside).detach(),
            'rank_loss_outside_topk': region_mean(outside).detach(),
            'rank_loss_share_inside': (
                (per_pair * inside.float()).sum()
                / (per_pair * active).sum().clamp_min(EPS)).detach(),
            'rank_gap_inside_topk': (
                (gap_student.abs() * inside.float()).sum()
                / inside.float().sum().clamp_min(1.0)).detach(),
            'rank_gap_outside_topk': (
                (gap_student.abs() * outside.float()).sum()
                / outside.float().sum().clamp_min(1.0)).detach(),
        })
    return loss, metrics


@torch.no_grad()
def score_geometry(top_scores, top_valid):
    """How much separation the Top-K scores actually carry. [B, K]."""
    floor = torch.finfo(top_scores.dtype).min
    scores = top_scores.masked_fill(~top_valid, floor)
    ordered, _ = scores.sort(dim=-1, descending=True)
    valid_count = top_valid.sum(-1)
    finite = ordered.masked_fill(ordered <= floor / 2, float('nan'))
    last = torch.gather(ordered, 1, (valid_count - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    out = {
        'topk_score_mean': finite.nanmean(),
        'topk_score_std': _nanstd(finite),
        'top1_minus_top10': (ordered[:, 0] - last).mean(),
        'score_range': (ordered[:, 0] - last).mean(),
    }
    if ordered.size(-1) > 1:
        out['top1_minus_top2'] = (ordered[:, 0] - ordered[:, 1]).mean()
    return out


@torch.no_grad()
def weight_geometry(alpha, top_valid):
    """What the Top-K softmax does with those scores. [B, K].

    `effective_k` = exp(entropy): the number of candidates the weighting really
    keeps. At ln(K) it keeps all of them, which is a plain average.
    """
    weights = (alpha * top_valid.float()).float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(EPS)
    entropy = -(weights * (weights + 1e-12).log()).sum(-1)
    count = top_valid.sum(-1).clamp_min(1).float()
    largest = weights.max(-1).values
    smallest = weights.masked_fill(~top_valid, 1.0).min(-1).values
    return {
        'weight_entropy': entropy.mean(),
        'normalized_weight_entropy': (entropy / count.log().clamp_min(EPS)).mean(),
        'effective_k': entropy.exp().mean(),
        'max_weight': largest.mean(),
        'min_weight': smallest.mean(),
        'max_min_weight_ratio': (largest / smallest.clamp_min(EPS)).mean(),
    }


def _nanstd(values):
    mean = values.nanmean()
    centered = (values - mean).square()
    count = (~torch.isnan(values)).float().sum().clamp_min(1.0)
    return (centered.nansum() / count).sqrt()


@torch.no_grad()
def embedding_geometry(z_q, z_k=None):
    """Is the encoder still using its space, or has it collapsed?

    A margin loss on cosine scores can be satisfied two ways: by separating
    candidates, or by degenerating the space until nothing is comparable. The
    smoke run showed exactly that risk -- train-time Top-K separation of 0.068
    alongside a test-time mean similarity of 0.9986 -- so these are logged
    alongside the score geometry rather than after the fact.

    `effective_rank` is exp of the entropy of the normalised eigenvalue spectrum:
    how many directions the embeddings actually occupy.
    """
    embeddings = z_q if z_k is None else torch.cat([z_q, z_k.reshape(-1, z_k.size(-1))], dim=0)
    embeddings = embeddings.float()
    normalized = torch.nn.functional.normalize(embeddings, dim=-1, eps=EPS)
    similarity = torch.matmul(normalized, normalized.transpose(0, 1))
    count = similarity.size(0)
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    pairwise = similarity[off_diagonal]

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    dimension_std = centered.std(dim=0)
    covariance = torch.matmul(centered.transpose(0, 1), centered) / max(count - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)
    spectrum = eigenvalues / eigenvalues.sum().clamp_min(EPS)
    entropy = -(spectrum * (spectrum + 1e-12).log()).sum()
    return {
        'embedding_pairwise_cosine_mean': pairwise.mean(),
        'embedding_pairwise_cosine_std': pairwise.std(),
        'embedding_variance': centered.square().mean(),
        'embedding_dimension_std_mean': dimension_std.mean(),
        'embedding_effective_rank': entropy.exp().float(),
        'embedding_effective_rank_ratio': (entropy.exp().float() / float(embeddings.size(-1))),
        'embedding_dead_dimension_fraction': (dimension_std < 1e-3).float().mean(),
    }
