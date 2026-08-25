"""Query-conditioned reranker over a frozen retriever's shortlist.

The global bi-encoder keeps its job -- producing a broad Top-M neighbourhood --
and this scores (query, candidate) pairs inside that shortlist only. A pair
scorer is not bound by the constraint a retriever is under: it never has to
order the whole memory bank with one inner product, so it can express
interactions a global geometry cannot.

    past_pair       r(q,k) = MLP([z_q, z_k, |z_q-z_k|, z_q*z_k, s_ret])
    residual_aware  r(q,k) = MLP([z_q, z_k, e(R_k), |z_q-z_k|, z_q*z_k, s_ret])

z_q and z_k are the frozen Stage-1 embeddings, so a difference between the two
arms is a difference in *candidate value information*, not in encoder capacity.
R_k is the candidate's own historical residual: memory-side, observable at
inference. The query's future, its residual and the utility labels never enter
`forward` -- they build targets and metrics only.
"""

import torch
import torch.nn as nn

ARMS = ('past_pair', 'residual_aware')


class UtilityReranker(nn.Module):
    def __init__(self, dim, horizon=0, residual_dim=64, hidden=None,
                 dropout=0.1, use_retriever_score=True):
        super().__init__()
        hidden = hidden or 2 * dim
        self.use_retriever_score = bool(use_retriever_score)
        self.residual_proj = (
            nn.Sequential(nn.Linear(horizon, residual_dim), nn.GELU())
            if horizon else None
        )
        width = 4 * dim + (residual_dim if horizon else 0) + int(self.use_retriever_score)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @property
    def arm(self):
        return 'residual_aware' if self.residual_proj is not None else 'past_pair'

    def forward(self, z_q, z_k, retriever_score=None, candidate_residual=None):
        """z_q [B, 1, d] or [B, M, d], z_k [B, M, d] -> scores [B, M]."""
        if z_q.size(1) == 1:
            z_q = z_q.expand_as(z_k)
        if z_q.shape != z_k.shape:
            raise ValueError(
                f'query/candidate embedding mismatch: {tuple(z_q.shape)} vs {tuple(z_k.shape)}'
            )
        parts = [z_q, z_k, (z_q - z_k).abs(), z_q * z_k]
        if self.residual_proj is not None:
            if candidate_residual is None:
                raise ValueError('this arm was built with a candidate-residual branch')
            if candidate_residual.shape[:2] != z_k.shape[:2]:
                raise ValueError(
                    f'residual shape {tuple(candidate_residual.shape)} does not match '
                    f'candidates {tuple(z_k.shape[:2])}'
                )
            parts.append(self.residual_proj(candidate_residual))
        if self.use_retriever_score:
            if retriever_score is None:
                raise ValueError('this arm consumes the retriever score')
            parts.append(retriever_score.unsqueeze(-1))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


def build_reranker(arm, dim, horizon, **kwargs):
    if arm not in ARMS:
        raise ValueError(f'unknown reranker arm: {arm}')
    return UtilityReranker(dim, horizon if arm == 'residual_aware' else 0, **kwargs)


FEATURE_LADDER = {
    # arm: (candidate residual, predicted query residual, true query residual,
    #        query future), deployable
    'A_past':               ((False, False, False, False), True),
    'B_cand_residual':      ((True,  False, False, False), True),
    'C_pred_query_residual':((True,  True,  False, False), True),
    'D_true_query_residual':((True,  False, True,  False), False),
    'E_query_future':       ((True,  False, False, True),  False),
}


class LadderReranker(nn.Module):
    """Same backbone for every rung; only the feature set grows.

    Each optional group is projected to the same width `d` before the concat, so
    an arm that sees more information does not also get a wider first layer than
    it needs. Parameter counts still differ slightly and are reported, but the
    interaction capacity is held as close to constant as concatenation allows.

    Deployability is a property of the *arm*, enforced by the caller: groups D
    and E read the query's own future and cannot be shipped. The module keeps
    them behind explicit flags so an accidental read is impossible.
    """

    def __init__(self, dim, horizon, use_candidate_residual=False,
                 use_predicted_query_residual=False,
                 use_true_query_residual=False, use_query_future=False,
                 hidden=None, dropout=0.1, use_retriever_score=True):
        super().__init__()
        hidden = hidden or 2 * dim
        self.use_retriever_score = bool(use_retriever_score)
        make = lambda width: nn.Sequential(nn.Linear(width, dim), nn.GELU())
        self.candidate_residual = make(horizon) if use_candidate_residual else None
        self.predicted_query_residual = make(horizon) if use_predicted_query_residual else None
        self.true_query_residual = make(horizon) if use_true_query_residual else None
        self.query_future = make(horizon) if use_query_future else None

        groups = 4 + sum(branch is not None for branch in (
            self.candidate_residual, self.predicted_query_residual,
            self.true_query_residual, self.query_future))
        width = groups * dim + int(self.use_retriever_score)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_q, z_k, retriever_score=None, candidate_residual=None,
                predicted_query_residual=None, true_query_residual=None,
                query_future=None):
        if z_q.size(1) == 1:
            z_q = z_q.expand_as(z_k)
        if z_q.shape != z_k.shape:
            raise ValueError(
                f'query/candidate embedding mismatch: {tuple(z_q.shape)} vs {tuple(z_k.shape)}')
        parts = [z_q, z_k, (z_q - z_k).abs(), z_q * z_k]

        def add(branch, value, name, per_candidate):
            if branch is None:
                return
            if value is None:
                raise ValueError(f'this arm was built with the {name} branch')
            if per_candidate:
                if value.shape[:2] != z_k.shape[:2]:
                    raise ValueError(
                        f'{name} shape {tuple(value.shape)} does not match '
                        f'candidates {tuple(z_k.shape[:2])}')
                parts.append(branch(value))
            else:
                if value.dim() != 2 or value.size(0) != z_k.size(0):
                    raise ValueError(
                        f'{name} must be [B, horizon], got {tuple(value.shape)}')
                parts.append(branch(value).unsqueeze(1).expand(-1, z_k.size(1), -1))

        add(self.candidate_residual, candidate_residual, 'candidate_residual', True)
        add(self.predicted_query_residual, predicted_query_residual,
            'predicted_query_residual', False)
        add(self.true_query_residual, true_query_residual, 'true_query_residual', False)
        add(self.query_future, query_future, 'query_future', False)

        if self.use_retriever_score:
            if retriever_score is None:
                raise ValueError('this arm consumes the retriever score')
            parts.append(retriever_score.unsqueeze(-1))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class QueryResidualPredictor(nn.Module):
    """Arm C's separate model: query past -> its own base-forecast error.

    Trained on the train split only and frozen for val/test, so arm C stays
    deployable: it never sees a query future, only a prediction of one.
    """

    def __init__(self, seq_len, pred_len, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, pred_len),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
