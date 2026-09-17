"""EXP-ORACLE-WCE-CONTROL01 -- shared WCE step-loss, one function used by
BOTH `individual_onpolicy_cosine_wce` and `set_onpolicy_cosine_wce` (spec
section 5: "두 arm은 동일한 WCE 함수를 공유해야 한다... arm별 WCE 코드를
복사해서 따로 구현하지 마라").

Built entirely from the two existing `models/RelationStage1.py` primitives
the spec names explicitly (`prepare_topk_coverage_targets`,
`weighted_topk_listwise_ce`), not from `scripts/train_onpolicy_rankloss01.py`'s
`_wce_step_loss` convenience wrapper -- that wrapper was audited during
this experiment's implementation and found to NOT match the spec's exact
formula: it never divides the student logits `u_hat` by any temperature
before `log_softmax` (only the teacher weighting uses `tau_teacher`),
whereas the spec's `p_theta(i) = softmax(s_theta(i)/tau_S)` requires a
student temperature too. Confirmed via `tests/test_oracle_utility_wce.py::
test_m1_equals_hard_choice_ce` (M=1 WCE must degenerate to Hard Choice
CE bit-exactly; it did not, against `_wce_step_loss`, until this wrapper's
explicit `/tau` division was added).

`tau` here plays BOTH roles (`tau_S` for the student softmax, `tau_T` for
the teacher softmax) -- the SAME `tau_choice` Hard Choice CE already uses
for this arm, never a new/unapproved temperature (spec section 6:
"temperature 임의 변경 금지").
"""
import torch
import torch.nn.functional as F

from models.RelationStage1 import prepare_topk_coverage_targets, weighted_topk_listwise_ce


def wce_step_loss(u_hat, u_target, valid_now, tau, top_k_oracle=10):
    """u_target: higher-is-better utility, SAME convention
    `oracle_choice_step_loss` uses (so Individual's `individual_utility`
    and Set's `greedy_set_utility` outputs are passed in UNCHANGED --
    distance = -u_target is computed here, once, for both Oracles).

    Returns (loss, diag) where diag merges weighted_topk_listwise_ce's own
    metrics with the raw distance the teacher was built from, for
    diagnostics only (no gradient path)."""
    distance = -u_target
    targets = prepare_topk_coverage_targets(distance, valid_now, top_k_oracle)
    logits = u_hat.masked_fill(~valid_now, torch.finfo(u_hat.dtype).min / 4) / float(tau)
    log_prob = F.log_softmax(logits, dim=-1)
    loss, metrics = weighted_topk_listwise_ce(log_prob, targets, tau_teacher=float(tau))
    diag = {k: float(v) for k, v in metrics.items()}
    return loss, diag
