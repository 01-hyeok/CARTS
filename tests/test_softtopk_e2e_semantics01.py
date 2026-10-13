"""Spec section 28, items 13 and 14 -- semantic (not just unit-level)
checks on the three E2E trainer scripts themselves."""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _assign_targets_for_call(tree, func_name):
    """Find `X = func_name(...)` assignments anywhere in the module and
    return the list of (lhs_name, call_node) pairs."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            fname = getattr(call.func, 'id', None) or getattr(call.func, 'attr', None)
            if fname == func_name and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out.append(node.targets[0].id)
    return out


def test_v_family_uses_mean_of_softmaxes_not_logit_averaging():
    """Spec 28-13: V2/V5 Mean-Mixture must be mean_h softmax(s_h/tau),
    never softmax(mean_h s_h/tau) (logit averaging ranks differently).
    Static check: `p_heads = torch.softmax(...)` must appear BEFORE the
    `.mean(dim=1)` call that produces p_mix -- i.e. softmax happens
    per-head first, then the mean is taken over already-exponentiated
    probabilities."""
    src = (REPO_ROOT / 'scripts' / 'train_softtopk_e2e_v01.py').read_text()
    softmax_idx = src.index('p_heads = torch.softmax')
    mean_idx = src.index('p_mix = p_heads.mean(dim=1)')
    assert softmax_idx < mean_idx, (
        'p_mix must be computed as mean-of-softmaxes (softmax per head, then averaged), '
        'not softmax-of-mean-logits')
    # Also confirm no logit-averaging shortcut exists anywhere in the file.
    assert 'scores.mean(dim=1)' not in src, 'found a logit-averaging shortcut (scores.mean before softmax)'


def test_hard_forecast_branch_never_uses_future_aware_winner():
    """Spec 28-14: the Hard trainer's forecasting/fusion branch must
    build its Soft Top-K distribution from `p_mix` (mean-of-softmaxes,
    future-blind), never from `winner_idx` (the future-aware winner
    used only for the UNCHANGED retrieval loss)."""
    src = (REPO_ROOT / 'scripts' / 'train_softtopk_e2e_hard01.py').read_text()
    # winner_idx must be consumed ONLY by hard_expert_loss, never by
    # soft_topk_forecast / fuse_noqa / fuse_qa / stable_topk_indices.
    forecast_fn_calls = ['soft_topk_forecast(', 'fuse_noqa(', 'fuse_qa(', 'stable_topk_indices(']
    for line in src.splitlines():
        if 'winner_idx' in line:
            for forbidden in forecast_fn_calls:
                assert forbidden not in line, (
                    f'winner_idx must never feed the forecasting branch, but appears on the same '
                    f'line as {forbidden!r}: {line.strip()}')
    assert 'hard_expert_loss(kl_vals, winner_idx)' in src, (
        'winner_idx should still be used for the UNCHANGED retrieval loss')
    assert 'p_mix = p_heads.mean(dim=1)' in src, (
        'forecast branch must use p_mix (mean-of-softmaxes), not the winner')


def test_joint_uses_router_mixed_distribution_for_soft_topk():
    src = (REPO_ROOT / 'scripts' / 'train_softtopk_e2e_joint01.py').read_text()
    assert 'r_c, weights = soft_topk_forecast(p_r, picks, y_sel)' in src
    assert 'p_r = (pi.unsqueeze(-1) * p_h).sum(dim=1)' in src
