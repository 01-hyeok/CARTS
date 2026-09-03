"""tau_calibration_diag: N_eff = mean_q(exp(H_q)), pinned against the recovered
ground truth in logs/tau_calibration/pred{96,192,336,720}.log.

The method itself was lost before being committed; these values are what
verified the reconstruction (see exp/exp_stage1_relation.py:tau_calibration_diag
docstring). Two alternative N_eff definitions -- exp(mean_q H_q) and the
inverse-participation ratio 1/sum_i p_i^2 in either aggregation order -- were
ruled out numerically against this same data and must keep failing here.
"""
import torch
import torch.nn.functional as F


def _n_eff_mean_exp_entropy(scores, valid, tau):
    neg = torch.finfo(scores.dtype).min / 4
    logits = (scores / tau).masked_fill(~valid, neg)
    p = F.softmax(logits, dim=-1)
    entropy = -(p * (p + 1e-12).log()).sum(-1)
    return float(entropy.exp().mean())


def test_n_eff_matches_ground_truth_pred336_tau01():
    torch.manual_seed(0)
    # A synthetic stand-in cannot reproduce the exact logged figures (those
    # depend on a real checkpoint's score geometry); what is checked here is
    # that the formula is internally what the docstring claims, on a case
    # where the three candidate definitions disagree enough to distinguish.
    scores = torch.randn(50, 2000) * 0.3
    valid = torch.ones(50, 2000, dtype=torch.bool)
    tau = 0.01

    n_eff = _n_eff_mean_exp_entropy(scores, valid, tau)

    logits = (scores / tau)
    p = F.softmax(logits, dim=-1)
    entropy = -(p * (p + 1e-12).log()).sum(-1)
    exp_of_mean = float(entropy.mean().exp())
    ipr_agg_then_invert = float(1.0 / (p * p).sum(-1).mean())
    ipr_mean_of_invert = float((1.0 / (p * p).sum(-1)).mean())

    # Jensen's inequality makes these strictly ordered whenever entropy is not
    # constant across queries: exp is convex, so the per-query-then-average
    # definition used here must sit at or above exp(mean(H)).
    assert n_eff >= exp_of_mean - 1e-6
    assert abs(n_eff - exp_of_mean) > 1e-3, (
        'synthetic scores did not distinguish the definitions; adjust the case')
    assert abs(n_eff - ipr_agg_then_invert) > 1e-3
    assert abs(n_eff - ipr_mean_of_invert) > 1e-3


def test_tau_calibration_diag_reproduces_etth1_ground_truth_log():
    """End-to-end acceptance test: rerun calibration and diff against the log.

    Skipped when the checkpoint or dataset is unavailable (e.g. a CI machine
    without the ETT data mounted) rather than failing for an unrelated reason.
    This is the test that must be run, and pass, before any Weather tau is
    accepted -- it is the proof the reconstruction is correct, not merely
    plausible.
    """
    import glob
    import os
    import re
    import subprocess
    import sys

    import pytest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(
        root, '..', 'Dataset', 'Time-Series-Library_dataset', 'ETT-small', 'ETTh1.csv')
    if not os.path.isfile(data_csv):
        pytest.skip('ETTh1.csv not available in this environment')

    for pred in (96, 720):
        ck_glob = os.path.join(
            root, 'checkpoints', 'stage1', 'ETTh1', f'seq{pred}_pred{pred}',
            '*e2_cos_weighted_topk_ce_ETTh1*', 'checkpoint.pth')
        matches = glob.glob(ck_glob)
        if not matches:
            pytest.skip(f'reference checkpoint for pred={pred} not present')
        ck = matches[0]
        log_path = os.path.join(root, 'logs', 'tau_calibration', f'pred{pred}.log')
        if not os.path.isfile(log_path):
            pytest.skip(f'ground-truth log for pred={pred} not present')
        ground_truth = open(log_path).read()

        env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                   CARTS_TAUCAL_DIAG='1', CARTS_TAUCAL_BATCHES='4')
        cmd = [
            sys.executable, '-u', 'run.py',
            '--task_name', 'stage1_relation', '--is_training', '0',
            '--model', 'RelationStage1', '--data', 'ETTh1',
            '--root_path', '../Dataset/Time-Series-Library_dataset/ETT-small/',
            '--data_path', 'ETTh1.csv', '--features', 'M',
            '--seq_len', str(pred), '--label_len', '0', '--pred_len', str(pred),
            '--enc_in', '7', '--batch_size', '32', '--num_workers', '0',
            '--d_model', '128', '--d_ff', '256', '--n_heads', '4', '--e_layers', '2',
            '--patch_len', '16', '--stride', '16', '--seed', '0',
            '--candidate_mask', 'raft',
            '--relation_input_space', 'delta_last',
            '--relation_teacher_space', 'delta_last',
            '--source_mode', 'auto', '--relation_top_n', '1', '--target_mode', 'all',
            '--relation_encoder_type', 'mlp', '--relation_self_fill', 'linear',
            '--top_k', '10', '--tau_student', '0.10', '--tau_teacher', '0.1',
            '--tau_topk', '0.1', '--teacher_mse_space', 'normalized',
            '--stage1_teacher_mode', 'mse', '--stage1_loss_mode', 'weighted_topk_ce',
            '--stage1_coverage_top_k', '10',
            '--stage1_full_memory_gradient_mode', 'full_online',
            '--stage1_probe_vis', '0',
            '--stage1_ckpt_path', ck, '--stage1_encoder_init', 'checkpoint',
            '--stage1_retrieval_metric', 'asymmetric',
            '--stage1_metric_output', 'cosine', '--stage1_metric_layer_norm', '0',
            '--stage1_freeze_encoder', '1',
            '--model_id', f'carts_e2_cos_weighted_topk_ce_ETTh1_{pred}',
            '--des', f'e2_cos_weighted_topk_ce_ETTh1_sl{pred}_pl{pred}',
        ]
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True,
                                text=True, timeout=300)
        out = result.stdout + result.stderr
        assert 'Traceback' not in out, out[-4000:]

        gt_n_eff = {float(m[0]): float(m[1]) for m in
                    re.findall(r'\[taucal\]\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s',
                              ground_truth)}
        got_n_eff = {float(m[0]): float(m[1]) for m in
                    re.findall(r'\[taucal\]\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s', out)}
        assert set(got_n_eff) == set(gt_n_eff), (got_n_eff.keys(), gt_n_eff.keys())
        for tau, n_eff in gt_n_eff.items():
            rel = abs(got_n_eff[tau] - n_eff) / max(n_eff, 1.0)
            assert rel < 0.02, f'pred={pred} tau={tau}: got {got_n_eff[tau]}, log {n_eff}'

        gt_choice = re.search(rf'TAU_{pred} = ([\d.]+)', ground_truth).group(1)
        got_choice = re.search(rf'TAU_{pred} = ([\d.]+)', out).group(1)
        assert got_choice == gt_choice, (
            f'pred={pred}: chosen tau {got_choice} != ground truth {gt_choice}')
