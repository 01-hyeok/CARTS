#!/usr/bin/env python3
"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase C: fusion ablation (C1),
retriever-checkpoint control (C2), oracle headroom (C3), quality
calibration (C4), shuffled-feature control (C5). CPU-only, operates on
the raw tensors `extract_fusion_diagnostics_raw01.py` already saved
(never touches a GPU, never touches TEST -- VALIDATION only, further
split chronologically into Val-Select/Val-Holdout).

For EACH of the 5 representative (cell, method) pairs, and for EACH
retriever basis (`noqa` = the NOQA-trained retriever's own R/weights/
features, `qa` = the QA-trained retriever's own) -- per spec C1+C2,
the fusion ablation is run TWICE, once per retriever checkpoint, to
separate "which fusion function" from "which retriever" as causes of
any MSE difference:

  F0 Base-only            Y_hat = B
  F1 Global Lambda        one scalar, closed-form-fit on Val-Select
  F2 Per-channel Lambda   one scalar per channel, closed-form-fit on Val-Select
  F3 Constant Mean-QA     a QualityGate fit on Val-Select, then its
                          Val-Select MEAN prediction frozen as a constant
  F4 Dynamic QA           the SAME fit QualityGate, applied per-query
                          (on the `qa` basis, F4 can also use the REAL
                          E2E-trained gate's own lambda -- reported
                          alongside the Val-Select-refit one as
                          `f4_e2e_trained`)

Also: C3 oracle headroom (Global/Per-channel/Dynamic oracle, fit
directly on Val-Holdout's own Y -- an upper bound, never a generalizing
fit), C4 quality calibration (Pearson/Spearman(lambda, retrieval
advantage), lambda-bin breakdown), C5 shuffled-feature control (permute
qa_features_raw within-channel on Val-Holdout only, before the TRAINED
gate, recompute MSE -- Original-QA-MSE < Shuffled-QA-MSE is the
predicted direction if the gate uses real per-query signal).
"""
import argparse
import json
from pathlib import Path

import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.quality_aware_gate01 import QA_FEATURE_DIM, QualityGate, TrainFeatureNormalizer, validation_oracle_lambda
from utils.validation_diagnostics01 import quality_calibration_report, retrieval_advantage
from utils.validation_split01 import chronological_select_holdout_indices

PAIRS = [
    ('ETTh1_192', 'V5', 192), ('ETTh1_336', 'Hard', 336), ('Weather_720', 'V5', 720),
    ('Weather_336', 'Joint', 336), ('Solar_96', 'V5', 96),
]


def split_mask(window_start_idx, seq_len, pred_len, select_fraction=0.5):
    uniq = sorted(set(window_start_idx.tolist()))
    n = len(uniq)
    sel_idx, hold_idx = chronological_select_holdout_indices(n, seq_len, pred_len, select_fraction)
    sel_set = {uniq[i] for i in sel_idx}
    hold_set = {uniq[i] for i in hold_idx}
    starts = window_start_idx.tolist()
    is_sel = torch.tensor([s in sel_set for s in starts])
    is_hold = torch.tensor([s in hold_set for s in starts])
    return is_sel, is_hold


def mse(y_hat, y):
    return float(((y_hat - y) ** 2).mean())


def fit_global_lambda(B, R, Y):
    r_minus_b = R - B
    y_minus_b = Y - B
    num = (r_minus_b * y_minus_b).sum()
    den = (r_minus_b ** 2).sum().clamp_min(1e-12)
    return float((num / den).clamp(0.0, 1.0))


def fit_per_channel_lambda(B, R, Y, channel):
    lam = {}
    for c in channel.unique().tolist():
        m = channel == c
        lam[c] = fit_global_lambda(B[m], R[m], Y[m])
    return lam


def apply_per_channel_lambda(lam_dict, B, R, channel):
    lam_vec = torch.zeros(B.size(0))
    for c, lam in lam_dict.items():
        lam_vec[channel == c] = lam
    return B + lam_vec.unsqueeze(-1) * (R - B)


def fit_quality_gate(features, B, R, Y, epochs=300, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    normalizer = TrainFeatureNormalizer().fit(features)
    gate = QualityGate(input_dim=QA_FEATURE_DIM)
    feats_n = normalizer.transform(features)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        lam = gate(feats_n)
        y_hat = B + lam.unsqueeze(-1) * (R - B)
        loss = ((y_hat - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return gate, normalizer


def gate_lambda(gate, normalizer, features):
    with torch.no_grad():
        return gate(normalizer.transform(features))


def run_basis(tag, B, R, Y, weights, channel, qa_features_raw, is_sel, is_hold, e2e_qa_lambda=None):
    B_s, R_s, Y_s, ch_s, feat_s = B[is_sel], R[is_sel], Y[is_sel], channel[is_sel], qa_features_raw[is_sel]
    B_h, R_h, Y_h, ch_h, feat_h = B[is_hold], R[is_hold], Y[is_hold], channel[is_hold], qa_features_raw[is_hold]

    out = {'basis': tag, 'n_select': int(is_sel.sum()), 'n_holdout': int(is_hold.sum())}

    # F0
    out['f0_base_only'] = mse(B_h, Y_h)

    # F1 Global Lambda (fit on Val-Select)
    lam_g = fit_global_lambda(B_s, R_s, Y_s)
    out['f1_global_lambda'] = {'lambda': lam_g, 'val_holdout_mse': mse(B_h + lam_g * (R_h - B_h), Y_h)}

    # F2 Per-channel Lambda (fit on Val-Select)
    lam_c = fit_per_channel_lambda(B_s, R_s, Y_s, ch_s)
    out['f2_per_channel_lambda'] = {
        'lambda_by_channel': lam_c,
        'val_holdout_mse': mse(apply_per_channel_lambda(lam_c, B_h, R_h, ch_h), Y_h),
    }

    # F3/F4: fit a fresh QualityGate on Val-Select
    gate, normalizer = fit_quality_gate(feat_s, B_s, R_s, Y_s)
    lam_s_fitted = gate_lambda(gate, normalizer, feat_s)
    const_lam = float(lam_s_fitted.mean())
    out['f3_constant_mean_qa'] = {
        'constant_lambda': const_lam,
        'val_holdout_mse': mse(B_h + const_lam * (R_h - B_h), Y_h),
    }
    lam_h_fitted = gate_lambda(gate, normalizer, feat_h)
    out['f4_dynamic_qa_refit_on_valselect'] = {
        'lambda_mean': float(lam_h_fitted.mean()),
        'val_holdout_mse': mse(B_h + lam_h_fitted.unsqueeze(-1) * (R_h - B_h), Y_h),
    }
    if e2e_qa_lambda is not None:
        lam_e2e_h = e2e_qa_lambda[is_hold]
        out['f4_e2e_trained_gate'] = {
            'lambda_mean': float(lam_e2e_h.mean()),
            'val_holdout_mse': mse(B_h + lam_e2e_h.unsqueeze(-1) * (R_h - B_h), Y_h),
        }

    # C3 oracle headroom (fit DIRECTLY on Val-Holdout's own Y -- upper bound)
    oracle_g = fit_global_lambda(B_h, R_h, Y_h)
    oracle_c = fit_per_channel_lambda(B_h, R_h, Y_h, ch_h)
    oracle_dyn = validation_oracle_lambda(B_h, R_h, Y_h)
    out['c3_oracle_headroom'] = {
        'global_oracle_mse': mse(B_h + oracle_g * (R_h - B_h), Y_h),
        'per_channel_oracle_mse': mse(apply_per_channel_lambda(oracle_c, B_h, R_h, ch_h), Y_h),
        'dynamic_oracle_mse': mse(B_h + oracle_dyn.unsqueeze(-1) * (R_h - B_h), Y_h),
    }

    # C4 quality calibration (using the Val-Select-refit dynamic gate's lambda on Val-Holdout)
    advantage_h = retrieval_advantage(B_h, R_h, Y_h)
    out['c4_quality_calibration'] = quality_calibration_report(lam_h_fitted, advantage_h)

    # C5 shuffled-feature control (Val-Holdout only, permute WITHIN channel)
    torch.manual_seed(0)
    perm = torch.arange(feat_h.size(0))
    for c in ch_h.unique().tolist():
        idx = (ch_h == c).nonzero(as_tuple=True)[0]
        perm[idx] = idx[torch.randperm(len(idx))]
    feat_h_shuffled = feat_h[perm]
    lam_h_shuffled = gate_lambda(gate, normalizer, feat_h_shuffled)
    out['c5_shuffled_feature_control'] = {
        'original_qa_mse': out['f4_dynamic_qa_refit_on_valselect']['val_holdout_mse'],
        'shuffled_qa_mse': mse(B_h + lam_h_shuffled.unsqueeze(-1) * (R_h - B_h), Y_h),
        'original_better_than_shuffled': out['f4_dynamic_qa_refit_on_valselect']['val_holdout_mse']
            < mse(B_h + lam_h_shuffled.unsqueeze(-1) * (R_h - B_h), Y_h),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='results/TRACK-P100-QA-CAUSAL-VALIDATION01/phase_c_raw')
    ap.add_argument('--out_json', required=True)
    cli = ap.parse_args()

    report = {}
    for cell, method, h in PAIRS:
        noqa_raw = torch.load(Path(cli.raw_dir) / cell / method / 'noqa_raw.pt', map_location='cpu')
        qa_raw = torch.load(Path(cli.raw_dir) / cell / method / 'qa_raw.pt', map_location='cpu')

        assert torch.allclose(noqa_raw['Y'], qa_raw['Y']), f'{cell}/{method}: Y mismatch between arms!'
        assert torch.equal(noqa_raw['channel'], qa_raw['channel']), f'{cell}/{method}: channel mismatch!'

        is_sel, is_hold = split_mask(noqa_raw['window_start_idx'], h, h)
        print(f'[phase_c] {cell}/{method}: n_select={int(is_sel.sum())} n_holdout={int(is_hold.sum())}')

        noqa_basis = run_basis('noqa_retriever', noqa_raw['B'], noqa_raw['R'], noqa_raw['Y'], noqa_raw['weights'],
                               noqa_raw['channel'], noqa_raw['qa_features_raw'], is_sel, is_hold)
        qa_basis = run_basis('qa_retriever', qa_raw['B'], qa_raw['R'], qa_raw['Y'], qa_raw['weights'],
                             qa_raw['channel'], qa_raw['qa_features_raw'], is_sel, is_hold,
                             e2e_qa_lambda=qa_raw.get('qa_lambda'))

        report[f'{cell}/{method}'] = {'noqa_retriever_basis': noqa_basis, 'qa_retriever_basis': qa_basis}
        print(f'[phase_c] {cell}/{method} DONE. noqa-basis F0={noqa_basis["f0_base_only"]:.6f} '
             f'F4={noqa_basis["f4_dynamic_qa_refit_on_valselect"]["val_holdout_mse"]:.6f} | '
             f'qa-basis F0={qa_basis["f0_base_only"]:.6f} '
             f'F4_e2e={qa_basis.get("f4_e2e_trained_gate", {}).get("val_holdout_mse")}')

    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out_json).write_text(json.dumps(report, indent=2))
    print(f'[phase_c] wrote {cli.out_json}')


if __name__ == '__main__':
    main()
