#!/usr/bin/env python3
"""Post-hoc HardAggregateMSE@10 (= MSE(mean_i y_i, y_q), unweighted, as
defined in research/RESEARCH_CONTEXT.md) for EXP-SEQDIAG01's four arms.
Not computed by scripts/eval_seqfull01_stage2.py (that script only reports
the B0-weighted A_weighted aggregate). Mirrors that script's evaluation
loop exactly (same picks, same candidate masking, same test split) but adds
the equal-weight aggregate alongside it. Read-only w.r.t. existing result
files; writes a new file, does not overwrite scripts/eval_seqfull01_stage2.py
or any existing results/EXP-SEQDIAG01/*.json.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path('/data/pjh_workspace/CARTS')
sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_seqfull01 import run_sequence
from utils.retrieval_diagnostics import load_stage2


def load_trained_selector(seq_ckpt_path, device):
    ckpt = torch.load(seq_ckpt_path, map_location='cpu')
    args = SimpleNamespace(**ckpt['args'])
    exp = Exp_Stage1_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    set_conditioner.load_state_dict(ckpt['set_conditioner_state_dict'])
    set_conditioner.eval()
    empty_token = EmptySetToken(d_model).to(device)
    empty_token.load_state_dict(ckpt['empty_token_state_dict'])
    empty_token.eval()
    return model, set_conditioner, empty_token, args


def encode(model, x, c):
    return F.normalize(model.encoder(model._relation_tensor(x, c, c)), dim=-1)


@torch.no_grad()
def evaluate(s2_ckpt, seq_ckpt, teacher_cache_path, k, device):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    seq_model, set_cond, empty_token, seq_args = load_trained_selector(seq_ckpt, device)
    teacher = torch.load(teacher_cache_path, map_location='cpu')
    _, loader = b0_exp._get_data(flag='test', shuffle=False)
    channels = list(b0_model.target_channels())

    sums = {k2: {'se': 0.0, 'n': 0} for k2 in ('individual', 'set_oracle', 'seq', 'b0')}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        vq = valid_query
        rows = [teacher['splits']['test']['start_to_row'][int(s)] for s in batch_start_idx.tolist()]

        for c in channels:
            E = encode(seq_model, memory_x, c)
            q = encode(seq_model, batch_x, c)
            _, picks = run_sequence(q, E, cand_mask, set_cond, empty_token, k, teacher_idx=None)

            memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
            q_tgt = batch_y[:, :, c]
            teacher_idx_c = teacher['splits']['test']['teacher_idx'][int(c)][rows].to(device)

            z_q_ref = b0_model._branch_embedding(batch_x, c, c)
            z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
            score_fn = b0_model._retrieval_score_fn()
            learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else torch.matmul(
                z_q_ref, z_mem_ref.transpose(0, 1))

            all_tgt = memory_c + offset_c.view(-1, 1, 1)  # [B, N, H]

            # individual oracle: best single candidate's own future (equal-weight of 1)
            dist_individual = (all_tgt - q_tgt.unsqueeze(1)).pow(2).mean(-1)
            dist_individual = dist_individual.masked_fill(~cand_mask, float('inf'))
            best_idx = dist_individual.argmin(dim=-1, keepdim=True)
            ind_tgt = all_tgt.gather(1, best_idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1))).squeeze(1)
            se = (ind_tgt - q_tgt).pow(2).sum(-1)
            sums['individual']['se'] += float(se[vq].sum()); sums['individual']['n'] += int(vq.sum())

            def hard_agg(idx):
                tgt = all_tgt.gather(1, idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
                agg = tgt.mean(1)
                return (agg - q_tgt).pow(2).sum(-1)

            se = hard_agg(teacher_idx_c)
            sums['set_oracle']['se'] += float(se[vq].sum()); sums['set_oracle']['n'] += int(vq.sum())

            se = hard_agg(picks)
            sums['seq']['se'] += float(se[vq].sum()); sums['seq']['n'] += int(vq.sum())

            b0_neg = torch.finfo(learned_ref.dtype).min / 4
            b0_masked = learned_ref.masked_fill(~cand_mask, b0_neg)
            b0_topk_idx = b0_masked.topk(k, dim=-1).indices
            se = hard_agg(b0_topk_idx)
            sums['b0']['se'] += float(se[vq].sum()); sums['b0']['n'] += int(vq.sum())

    out = {}
    for label, d in sums.items():
        out[f'hard_aggregate_mse_{label}'] = d['se'] / max(d['n'], 1) / q_tgt.size(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = evaluate(args.stage2_checkpoint, args.sequential_checkpoint,
                    args.teacher_cache, args.top_k, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    for key, val in out.items():
        print(f'[hard_agg] {key} = {val}')


if __name__ == '__main__':
    main()
