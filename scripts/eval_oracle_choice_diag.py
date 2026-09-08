#!/usr/bin/env python3
"""EXP-ORACLE-CHOICE01 teacher-forced decision diagnostics: per-step
(t=1..K) Oracle next-choice Top-1/5/10 accuracy, predicted rank, and
step regret, on a D1 (or any oracle-prefix-trained) checkpoint. Reuses
`oracle_choice_step_loss` (the exact function D1's own training uses)
directly -- no reimplementation of the accuracy/rank/regret definitions.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import encode, memory_value, run_sequence_dense
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.dense_utility import dense_utility


def load_d1_selector(seq_ckpt_path, device):
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
    utility_head = UtilityHead().to(device)
    utility_head.load_state_dict(ckpt['utility_head_state_dict'])
    utility_head.eval()
    tau = float(ckpt.get('tau', args.tau_topk))
    return exp, model, set_conditioner, empty_token, utility_head, args, tau


@torch.no_grad()
def evaluate(seq_ckpt, teacher_cache_path, k, chunk_size, n_queries, device, split='test'):
    exp, model, set_cond, empty_token, utility_head, args, tau = load_d1_selector(seq_ckpt, device)
    exp._ensure_memory()
    channels = list(model.target_channels())
    c = channels[0]
    teacher = torch.load(teacher_cache_path, map_location='cpu')
    _, loader = exp._get_data(flag=split, shuffle=False)

    step_diag = {t: {'top1_acc': [], 'top5_acc': [], 'top10_acc': [], 'pred_rank_mean': [],
                      'top1_top2_margin_mean': [], 'step_regret': []} for t in range(k)}
    seen = 0
    E = encode(model, exp.memory_x, c)
    for batch_x, batch_y, batch_start_idx in loader:
        if seen >= n_queries:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        q = encode(model, batch_x, c)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        futures = memory_c + offset_c.view(-1, 1, 1)
        query_future = batch_y[:, :, c]

        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
        teacher_idx = teacher['splits'][split]['teacher_idx'][int(c)][rows].to(device)
        query_valid = teacher_idx[:, 0] != -1
        if not bool(query_valid.any()):
            continue
        take = min(int(query_valid.sum()), n_queries - seen)
        idx_take = query_valid.nonzero(as_tuple=True)[0][:take]

        u_hat_steps, u_target_steps, valid_steps, _ = run_sequence_dense(
            q, E, cand_mask, set_cond, empty_token, utility_head,
            futures, query_future, tau, k, chunk_size, teacher_idx=teacher_idx)

        from utils.dense_utility import candidate_weights
        w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)
        for t in range(k):
            u_hat_v = u_hat_steps[t][idx_take]
            u_target_v = u_target_steps[t][idx_take]
            valid_v = valid_steps[t][idx_take]
            if u_hat_v.size(0) == 0:
                continue
            _, diag = oracle_choice_step_loss(u_hat_v, u_target_v, valid_v, tau)
            for kk in step_diag[t]:
                if kk == 'step_regret':
                    continue
                v = diag.get(kk, float('nan'))
                if v == v:
                    step_diag[t][kk].append(v)
            # step regret: A(prefix + model_pick) - A(prefix + oracle_pick),
            # recomputed on the SAME oracle-prefix state D1's own loss uses
            # (matches spec's R_t = A(S*_{t-1}+{i_hat_t}) - A(S*_{t-1}+{i_t*}))
            prefix = teacher_idx[idx_take, :t].clamp_min(0)
            neg_inf = torch.finfo(u_target_v.dtype).min / 4
            a_dense = dense_utility(prefix, w[idx_take], futures[idx_take], query_future[idx_take],
                                     chunk_size=chunk_size)
            model_pick = u_hat_v.masked_fill(~valid_v, neg_inf).argmax(dim=-1)
            oracle_pick = u_target_v.masked_fill(~valid_v, neg_inf).argmax(dim=-1)
            regret = a_dense.gather(1, model_pick.unsqueeze(-1)).squeeze(-1) - \
                a_dense.gather(1, oracle_pick.unsqueeze(-1)).squeeze(-1)
            step_diag[t]['step_regret'].append(float(regret.mean()))
        seen += take

    def summarize(step_dict):
        out = {}
        for t in range(k):
            for kk, vals in step_dict[t].items():
                out[f't{t+1}_{kk}'] = sum(vals) / max(len(vals), 1) if vals else float('nan')
        return out

    result = summarize(step_diag)
    result['n_queries'] = seen
    result['split'] = split
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = evaluate(args.sequential_checkpoint, args.teacher_cache, args.top_k,
                       args.chunk_size, args.n_queries, device, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=2)
    for kk, vv in result.items():
        print(f'[oracle_choice_diag] {kk} = {vv}')
    print(f'[oracle_choice_diag] written to {args.out}')


if __name__ == '__main__':
    main()
