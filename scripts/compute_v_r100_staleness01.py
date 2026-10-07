#!/usr/bin/env python3
"""TRACK-V-R100-EFFICIENCY01 -- staleness diagnostic (spec section 15).

Loads a trained R100 Stage-1 checkpoint and FAITHFULLY CONTINUES the
exact R100 training recipe for up to 99 more optimizer steps -- the
training loss at every one of those steps uses the SAME frozen
candidate bank snapshot taken at step 0 of this script (exactly what
the real R100 run did between two refreshes), never a fresh encode.
At ages {0, 50, 99} relative to that snapshot, a FIXED probe query
subset (256 queries, deterministic, from val -- `build_probe_set`,
reused unmodified) is scored against BOTH the frozen cached bank and a
freshly-recomputed full bank (current model weights, full N
candidates, never pruned), and the two rankings are compared. This
isolates whether staleness (vs. e.g. the missing candidate-side
gradient) is the dominant driver of any R100 performance gap --
diagnostic only, never used for training/selection.

Reused UNMODIFIED: `build_model`/`build_probe_set`
(`train_j_shared_encoder_drift01`), `arm_score`
(`train_factorial_e2e01`), `SlotHeads` (`train_k_multislot_predictive_retrieval01`),
`round_robin_topk_selection` (`train_t_pure_multislot01`),
`stable_topk_indices` (`RelationStage1`), `memory_value`
(`train_margutil01`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `individual_utility_memsafe`
(`train_factorial_e2e01`)."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import make_loader_generator
from scripts.train_factorial_e2e01 import arm_score, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, build_probe_set
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.full_candidate_bank import FullCandidateBank, encode_raw_channel_first

AGES = (0, 50, 99)
TOP_K = 10


def _pairwise(cached, fresh, probe_cand_mask):
    """cached, fresh: [Q, N] score rows. Returns overlap/correlation stats."""
    top10_c = stable_topk_indices(cached.masked_fill(~probe_cand_mask, float('-inf')), 10, largest=True)
    top10_f = stable_topk_indices(fresh.masked_fill(~probe_cand_mask, float('-inf')), 10, largest=True)
    top50_c = stable_topk_indices(cached.masked_fill(~probe_cand_mask, float('-inf')), 50, largest=True)
    top50_f = stable_topk_indices(fresh.masked_fill(~probe_cand_mask, float('-inf')), 50, largest=True)
    overlap10 = (top10_c.unsqueeze(-1) == top10_f.unsqueeze(-2)).any(-1).float().sum(-1) / 10.0
    overlap50 = (top50_c.unsqueeze(-1) == top50_f.unsqueeze(-2)).any(-1).float().sum(-1) / 50.0
    rhos = []
    c_np, f_np, m_np = cached.detach().cpu().numpy(), fresh.detach().cpu().numpy(), probe_cand_mask.cpu().numpy()
    for b in range(c_np.shape[0]):
        m = m_np[b]
        if m.sum() < 2:
            continue
        rho, _ = spearmanr(c_np[b, m], f_np[b, m])
        if rho == rho:
            rhos.append(rho)
    return {'top10_overlap': float(overlap10.mean()), 'top50_overlap': float(overlap50.mean()),
           'score_spearman': float(sum(rhos) / len(rhos)) if rhos else float('nan')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--arm', required=True, choices=('V0', 'V1', 'V2', 'V5'))
    ap.add_argument('--stage1_checkpoint', required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=1000)  # distinct from training's own loader seed
    ap.add_argument('--n_probe', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    num_slots = {'V0': 1, 'V1': 1, 'V2': 2, 'V5': 5}[cli.arm]
    slot_heads = SlotHeads(int(args.d_model), n_slots=num_slots).to(device) if cli.arm != 'V0' else None

    bl = torch.load(cli.stage1_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    if slot_heads is not None:
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])

    _, val_loader = exp._get_data(flag='val', shuffle=False)
    probe = build_probe_set(exp, val_loader, cli.n_probe, device)

    # age=0 snapshot: the frozen candidate bank this diagnostic run will train
    # against -- built via the SAME `FullCandidateBank.refresh` the real R100
    # trainers use (model.eval() internally), so this is byte-identical to what
    # an actual R100 run's refresh would have produced at this point.
    bank_obj = FullCandidateBank(channels)
    bank_obj.refresh(model, exp.memory_x)
    cached_bank = {c: bank_obj.get(c) for c in channels}

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + (list(slot_heads.parameters()) if slot_heads is not None else [])
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=gen)
    train_iter = iter(train_loader)

    def continuation_step():
        nonlocal train_iter
        try:
            batch_x, batch_y, batch_start_idx = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y, batch_start_idx = next(train_iter)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        for c in channels:
            z_q = encode_raw_channel_first(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            if slot_heads is None:
                s = arm_score(z_q, cached_bank[c], None)
                l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            else:
                q = slot_heads(z_q)
                k_full = F.normalize(cached_bank[c], dim=-1)
                scores = torch.einsum('bsd,nd->bsn', q, k_full)
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                l = kl_loss_from_prob(p_t, p_bar, cand_mask)
            (l / len(channels)).backward()
        optimizer.step()

    def diag_at_age(age):
        was_training = model.training
        model.eval()
        rows = []
        with torch.no_grad():
            for c in channels:
                fresh_c = encode_raw_channel_first(model, exp.memory_x, c)
                emb_cos = F.cosine_similarity(cached_bank[c], fresh_c, dim=-1)
                if slot_heads is None:
                    z_q = encode_raw_channel_first(model, probe['x'], c)
                    s_cached = arm_score(z_q, cached_bank[c], None)
                    s_fresh = arm_score(z_q, fresh_c, None)
                    pair = _pairwise(s_cached, s_fresh, probe['cand_mask'])
                    rows.append({'age': age, 'channel': c, 'head': 'single',
                                **pair, 'embedding_cosine_mean': float(emb_cos.mean()),
                                'embedding_cosine_std': float(emb_cos.std())})
                else:
                    z_q = encode_raw_channel_first(model, probe['x'], c)
                    q = slot_heads(z_q)
                    k_cached = F.normalize(cached_bank[c], dim=-1)
                    k_fresh = F.normalize(fresh_c, dim=-1)
                    scores_cached = torch.einsum('bsd,nd->bsn', q, k_cached)
                    scores_fresh = torch.einsum('bsd,nd->bsn', q, k_fresh)
                    for h in range(num_slots):
                        pair = _pairwise(scores_cached[:, h, :], scores_fresh[:, h, :], probe['cand_mask'])
                        rows.append({'age': age, 'channel': c, 'head': f'H{h+1}',
                                    **pair, 'embedding_cosine_mean': float(emb_cos.mean()),
                                    'embedding_cosine_std': float(emb_cos.std())})
                    picks_cached = round_robin_topk_selection(scores_cached, probe['cand_mask'], k=TOP_K)
                    picks_fresh = round_robin_topk_selection(scores_fresh, probe['cand_mask'], k=TOP_K)
                    agreement = (picks_cached.unsqueeze(-1) == picks_fresh.unsqueeze(-2)).any(-1).float().sum(-1) / TOP_K
                    rows.append({'age': age, 'channel': c, 'head': 'roundrobin_final',
                                'top10_overlap': float(agreement.mean()), 'top50_overlap': float('nan'),
                                'score_spearman': float('nan'),
                                'embedding_cosine_mean': float(emb_cos.mean()),
                                'embedding_cosine_std': float(emb_cos.std())})
        if was_training:
            model.train()
        return rows

    all_rows = []
    model.train()
    all_rows += diag_at_age(0)
    for step in range(1, max(AGES) + 1):
        continuation_step()
        if step in AGES:
            all_rows += diag_at_age(step)
            print(f'[staleness] {cli.cell}/{cli.arm} age={step} diagnostic recorded')

    with open(out_dir / f'staleness_diagnostics_{cli.cell}_{cli.arm}.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in all_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f'[staleness] wrote {out_dir}/staleness_diagnostics_{cli.cell}_{cli.arm}.csv')


if __name__ == '__main__':
    main()
