#!/usr/bin/env python3
"""TRACK-V-R100-EFFICIENCY01 -- same-environment efficiency benchmark
(spec sections 13 and 14), run separately from the real training runs
so neither contaminates the other's timing.

Three configs, SAME GPU/code revision/batch size/fixed step count,
each starting from an IDENTICAL freshly-built model (build_model with
the same init_seed) and an identical loader order (same loader_seed):

  A0  = legacy preprocessing (`train_factorial_e2e01.encode_raw`) +
        full-online candidate encode every step (refresh_interval=1
        implemented inline, not via the cache class, so this path never
        imports anything from `utils/full_candidate_bank.py`)
  A1  = channel-first preprocessing (`encode_raw_channel_first`) +
        full-online candidate encode every step (same refresh_interval=1
        semantics, via the SAME R100 trainer module with
        `--refresh_interval 1` -- mathematically equivalent to always-
        fresh, see the trainer modules' own docstrings)
  R100 = channel-first preprocessing + candidate bank refreshed every
        100 steps (`--refresh_interval 100`)

A0 vs A1 isolates the channel-first preprocessing change alone (section
14): their step-0 forward pass (before any optimizer.step mutates
weights) must be numerically equivalent (encoder input/output max abs
diff, score max abs diff, Top10 agreement). A1 vs R100 isolates the
periodic-refresh change alone (section 13).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import make_loader_generator
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import compute_scores_full_grad
from utils.full_candidate_bank import encode_raw_channel_first


def _fresh_model_and_batches(cli, device, n_steps):
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=gen)
    batches = []
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
        if bi >= n_steps:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        batches.append((batch_x, batch_y, cand_mask))
    return exp, args, model, channels, batches


def _run_v0(cli, device, n_steps, config_name):
    """config_name in {'A0_legacy', 'A1_channelfirst', 'R100'}."""
    exp, args, model, channels, batches = _fresh_model_and_batches(cli, device, n_steps)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate)

    use_legacy_preproc = (config_name == 'A0_legacy')
    refresh_interval = 1 if config_name in ('A0_legacy', 'A1_channelfirst') else cli.refresh_interval
    enc = encode_raw if use_legacy_preproc else encode_raw_channel_first

    bank = {}
    def refresh_bank():
        with torch.no_grad():
            for c in channels:
                bank[c] = enc(model, exp.memory_x, c)
    refresh_bank()

    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for step, (batch_x, batch_y, cand_mask) in enumerate(batches, start=1):
        optimizer.zero_grad()
        for c in channels:
            z_q = enc(model, batch_x, c)
            E = bank[c]
            s = arm_score(z_q, E, None)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            (l / len(channels)).backward()
        optimizer.step()
        if step % refresh_interval == 0:
            refresh_bank()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.time() - t0
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    return {'config': config_name, 'n_steps': len(batches), 'wall_seconds': wall,
           'sec_per_step': wall / max(len(batches), 1), 'peak_vram_mib': vram}


def _run_v(cli, device, n_steps, config_name, num_slots):
    exp, args, model, channels, batches = _fresh_model_and_batches(cli, device, n_steps)
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=num_slots, std=cli.slot_std).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    use_legacy_preproc = (config_name == 'A0_legacy')
    refresh_interval = 1 if config_name in ('A0_legacy', 'A1_channelfirst') else cli.refresh_interval
    enc = encode_raw if use_legacy_preproc else encode_raw_channel_first

    bank = {}
    def refresh_bank():
        with torch.no_grad():
            for c in channels:
                bank[c] = enc(model, exp.memory_x, c)
    refresh_bank()

    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for step, (batch_x, batch_y, cand_mask) in enumerate(batches, start=1):
        optimizer.zero_grad()
        for c in channels:
            z_q = enc(model, batch_x, c)
            q = slot_heads(z_q)
            k_full = F.normalize(bank[c], dim=-1)
            scores = torch.einsum('bsd,nd->bsn', q, k_full)
            s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
            p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
            p_bar = p_m.mean(dim=1)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            l = kl_loss_from_prob(p_t, p_bar, cand_mask)
            (l / len(channels)).backward()
        optimizer.step()
        if step % refresh_interval == 0:
            refresh_bank()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.time() - t0
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    return {'config': config_name, 'n_steps': len(batches), 'wall_seconds': wall,
           'sec_per_step': wall / max(len(batches), 1), 'peak_vram_mib': vram}


def _equivalence_check_v0(cli, device):
    """A0 vs A1, step-0 forward pass only, before any weight mutation.
    model.eval() is mandatory here: the MLP relation encoder has
    dropout, and comparing two forward passes under train() would see
    two INDEPENDENT dropout masks and spuriously "disagree" even though
    the deterministic computation is identical -- this is a benchmark-
    harness correctness requirement, not a statement about training
    (training itself keeps dropout on in both the legacy and R100
    paths, unchanged)."""
    exp, args, model, channels, batches = _fresh_model_and_batches(cli, device, 1)
    model.eval()
    batch_x, batch_y, cand_mask = batches[0]
    rows = []
    for c in channels:
        with torch.no_grad():
            in_legacy_q = model._relation_tensor(batch_x, c, c)
            in_fast_q = torch.stack([v[..., 0] for v in __import__(
                'models.RelationStage1', fromlist=['transform_relation_features']
            ).transform_relation_features(batch_x[..., c:c + 1], model.relation_input_space)], dim=1)
            enc_in_diff = float((in_legacy_q - in_fast_q).abs().max())
            out_legacy = encode_raw(model, batch_x, c)
            out_fast = encode_raw_channel_first(model, batch_x, c)
            enc_out_diff = float((out_legacy - out_fast).abs().max())
            k_legacy = encode_raw(model, exp.memory_x, c)
            k_fast = encode_raw_channel_first(model, exp.memory_x, c)
            s_legacy = arm_score(out_legacy, k_legacy, None)
            s_fast = arm_score(out_fast, k_fast, None)
            score_diff = float((s_legacy - s_fast).abs().max())
            top10_legacy = stable_topk_indices(s_legacy.masked_fill(~cand_mask, float('-inf')), 10, largest=True)
            top10_fast = stable_topk_indices(s_fast.masked_fill(~cand_mask, float('-inf')), 10, largest=True)
            agreement = float((top10_legacy == top10_fast).float().mean())
        rows.append({'channel': c, 'encoder_input_max_abs_diff': enc_in_diff,
                    'encoder_output_max_abs_diff': enc_out_diff, 'score_max_abs_diff': score_diff,
                    'top10_agreement': agreement})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--arm', required=True, choices=('V0', 'V1', 'V2', 'V5'))
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--refresh_interval', type=int, default=100)
    ap.add_argument('--n_steps', type=int, default=50)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_slots = {'V0': None, 'V1': 1, 'V2': 2, 'V5': 5}[cli.arm]
    run_fn = (lambda name: _run_v0(cli, device, cli.n_steps, name)) if num_slots is None else \
            (lambda name: _run_v(cli, device, cli.n_steps, name, num_slots))

    results = {}
    for name in ('A0_legacy', 'A1_channelfirst', 'R100'):
        results[name] = run_fn(name)
        print(f'[bench] {cli.cell}/{cli.arm} {name}: {results[name]}')

    equiv_rows = _equivalence_check_v0(cli, device) if num_slots is None else None
    payload = {'cell': cli.cell, 'arm': cli.arm, 'n_steps': cli.n_steps, 'refresh_interval': cli.refresh_interval,
              'timing': results,
              'speedup_R100_over_A0': results['A0_legacy']['wall_seconds'] / max(results['R100']['wall_seconds'], 1e-9),
              'speedup_R100_over_A1': results['A1_channelfirst']['wall_seconds'] / max(results['R100']['wall_seconds'], 1e-9),
              'vram_reduction_pct_R100_vs_A0': 100.0 * (1 - results['R100']['peak_vram_mib'] /
                                                        max(results['A0_legacy']['peak_vram_mib'], 1e-9)),
              'channel_first_preprocessing_equivalence': equiv_rows}
    (out_dir / f'bench_{cli.cell}_{cli.arm}.json').write_text(json.dumps(payload, indent=2))
    print(f'[bench] wrote {out_dir / f"bench_{cli.cell}_{cli.arm}.json"}')


if __name__ == '__main__':
    main()
