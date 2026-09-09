#!/usr/bin/env python3
"""Track B4: EXP-CORRECTION-ENCODER01 -- does training a Stage-1 encoder
FROM SCRATCH against a CORRECTION-oriented Oracle (candidates whose
residual `r_i = Y_i - B_i` best corrects the frozen Base Predictor's OWN
prediction error) retrieve more downstream-useful candidates than the
EXISTING, already-trained Future-oriented encoder ("F-Encoder", the
production Stage-2 checkpoint's own encoder, reused unmodified, never
retrained)?

Continues Track B (B1 `diag_correction_oracle01.py`: real, mixed Correction
Oracle headroom over Future Oracle at the VALUE level; B2 `train_correction_
stage2_semantics01.py`: reusing the EXISTING future-oriented Top-K for a
correction fusion does NOT realize that headroom; B3 `train_correction_
selector01.py`: a Choice-CE selector trained on frozen Future-encoder
embeddings ALSO fails to clearly realize it, more consistent with
information-insufficiency than selection-mismatch). This experiment asks
whether the bottleneck is the REPRESENTATION itself -- retrain the encoder,
not just the selector on top of a frozen Future-oriented one.

Base Predictor: the SAME frozen Stage-2 checkpoint used throughout Track B
-- `base_forecast`/`load_stage2`/`unwrap` (`utils/retrieval_diagnostics.py`,
reused unmodified). NEVER updated here: eval mode, `requires_grad=False`,
never appears in the optimizer.

C-Encoder: SCRATCH random init (`Exp_Stage1_Relation._build_model()` never
auto-loads a checkpoint -- same scratch-init discipline as
`EXP-ORACLE-SCRATCH01`), trained via the SAME sequential Choice-CE machinery
as that experiment's "Individual Oracle" arm (`run_sequence_individual`,
`encode_raw`, `base_score`, all imported from `scripts/train_oracle_
scratch01.py`, not reimplemented) -- ONLY the per-candidate utility target
changes, from `u_i = -MSE(Y_i, Y_q)` (future) to `u_i = -MSE(B_q+r_i, Y_q)`
(correction). Cosine scorer only (this experiment does not vary scorer
kind, per the spec's "change one thing at a time" principle) -- one axis
(the training TARGET) isolated from every other Track A/B axis already
explored this session.

`r_i`, `Y_i`, `B_i` are NEVER encoder/scorer input -- only used to build the
Choice-CE loss target, matching every other Track B experiment's
information-control discipline. Candidate residuals precomputed ONCE
(frozen Base Predictor, full memory) before training starts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import oracle_rank_statistics
from scripts.train_margutil01 import build_experiment
from scripts.train_oracle_scratch01 import base_score, encode_raw, run_sequence_individual
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


def correction_targets(B_q, r_i_c, query_future):
    """u_i = -MSE(B_q+r_i, Y_q). Computed via the ALGEBRAICALLY EQUIVALENT
    `-MSE(r_i, r_q)` (B_q cancels) -- the equivalence itself is verified as
    a runtime sanity check by the caller on real batches, not assumed."""
    r_q = query_future - B_q
    d_i = (r_i_c.unsqueeze(0) - r_q.unsqueeze(1)).pow(2).mean(dim=-1)
    return -d_i, r_q


def run_epoch(exp, b0_model, model, loader, split, train, channel_list, k, tau_choice,
              chunk_size, r_i_full, optimizer, device):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    diag_sums, diag_n = {}, 0
    rank_frac_sums, rank_frac_n = 0.0, 0.0
    equiv_max_err = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, counts = exp._candidate_mask(batch_start_idx)
            if train:
                optimizer.zero_grad()
            batch_loss = 0.0
            with torch.no_grad():
                B_q_full = base_forecast(b0_model, batch_x, chunk_size=chunk_size)
            for c in channel_list:
                E = encode_raw(model, exp.memory_x, c)
                z_q = encode_raw(model, batch_x, c)
                b_i = base_score(z_q, E, None)

                B_q = B_q_full[:, :, c]
                query_future = batch_y[:, :, c]
                r_i_c = r_i_full[:, :, c]
                u_target, r_q = correction_targets(B_q, r_i_c, query_future)

                with torch.no_grad():
                    obj_a = (B_q + r_i_c[0] - query_future).pow(2).mean(-1)  # sample check, i=0 only (cheap)
                    obj_b = (r_i_c[0].unsqueeze(0) - r_q).pow(2).mean(-1)
                    equiv_max_err = max(equiv_max_err, float((obj_a - obj_b).abs().max()))

                losses, diags, picks = run_sequence_individual(b_i, u_target, cand_mask, tau_choice, k)
                for t, loss_t in enumerate(losses):
                    for kk, vv in diags[t].items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                    diag_n += 1
                batch_loss = batch_loss + sum(losses) / k

                if not train:
                    selected_mask = torch.zeros_like(cand_mask)
                    neg_inf = torch.finfo(u_target.dtype).min / 4
                    for t in range(k):
                        valid_now = cand_mask & ~selected_mask
                        stats = oracle_rank_statistics(
                            b_i, u_target.masked_fill(~valid_now, neg_inf).argmax(-1, keepdim=True),
                            valid_now, oracle_valid=valid_now.any(dim=-1))
                        frac = float(stats['oracle_top10_rank_fraction'])
                        if frac == frac:
                            rank_frac_sums += frac
                            rank_frac_n += 1
                        nxt = u_target.masked_fill(~valid_now, neg_inf).argmax(-1, keepdim=True)
                        selected_mask = selected_mask.scatter(1, nxt, True)

            batch_loss = batch_loss / len(channel_list)
            if train:
                batch_loss.backward()
                optimizer.step()
            total_loss += float(batch_loss.detach())
            n_batches += 1

    diag_means = {kk: vv / max(diag_n, 1) for kk, vv in diag_sums.items()}
    return {'loss': total_loss / max(n_batches, 1), 'diag': diag_means,
            'rank_fraction_overall': rank_frac_sums / max(rank_frac_n, 1), 'equiv_max_err': equiv_max_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True, help='EXISTING Stage-1 checkpoint, ARGS ONLY -- weights never loaded onto the C-Encoder')
    ap.add_argument('--stage2_checkpoint', required=True, help='frozen Base Predictor source (SAME checkpoint as Track B1/B2/B3)')
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_correction_encoder01')
    ap.add_argument('--model_id', default='carts_correction_encoder01_main')
    ap.add_argument('--des', default='correction_encoder01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau_choice', type=float, default=0.1)
    cli = ap.parse_args()

    overrides = {
        'is_training': 1, 'model_id': cli.model_id, 'des': cli.des,
        'checkpoints': cli.checkpoints, 'seed': cli.seed, 'top_k': cli.top_k,
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'stage1_retrieval_metric': 'cosine', 'stage1_full_memory_gradient_mode': 'full_online',
        'train_epochs': cli.train_epochs, 'patience': cli.patience,
    }
    exp, args = build_experiment(cli.reference_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)
    tau_choice = float(cli.tau_choice)

    for p in model.encoder.parameters():
        assert p.requires_grad, 'C-Encoder must be TRAINABLE (scratch)'

    b0_exp, b0_args = load_stage2(cli.stage2_checkpoint, device=device)
    b0_exp.model.to(device)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-CORRECTION-ENCODER01 is self-only; channel {c} has sources {sources}')

    with torch.no_grad():
        B_i_full = base_forecast(b0_model, exp.memory_x, chunk_size=cli.chunk_size)
        assert torch.isfinite(B_i_full).all(), 'non-finite candidate base forecast'
        r_i_full = exp.memory_y - B_i_full
        assert torch.isfinite(r_i_full).all(), 'non-finite candidate residual'

    trainable_params = list(model.encoder.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    print(f'[correction_encoder01] pred_len={cli.pred_len} trainable_params={n_params} tau_choice={tau_choice}')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / cli.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_epoch = -1
    patience_left = int(args.patience)
    history = []
    wall_start = time.time()

    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(exp, b0_model, model, train_loader, 'train', True, channels, k,
                                   tau_choice, cli.chunk_size, r_i_full, optimizer, device)
        val_metrics = run_epoch(exp, b0_model, model, val_loader, 'val', False, channels, k,
                                 tau_choice, cli.chunk_size, r_i_full, optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        print(f"[correction_encoder01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_rank_frac={val_metrics['rank_fraction_overall']:.5f} "
              f"val_top1_acc={val_metrics['diag'].get('top1_acc', float('nan')):.4f} "
              f"equiv_max_err(tr/va)={train_metrics['equiv_max_err']:.2e}/{val_metrics['equiv_max_err']:.2e} "
              f"time={dt:.1f}s")
        assert train_metrics['equiv_max_err'] < 1e-3, 'runtime equivalence MSE(B_q+r_i,Y_q)==MSE(r_i,r_q) violated on TRAIN'
        assert val_metrics['equiv_max_err'] < 1e-3, 'runtime equivalence MSE(B_q+r_i,Y_q)==MSE(r_i,r_q) violated on VAL'

        ckpt_payload = {'model_state_dict': model.state_dict(), 'args': vars(args), 'epoch': epoch + 1,
                         'val_rank_fraction_overall': val_metrics['rank_fraction_overall'],
                         'val_diag': val_metrics['diag'], 'stage2_checkpoint': cli.stage2_checkpoint,
                         'tau_choice': tau_choice}
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_metrics['rank_fraction_overall'] < best_val:
            best_val = val_metrics['rank_fraction_overall']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[correction_encoder01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {'best_epoch': best_epoch, 'best_val_rank_fraction': best_val, 'wall_clock_seconds': wall_time,
               'peak_gpu_memory_mib': peak_mem, 'history': history, 'trainable_params': n_params,
               'tau_choice': tau_choice, 'pred_len': cli.pred_len, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'stage2_checkpoint': cli.stage2_checkpoint}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[correction_encoder01] done. best_epoch={best_epoch} best_val_rank_fraction={best_val:.5f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[correction_encoder01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
