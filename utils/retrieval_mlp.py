"""MLP retrieval encoders for the Recall study, under two teachers.

The student is the same in both cases:

    student  : sim( E(query_past) , E(candidate_past) )
    loss     : KL( teacher || student )

teacher='ema_future' -- Stage-1's `stage1_teacher_mode=ema_target`:

    teacher  : sim( E'(query_future) , E'(candidate_future) )   E' = EMA(E)

The teacher applies the same representation to the futures that the student
applies to the pasts. That keeps the two encoders shape-compatible -- which
matters for arima_residual, whose window is L-2 rather than L -- and keeps the
teacher defined in the same space as the student, so the KL target is not
measured in a different geometry than the thing being trained.

teacher='future_mse' -- Stage-1's `stage1_teacher_mode=mse`:

    teacher  : -MSE( query_future , candidate_future ) / tau

No encoder, no representation, no EMA: the target is the future distance
itself, which is the quantity the Recall oracle ranks by. So this teacher is a
fixed target from step one, where the EMA one is a moving target that only
becomes informative as the student learns.

Both read the same normalized futures the pipeline trains on, matching
Stage-1's default `teacher_mse_space=normalized`. Per channel the raw scale is
an affine rescale, so the induced ranking is identical either way and only the
softmax sharpness would move; normalized keeps one tau meaningful across
channels whose raw scales differ by orders of magnitude.

The teacher is only a training signal. The evaluation oracle stays raw-future
MSE and is built elsewhere, untouched by either of these.

decomposition trains one encoder per part (trend and seasonal do not share
weights) and fuses the two score matrices exactly as evaluation does.
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.retrieval_representations import transform
from utils.retrieval_scoring import (
    fuse_decomposition_scores,
    raw_future_mse,
    similarity_scores,
)

TEACHERS = ('ema_future', 'future_mse')


class RelationMLP(nn.Module):
    """rep_len -> d_ff -> d_model, the same shape Stage-1's mlp encoder uses."""

    def __init__(self, input_dim, d_model=128, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


def _score(parts, mem_parts, similarity, valid_mask):
    scores = [
        similarity_scores(q, m, similarity) for q, m in zip(parts, mem_parts)
    ]
    if len(scores) == 1:
        return scores[0]
    return fuse_decomposition_scores(scores, valid_mask)


@torch.no_grad()
def _embed_bank(encoders, windows, representation, ar_params, moving_avg,
                device, chunk=1024):
    """Encode every memory window, one part at a time, in chunks."""
    outs = None
    for start in range(0, windows.size(0), chunk):
        block = windows[start:start + chunk].to(device)
        parts = transform(
            block, representation, ar_params=ar_params, moving_avg=moving_avg
        )
        embedded = [enc(p) for enc, p in zip(encoders, parts)]
        if outs is None:
            outs = [[] for _ in embedded]
        for i, e in enumerate(embedded):
            outs[i].append(e)
    return [torch.cat(o, dim=0) for o in outs]


def train_encoders(
    memory_x,
    memory_y,
    sampler,
    channel,
    representation,
    similarity,
    ar_params,
    moving_avg,
    device,
    seq_len,
    epochs=10,
    lr=1e-3,
    batch_size=32,
    d_model=128,
    d_ff=256,
    tau_student=0.10,
    tau_teacher=0.07,
    tau_teacher_mse=0.10,
    ema_base=0.99,
    ema_final=0.9995,
    teacher='ema_future',
    log_every=100,
    log_prefix='',
):
    """Train the student encoders for one channel. Returns the student list."""
    if teacher not in TEACHERS:
        raise ValueError(f'Unsupported teacher: {teacher}')
    past = memory_x[:, :, channel]
    future = memory_y[:, :, channel]

    probe = transform(
        past[:1], representation, ar_params=ar_params, moving_avg=moving_avg
    )
    n_parts = len(probe)
    input_dim = probe[0].size(-1)

    students = [
        RelationMLP(input_dim, d_model, d_ff).to(device) for _ in range(n_parts)
    ]
    use_ema = teacher == 'ema_future'
    teachers = []
    future_bank = None
    if use_ema:
        teachers = [copy.deepcopy(s).to(device) for s in students]
        for t in teachers:
            for p in t.parameters():
                p.requires_grad_(False)
    else:
        # Fixed target: the candidate futures themselves, held on device once.
        future_bank = future.to(device)

    params = [p for s in students for p in s.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)

    n = past.size(0)
    starts = sampler.get_all_valid_starts() if hasattr(sampler, 'get_all_valid_starts') else None
    total_steps = max(1, (n // batch_size)) * epochs
    step = 0

    for epoch in range(epochs):
        # Both banks are rebuilt each epoch: the student moves, and the teacher
        # moves with it through the EMA.
        for s in students:
            s.eval()
        student_bank = _embed_bank(
            students, past, representation, ar_params, moving_avg, device
        )
        teacher_bank = None
        if use_ema:
            teacher_bank = _embed_bank(
                teachers, future, representation, ar_params, moving_avg, device
            )
        for s in students:
            s.train()

        perm = torch.randperm(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            q_past = past[idx].to(device)
            q_future = future[idx].to(device)

            mask_np, _ = sampler.valid_mask_batch(sampler.starts[idx.numpy()])
            valid = mask_np.bool().to(device)

            q_parts = transform(
                q_past, representation, ar_params=ar_params, moving_avg=moving_avg
            )
            student_parts = [enc(p) for enc, p in zip(students, q_parts)]
            student_scores = _score(student_parts, student_bank, similarity, valid)

            with torch.no_grad():
                if use_ema:
                    f_parts = transform(
                        q_future, representation,
                        ar_params=ar_params, moving_avg=moving_avg,
                    )
                    teacher_parts = [enc(p) for enc, p in zip(teachers, f_parts)]
                    teacher_scores = _score(
                        teacher_parts, teacher_bank, similarity, valid
                    )
                    teacher_temperature = tau_teacher
                else:
                    teacher_scores = -raw_future_mse(q_future, future_bank)
                    teacher_temperature = tau_teacher_mse

            neg_inf = torch.finfo(student_scores.dtype).min / 4
            student_log_prob = F.log_softmax(
                (student_scores / tau_student).masked_fill(~valid, neg_inf), dim=-1
            )
            teacher_prob = F.softmax(
                (teacher_scores / teacher_temperature).masked_fill(~valid, neg_inf),
                dim=-1,
            )
            loss = F.kl_div(
                student_log_prob, teacher_prob, reduction='batchmean'
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if use_ema:
                momentum = ema_final - (ema_final - ema_base) * (
                    1.0 - step / max(total_steps, 1)
                )
                with torch.no_grad():
                    for s, t in zip(students, teachers):
                        for sp, tp in zip(s.parameters(), t.parameters()):
                            tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)

            epoch_loss += float(loss.detach())
            batches += 1
            if log_every and batches % log_every == 0:
                print(
                    f'{log_prefix} ch{channel} epoch {epoch + 1}/{epochs} '
                    f'step {batches} kl={epoch_loss / batches:.4f}',
                    flush=True,
                )

        print(
            f'{log_prefix} ch{channel} epoch {epoch + 1}/{epochs} '
            f'mean_kl={epoch_loss / max(batches, 1):.4f}',
            flush=True,
        )

    for s in students:
        s.eval()
    return students


def checkpoint_path(root, dataset, seq_len, representation, similarity, seed,
                    teacher='ema_future'):
    # The ema_future name carries no teacher tag, so encoders trained before the
    # teacher axis existed are still found and reused.
    suffix = '' if teacher == 'ema_future' else f'_{teacher}'
    return os.path.join(
        root,
        f'{dataset}_L{seq_len}_{representation}_{similarity}_seed{seed}{suffix}.pt',
    )
