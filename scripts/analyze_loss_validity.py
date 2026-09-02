#!/usr/bin/env python3
"""Does the Stage-1 loss constrain the thing Stage-2 actually consumes?

Every teacher we tried improved a Stage-1 metric and moved no forecast. Before
proposing another target, this asks whether the *loss form* can reach the
quantity downstream depends on at all.

Stage-2 consumes `softmax(s / tau_topk)` over the model's own Top-10. Stage-1's
KL is taken over `softmax(s / tau_student)` across the whole bank -- thousands of
candidates. Those are different objects, and a student can match the teacher over
8449 candidates while its Top-10 stays a flat tie.

Three measurements, all on a trained checkpoint, no training:

  1. distribution shape   teacher and student entropy, effective candidate count
  2. mass placement       where the teacher puts its probability, and where the
                          KL's per-candidate contribution actually comes from
  3. Top-10 blindness     shuffle the student's own Top-10 scores among
                          themselves and re-measure the loss. Stage-2's weights
                          change completely under that permutation; if the loss
                          barely moves, it is blind to what Stage-2 reads.

Leakage: future MSE builds the teacher only, exactly as in training.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row  # noqa: E402

EPS = 1e-12
COLUMNS = [
    'dataset', 'pred_len', 'split', 'loss', 'queries', 'candidates',
    'tau_teacher', 'tau_student', 'top_k',
    'teacher_entropy', 'teacher_effective_count', 'teacher_top1_mass',
    'teacher_mass_on_oracle_top10', 'teacher_mass_on_student_top10',
    'student_entropy', 'student_effective_count', 'student_top1_mass',
    'student_mass_on_own_top10',
    'kl_total', 'kl_from_student_top10', 'kl_fraction_from_student_top10',
    'kl_from_teacher_top10', 'kl_fraction_from_teacher_top10',
    'kl_after_top10_shuffle', 'kl_shuffle_delta', 'kl_shuffle_relative_delta',
    'coverage_total', 'coverage_after_top10_shuffle', 'coverage_shuffle_relative_delta',
    'stage2_weight_entropy_before', 'stage2_weight_entropy_after_shuffle',
    'checkpoint',
]


def load_stage1(checkpoint_path):
    from exp.exp_stage1_relation import Exp_Stage1_Relation

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    experiment = Exp_Stage1_Relation(args)
    experiment.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    experiment.model.eval()
    return experiment, args


def _entropy(probability):
    return -(probability * (probability + EPS).log()).sum(-1)


@torch.no_grad()
def analyse(checkpoint, split='test', max_queries=512, top_k=10, seed=0):
    experiment, args = load_stage1(checkpoint)
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    experiment._ensure_memory()
    experiment._build_key_bank(log=False)
    device = experiment.device
    tau_teacher = float(model.tau_teacher)
    tau_student = float(model.tau_student)

    _, loader = experiment._get_data(flag=split, shuffle=False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    totals, seen = {}, 0

    for batch_x, batch_y, batch_start_idx in loader:
        if seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        seen += batch_x.size(0)

        for c in model.target_channels():
            slot = model.source_slot(c, c)
            future_mse = model._future_mse(
                batch_x, batch_y, experiment.memory_y, experiment.memory_x_last, c, c)
            z_q = model.encoder(model._relation_tensor(batch_x, c, c))
            z_mem = experiment.key_bank[c, slot].to(z_q.device, z_q.dtype)
            scores = torch.matmul(z_q, z_mem.transpose(0, 1))

            floor = torch.finfo(scores.dtype).min / 4
            teacher_logits = (-future_mse / tau_teacher).masked_fill(~cand_mask, floor)
            student_logits = (scores / tau_student).masked_fill(~cand_mask, floor)
            teacher = torch.softmax(teacher_logits, dim=-1)
            student_log = torch.log_softmax(student_logits, dim=-1)
            student = student_log.exp()

            # Per-candidate KL contribution: the loss is their sum, so the share
            # coming from a region is exactly how much that region drives it.
            per_candidate = teacher * ((teacher + EPS).log() - student_log)
            student_top = scores.masked_fill(~cand_mask, floor).topk(top_k, -1).indices
            teacher_top = future_mse.masked_fill(~cand_mask, float('inf')).topk(
                top_k, -1, largest=False).indices

            kl_total = per_candidate.sum(-1)
            kl_student_top = per_candidate.gather(1, student_top).sum(-1)
            kl_teacher_top = per_candidate.gather(1, teacher_top).sum(-1)

            # Permute the student's own Top-10 scores among themselves. Stage-2's
            # weighting is a function of exactly these, so its output changes;
            # a loss that does not notice is not constraining them.
            permutation = torch.argsort(torch.rand(
                student_top.shape, generator=generator), dim=-1).to(device)
            shuffled = scores.clone()
            top_values = scores.gather(1, student_top)
            shuffled.scatter_(1, student_top, top_values.gather(1, permutation))
            shuffled_log = torch.log_softmax(
                (shuffled / tau_student).masked_fill(~cand_mask, floor), dim=-1)
            kl_shuffled = (teacher * ((teacher + EPS).log() - shuffled_log)).sum(-1)

            # Coverage loss: -log of the student mass on the Oracle Top-K.
            coverage = -(student.gather(1, teacher_top).sum(-1) + EPS).log()
            coverage_shuffled = -(
                shuffled_log.exp().gather(1, teacher_top).sum(-1) + EPS).log()

            # What Stage-2 would do with these Top-10 scores, before and after.
            weights_before = torch.softmax(top_values / float(model.tau_student), -1)
            weights_after = torch.softmax(
                top_values.gather(1, permutation) / float(model.tau_student), -1)

            batch = {
                'teacher_entropy': _entropy(teacher),
                'teacher_effective_count': _entropy(teacher).exp(),
                'teacher_top1_mass': teacher.max(-1).values,
                'teacher_mass_on_oracle_top10': teacher.gather(1, teacher_top).sum(-1),
                'teacher_mass_on_student_top10': teacher.gather(1, student_top).sum(-1),
                'student_entropy': _entropy(student),
                'student_effective_count': _entropy(student).exp(),
                'student_top1_mass': student.max(-1).values,
                'student_mass_on_own_top10': student.gather(1, student_top).sum(-1),
                'kl_total': kl_total,
                'kl_from_student_top10': kl_student_top,
                'kl_fraction_from_student_top10': kl_student_top / kl_total.clamp_min(EPS),
                'kl_from_teacher_top10': kl_teacher_top,
                'kl_fraction_from_teacher_top10': kl_teacher_top / kl_total.clamp_min(EPS),
                'kl_after_top10_shuffle': kl_shuffled,
                'kl_shuffle_delta': kl_shuffled - kl_total,
                'kl_shuffle_relative_delta': (kl_shuffled - kl_total).abs() / kl_total.clamp_min(EPS),
                'coverage_total': coverage,
                'coverage_after_top10_shuffle': coverage_shuffled,
                'coverage_shuffle_relative_delta':
                    (coverage_shuffled - coverage).abs() / coverage.abs().clamp_min(EPS),
                'stage2_weight_entropy_before': _entropy(weights_before),
                'stage2_weight_entropy_after_shuffle': _entropy(weights_after),
            }
            for key, value in batch.items():
                totals.setdefault(key, []).append(value.detach().float().cpu())

    row = {key: float(torch.cat(value).mean()) for key, value in totals.items()}
    row.update({
        'dataset': args.data, 'pred_len': int(args.pred_len), 'split': split,
        'loss': getattr(args, 'stage1_loss_mode', 'kl'),
        'queries': seen, 'candidates': int(experiment.memory_y.size(0)),
        'tau_teacher': tau_teacher, 'tau_student': tau_student, 'top_k': top_k,
        'checkpoint': checkpoint,
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, nargs='+')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    for path in args.checkpoint:
        row = analyse(path, args.split, args.max_queries)
        print(f"=== {row['dataset']}/{row['pred_len']} {row['loss']} [{row['split']}] ===")
        for key in COLUMNS:
            if key == 'checkpoint':
                continue
            value = row[key]
            print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
        if args.csv:
            append_row(args.csv, row, COLUMNS)


if __name__ == '__main__':
    main()
