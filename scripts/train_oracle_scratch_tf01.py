#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-TF01 -- teacher-forced Set-Oracle variant.

This file deliberately DOES NOT modify or overwrite EXP-ORACLE-SCRATCH01's
on-policy implementation/results.  It reuses that experiment's entire scratch
encoder/scorer/training pipeline, but replaces ONLY the Set arm's prefix source:

  on-policy:       prefix_{t-1} = model's own previous picks
  teacher forcing: prefix_{t-1} = greedy Set Oracle's previous picks

Individual Oracle is already an oracle-ordered deterministic curriculum in
EXP-ORACLE-SCRATCH01, so no new Individual training rule is needed.

At every Set step, the target is recomputed from the current ORACLE prefix with
the same dense_utility / fixed-base-score weighting used by the parent
experiment.  The model score is trained by the same masked full-memory Choice
CE.  t=1 is unchanged and still bypasses SetConditioner.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_oracle_scratch01 as base
from utils.dense_utility import dense_utility


def run_sequence_set_teacher_forcing(z_q, E, cand_mask, set_conditioner, metric, w_base,
                                       futures, query_future, tau_choice, k, chunk_size):
    """Teacher-forced Set Oracle sequence.

    The selected prefix is advanced by the ORACLE argmax of the true set
    utility, never by the model argmax.  Therefore every t>=2 conditioning
    state is the greedy Set Oracle state.  The scorer still sees only past
    embeddings; Y_q is used only to construct the no-grad Oracle label.
    """
    bsz, device = z_q.size(0), z_q.device
    selected_mask = torch.zeros_like(cand_mask)
    oracle_picks = []
    losses, diags, a_dense_steps = [], [], []
    neg_inf = torch.finfo(futures.dtype).min / 4

    for t in range(k):
        if t == 0:
            s_hat = base.base_score(z_q, E, metric)  # conditioner bypassed
        else:
            prefix = torch.stack(oracle_picks, dim=1).detach()
            m = E[prefix].mean(dim=1)
            h_t = set_conditioner(z_q, m)
            s_hat = base.base_score(h_t, E, metric)

        valid_now = cand_mask & ~selected_mask
        prefix_now = (
            torch.stack(oracle_picks, dim=1).detach()
            if t > 0
            else torch.zeros(bsz, 0, dtype=torch.long, device=device)
        )
        with torch.no_grad():
            a_dense = dense_utility(
                prefix_now, w_base, futures, query_future, chunk_size=chunk_size
            )
            u_target = -a_dense
        a_dense_steps.append(a_dense)

        loss_t, diag_t = base.oracle_choice_step_loss(
            s_hat, u_target, valid_now, tau_choice
        )
        losses.append(loss_t)
        diags.append(diag_t)

        # TEACHER FORCING: advance with Oracle's own next choice.
        target_masked = u_target.masked_fill(~valid_now, neg_inf)
        nxt = target_masked.argmax(dim=-1, keepdim=True).detach()
        oracle_picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)

    return losses, diags, torch.stack(oracle_picks, dim=1), a_dense_steps


def main():
    print('[oracle_scratch_tf01] Set prefix mode = TEACHER FORCING (oracle prefix).')
    # run_epoch() in the parent module resolves run_sequence_set from that
    # module's global namespace at call time, so this one-line substitution
    # changes exactly one mechanism while preserving every other code path.
    base.run_sequence_set = run_sequence_set_teacher_forcing
    base.main()


if __name__ == '__main__':
    main()
