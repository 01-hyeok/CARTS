"""Shared pieces for the next-direction retrieval diagnostics.

Every analysis in this family needs the same three things: a past-only base
forecast for both queries and candidates, a way to turn a candidate ranking into
a forecast, and the gap-recovery framing that says how much of the achievable
improvement a retriever actually captured.

Leakage rule enforced here: base forecasts are produced from `batch_x` only.
Query futures appear exclusively inside Oracle/utility *targets*, never in
anything a retriever or forecaster consumes.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import torch

COVERAGE_DEPTHS = (10, 50, 100, 200, 500)
ALPHA_GRID = tuple(round(0.1 * i, 1) for i in range(11))


def load_stage2(checkpoint_path, device=None):
    """Rebuild a trained Stage-2 experiment from its own saved args."""
    from exp.exp_stage2_relation import Exp_Stage2_Relation

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'args' not in checkpoint:
        raise ValueError(f'checkpoint has no saved args: {checkpoint_path}')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    experiment = Exp_Stage2_Relation(args)
    experiment.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    experiment.model.eval()
    return experiment, args


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


@torch.no_grad()
def base_forecast(model, x, chunk_size=1024):
    """Past-only base-head forecast, chunked. [N, L, C] -> [N, pred_len, C].

    BaseForecastHead de-offsets its input by the window's last value and never
    restores it; RelationStage2.forward adds `output_offset = x[:, -1:, :]` back
    before anything compares the prediction to a target. Calling the head alone
    therefore returns a delta-space forecast, which is off by the last observed
    value -- so the offset is restored here too, exactly as the model does.
    """
    model = unwrap(model)
    outputs = []
    for start in range(0, x.size(0), chunk_size):
        window = x[start:start + chunk_size]
        outputs.append(model.base_head(window) + window[:, -1:, :])
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def collect_split(experiment, split, max_batches=0):
    """Run a trained Stage-2 model over a split, keeping the fusion parts apart.

    Returns y_base / y_ret / y_true / lambda so that any inference-time ablation
    (gate zeroed, scalar alpha, shuffled retrieval) can be replayed without
    retraining or re-running the model.
    """
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)
    experiment._build_retrieval_cache(split, loader)

    parts = {'base': [], 'ret': [], 'ret_pure': [], 'final': [], 'true': [],
             'lam': [], 'x': [], 'start': [], 'offset': []}
    for index, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx
        )
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        cache = experiment._cached_retrieval_for_batch(split, batch_start_idx)
        y_final, y_base, y_ret, _, lam, _ = experiment.model(
            batch_x=batch_x,
            memory_y=experiment.memory_y,
            valid_mask=cand_mask,
            key_bank=experiment.key_bank,
            memory_x_last=experiment.memory_x_last,
            retrieval_cache=cache,
            target_y=batch_y,
            teacher_key_bank=getattr(experiment, 'teacher_key_bank', None),
        )
        # Stage-2 fuses in delta space and restores the offset only at the
        # boundary, so y_ret already carries it. The pure correction -- the term
        # any alpha ablation must scale -- is y_ret minus that offset.
        offset = batch_x[:, -1:, :].detach()
        parts['base'].append(y_base.detach().float().cpu())
        parts['ret'].append(y_ret.detach().float().cpu())
        parts['ret_pure'].append((y_ret - offset).detach().float().cpu())
        parts['final'].append(y_final.detach().float().cpu())
        parts['offset'].append(offset.detach().float().cpu())
        parts['true'].append(batch_y.detach().float().cpu())
        parts['lam'].append(lam.detach().float().cpu())
        parts['x'].append(batch_x.detach().float().cpu())
        parts['start'].append(batch_start_idx.detach().cpu())
    return {key: torch.cat(value) for key, value in parts.items()}


def mse_mae(pred, true):
    """Canonical forecast metric: global mean over sample, horizon and channel.

    Shapes must match exactly. Broadcasting here is how a per-channel or
    delta-space tensor silently turns into a different number that still looks
    plausible, which is exactly what went wrong in the residual diagnostics.
    """
    if pred.shape != true.shape:
        raise ValueError(
            f'prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(true.shape)}'
        )
    return float((pred - true).square().mean()), float((pred - true).abs().mean())


def alpha_grid_search(y_base, y_ret, y_true, alphas=ALPHA_GRID):
    """Best scalar mixing weight under the residual fusion Stage-2 actually uses.

    `y_ret` must be the *pure* correction (`ret_pure` from collect_split), not
    the model's y_ret output: that one still carries the last-value offset, and
    scaling it by alpha would scale the offset too.
    """
    rows = [
        {'alpha': float(a), 'mse': mse_mae(y_base + a * y_ret, y_true)[0]}
        for a in alphas
    ]
    best = min(rows, key=lambda r: r['mse'])
    return best['alpha'], best['mse'], rows


def gap_recovery(retrieved, random_value, oracle_value, eps=1e-8, higher_is_better=False):
    """How much of the random -> Oracle span a retriever captured.

    0 means it did no better than random, 1 means it matched the Oracle. Values
    outside [0, 1] are kept as-is; clamping would hide a retriever that is worse
    than random.
    """
    if higher_is_better:
        return (retrieved - random_value) / (oracle_value - random_value + eps)
    return (random_value - retrieved) / (random_value - oracle_value + eps)


@torch.no_grad()
def coverage_at_m(scores, oracle_indices, valid_mask, depths=COVERAGE_DEPTHS):
    """Fraction of an Oracle Top-K set that a Top-M shortlist contains."""
    num_cand = scores.size(-1)
    oracle_valid = valid_mask.gather(1, oracle_indices)
    denominator = oracle_valid.sum(-1).clamp_min(1).float()
    out = {}
    for depth in depths:
        width = min(depth, num_cand)
        shortlist = scores.topk(width, dim=-1, largest=True).indices
        hit = (
            (oracle_indices.unsqueeze(-1) == shortlist.unsqueeze(-2)).any(-1)
            & oracle_valid
        )
        out[f'coverage_at_{depth}'] = (hit.sum(-1).float() / denominator).mean()
    return out


@torch.no_grad()
def rank_correlations(a, b, valid_mask, eps=1e-8):
    """Pearson and Spearman between two per-candidate scores, over valid pairs.

    Both are averaged per query so a few queries with many valid candidates do
    not dominate, matching how the retrieval metrics are aggregated elsewhere.
    """
    from models.RelationStage1 import stable_argsort

    pearson, spearman = [], []
    for row in range(a.size(0)):
        mask = valid_mask[row]
        count = int(mask.sum())
        if count < 2:
            continue
        x = a[row, mask].double()
        y = b[row, mask].double()
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            continue
        xc, yc = x - x.mean(), y - y.mean()
        denominator = (xc.square().sum() * yc.square().sum()).sqrt().clamp_min(eps)
        pearson.append((xc * yc).sum() / denominator)

        positions = torch.arange(count, dtype=torch.float64, device=x.device)
        rx, ry = torch.empty_like(x), torch.empty_like(y)
        rx[stable_argsort(x)] = positions
        ry[stable_argsort(y)] = positions
        rxc, ryc = rx - rx.mean(), ry - ry.mean()
        rden = (rxc.square().sum() * ryc.square().sum()).sqrt().clamp_min(eps)
        spearman.append((rxc * ryc).sum() / rden)

    if not pearson:
        return float('nan'), float('nan')
    return (
        float(torch.stack(pearson).mean()),
        float(torch.stack(spearman).mean()),
    )


def find_checkpoint(root, dataset, pred_len, pattern):
    """First checkpoint under root matching a name fragment, or None."""
    base = Path(root) / dataset / f'seq{pred_len}_pred{pred_len}'
    if not base.is_dir():
        return None
    for path in sorted(base.glob(f'*{pattern}*/checkpoint.pth')):
        return str(path)
    return None


def parse_setting_dims(checkpoint_path):
    """Pull d_model / d_ff back out of a generated experiment directory name."""
    d_model = re.search(r'_dm(\d+)_', checkpoint_path)
    d_ff = re.search(r'_df(\d+)_', checkpoint_path)
    return (
        int(d_model.group(1)) if d_model else None,
        int(d_ff.group(1)) if d_ff else None,
    )


def append_row(path, row, columns):
    """Incremental append so a crashed sweep keeps everything finished so far."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)
