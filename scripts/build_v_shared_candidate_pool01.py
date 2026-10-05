#!/usr/bin/env python3
"""Build one future-blind shared Top-100 candidate pool for TRACK-V.

The selector is the frozen reference checkpoint's ordinary past-only cosine
retriever. It is deliberately arm-independent: V0/V1/V2/V5 (and a later
router) all consume exactly the same cached [query, channel, 100] support.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.shared_candidate_pool01 import TOP100_SIZE, tensor_sha256
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_j_shared_encoder_drift01 import state_hash
from scripts.train_margutil01 import build_experiment


@torch.no_grad()
def build_split(exp, model, split, channels, key_cache, device):
    _, loader = exp._get_data(flag=split, shuffle=False)
    starts_all, pools_all = [], []
    for batch_x, _, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        base_mask, counts = exp._candidate_mask(batch_start_idx)
        if int(counts.min()) < TOP100_SIZE:
            raise ValueError(
                f"{split}: native candidate pool has fewer than {TOP100_SIZE} valid candidates "
                f"(min={int(counts.min())})"
            )
        batch_pool = torch.empty(
            batch_x.size(0), len(channels), TOP100_SIZE, dtype=torch.long, device="cpu"
        )
        for ci, c in enumerate(channels):
            z_q = encode_raw(model, batch_x, c)
            score = arm_score(z_q, key_cache[c], None)
            idx = stable_topk_indices(
                score.masked_fill(~base_mask, float("-inf")),
                TOP100_SIZE,
                largest=True,
            )
            assert bool(base_mask.gather(1, idx).all())
            batch_pool[:, ci, :] = idx.cpu()
        starts_all.append(
            batch_start_idx.clone().cpu()
            if torch.is_tensor(batch_start_idx)
            else torch.as_tensor(batch_start_idx, dtype=torch.long)
        )
        pools_all.append(batch_pool)

    starts = torch.cat(starts_all).long().contiguous()
    pools = torch.cat(pools_all).long().contiguous()
    if starts.unique().numel() != starts.numel():
        raise ValueError(f"{split}: duplicate query_start_idx in pool builder")
    return {
        "query_start_idx": starts,
        "candidate_indices": pools,
        "split": split,
        "pool_size": TOP100_SIZE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_ckpt", required=True)
    ap.add_argument("--pred_len", type=int, required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp, args = build_experiment(
        cli.reference_ckpt,
        {
            "pred_len": cli.pred_len,
            "seq_len": cli.seq_len,
            "batch_size": cli.batch_size,
            "seed": cli.seed,
            "top_k": 10,
            "relation_encoder_type": "mlp",
            "relation_self_fill": "linear",
            "relation_input_space": "delta_last",
            "relation_teacher_space": "delta_last",
            "relation_value_space": "delta_last",
            "candidate_mask": "raft",
            "patch_len": 16,
            "stride": 16,
        },
    )
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, "module") else exp.model
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    channels = list(range(int(args.enc_in)))
    with torch.no_grad():
        key_cache = {c: encode_raw(model, exp.memory_x, c) for c in channels}

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_fingerprints = {}
    split_sizes = {}
    for split in ("train", "val", "test"):
        payload = build_split(exp, model, split, channels, key_cache, device)
        torch.save(payload, out_dir / f"{split}.pt")
        fp = tensor_sha256(payload["candidate_indices"])
        split_fingerprints[split] = fp
        split_sizes[split] = int(payload["query_start_idx"].numel())
        print(
            f"[shared_pool] {split}: q={split_sizes[split]} C={len(channels)} "
            f"M={TOP100_SIZE} sha256={fp[:16]}"
        )

    metadata = {
        "mode": "top100",
        "pool_size": TOP100_SIZE,
        "source": "frozen_reference_encoder_past_only_cosine",
        "reference_ckpt": cli.reference_ckpt,
        "reference_model_state_sha256": state_hash(model),
        "n_candidates": int(exp.memory_x.size(0)),
        "channels": channels,
        "pred_len": int(cli.pred_len),
        "seq_len": int(cli.seq_len),
        "seed": int(cli.seed),
        "split_sizes": split_sizes,
        "split_fingerprints": split_fingerprints,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[shared_pool] wrote {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
