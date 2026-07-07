#!/usr/bin/env python
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage2_relation import Exp_Stage2_Relation  # noqa: E402
from utils.retrieval_ops import retrieve_relation_future  # noqa: E402


CHANNELS_7 = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def parse_int_list(text):
    if text is None or text == '':
        return None
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def resolve_repo_path(path_value):
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_checkpoint_args(stage2_ckpt_path, force_cpu):
    ckpt = torch.load(stage2_ckpt_path, map_location='cpu')
    saved_args = dict(ckpt.get('args', {}))
    if not saved_args:
        raise RuntimeError(f'No args found in Stage-2 checkpoint: {stage2_ckpt_path}')

    saved_args['is_training'] = 0
    saved_args['task_name'] = 'stage2_relation'
    saved_args['use_multi_gpu'] = False
    saved_args['devices'] = str(saved_args.get('devices', '0'))
    saved_args['device_ids'] = [0]
    saved_args['num_workers'] = 0
    saved_args['root_path'] = resolve_repo_path(saved_args.get('root_path', './data/ETT/'))
    saved_args['checkpoints'] = resolve_repo_path(saved_args.get('checkpoints', './checkpoints/'))
    saved_args['stage1_ckpt_path'] = resolve_repo_path(saved_args.get('stage1_ckpt_path', ''))

    if force_cpu or not torch.cuda.is_available():
        saved_args['use_gpu'] = False
    else:
        saved_args['use_gpu'] = True
        saved_args['gpu'] = int(saved_args.get('gpu', 0))
    return SimpleNamespace(**saved_args), ckpt


def setting_from_args(args):
    setting_task_name = args.task_name.replace('_relation', '')
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        setting_task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0,
    )


def channel_names(num_channels):
    if num_channels == 7:
        return CHANNELS_7
    return [f'ch{i}' for i in range(num_channels)]


def date_at(df_raw, absolute_idx):
    if df_raw is None or absolute_idx < 0 or absolute_idx >= len(df_raw):
        return ''
    return str(df_raw.iloc[int(absolute_idx)]['date'])


def to_raw_channel(values, scaler, channel_idx):
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        return None
    return values * scaler.scale_[channel_idx] + scaler.mean_[channel_idx]


def source_indices(source_arg, names):
    if source_arg == 'all':
        return list(range(len(names)))
    selected = []
    for item in source_arg.split(','):
        item = item.strip()
        if not item:
            continue
        if item in names:
            selected.append(names.index(item))
        else:
            selected.append(int(item))
    return selected


def choose_samples(loader, requested_rows, requested_starts, num_samples, memory_bank, device):
    requested_rows = set(requested_rows or [])
    requested_starts = set(requested_starts or [])
    chosen = []
    seen_rows = 0
    for batch_x, batch_y, batch_start_idx in loader:
        starts_np = batch_start_idx.numpy().astype(np.int64)
        cand_mask, counts = memory_bank.valid_mask_batch(starts_np)
        for local_idx, start in enumerate(starts_np):
            use = False
            if requested_rows or requested_starts:
                use = seen_rows in requested_rows or int(start) in requested_starts
            elif len(chosen) < num_samples:
                use = int(counts[local_idx]) > 0
            if use and int(counts[local_idx]) > 0:
                chosen.append({
                    'test_row': seen_rows,
                    'local_idx': local_idx,
                    'batch_x': batch_x[local_idx:local_idx + 1].float().to(device),
                    'batch_y': batch_y[local_idx:local_idx + 1].float().to(device),
                    'query_start': int(start),
                    'valid_mask': cand_mask[local_idx:local_idx + 1].bool().to(device),
                    'valid_count': int(counts[local_idx]),
                })
            seen_rows += 1
        if requested_rows or requested_starts:
            if requested_rows.issubset({item['test_row'] for item in chosen}) and requested_starts.issubset({item['query_start'] for item in chosen}):
                break
        elif len(chosen) >= num_samples:
            break
    return chosen


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def plot_sample(out_path, sample_title, query_x, query_y, retrieved, top_values, alpha, top_scores, future_mse, source_name, target_name):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top_k = top_values.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].plot(query_x, color='black', linewidth=2.0, label=f'query input {target_name}')
    for rank in range(top_k):
        axes[0].plot(top_values[rank]['memory_x'], alpha=0.35, linewidth=1.0, label=f'top{rank + 1}' if rank < 3 else None)
    axes[0].set_title(f'{sample_title} | source={source_name} | input windows aligned')
    axes[0].legend(loc='upper right', ncol=4, fontsize=8)

    axes[1].plot(query_y, color='black', linewidth=2.0, label=f'true future {target_name}')
    axes[1].plot(retrieved, color='tab:red', linewidth=2.0, label='alpha weighted retrieved')
    for rank in range(top_k):
        axes[1].plot(top_values[rank]['memory_y'], alpha=0.4, linewidth=1.0, label=f'top{rank + 1}' if rank < 3 else None)
    axes[1].set_title('future values')
    axes[1].legend(loc='upper right', ncol=4, fontsize=8)

    xs = np.arange(1, top_k + 1)
    axes[2].bar(xs - 0.22, alpha, width=0.22, label='alpha')
    axes[2].bar(xs, top_scores, width=0.22, label='cosine score')
    axes[2].bar(xs + 0.22, future_mse, width=0.22, label='future MSE')
    axes[2].set_xticks(xs)
    axes[2].set_xlabel('top-k rank')
    axes[2].set_title('top-k weights and quality')
    axes[2].legend(loc='upper right', fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage-2 OT top-k retrieval choices on the test split.')
    parser.add_argument('--stage2_ckpt_path', required=True, help='Path to a trained Stage-2 checkpoint.pth')
    parser.add_argument('--output_dir', default='./analysis/topk_ot', help='Directory for CSV/NPZ/PNG outputs')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of first valid test samples to inspect')
    parser.add_argument('--sample_rows', default='', help='Comma-separated test dataset row indices to inspect')
    parser.add_argument('--query_starts', default='', help='Comma-separated absolute query start indices to inspect')
    parser.add_argument('--target_channel', default='OT', help='Target channel name or index')
    parser.add_argument('--sources', default='all', help='Source channels to inspect: all, OT, or comma-separated names/indices')
    parser.add_argument('--plot_sources', default='OT', help='Source channels to plot: all, OT, or comma-separated names/indices')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args_cli = parser.parse_args()

    stage2_ckpt_path = Path(args_cli.stage2_ckpt_path).resolve()
    out_dir = Path(args_cli.output_dir).resolve()

    os.chdir(REPO_ROOT)
    run_args, ckpt = load_checkpoint_args(stage2_ckpt_path, force_cpu=args_cli.cpu)
    exp = Exp_Stage2_Relation(run_args)
    exp.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    exp.model.eval()
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    test_data, test_loader = exp._get_data(flag='test')

    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    names = channel_names(run_args.enc_in)
    target_idx = names.index(args_cli.target_channel) if args_cli.target_channel in names else int(args_cli.target_channel)
    src_indices = source_indices(args_cli.sources, names)
    plot_src_indices = set(source_indices(args_cli.plot_sources, names)) if args_cli.plot_sources != 'all' else set(src_indices)

    df_raw_path = Path(run_args.root_path) / run_args.data_path
    df_raw = pd.read_csv(df_raw_path) if df_raw_path.exists() else None
    scaler = getattr(getattr(test_data, 'base_dataset', None), 'scaler', None)

    samples = choose_samples(
        test_loader,
        requested_rows=parse_int_list(args_cli.sample_rows),
        requested_starts=parse_int_list(args_cli.query_starts),
        num_samples=args_cli.num_samples,
        memory_bank=exp.memory_bank,
        device=exp.device,
    )
    if not samples:
        raise RuntimeError('No valid test samples found for the requested selection.')

    summary_rows = []
    arrays = {}
    memory_y = exp.memory_y
    memory_x_np = exp.memory_bank.memory_x
    memory_y_np = exp.memory_bank.memory_y
    memory_starts = exp.memory_bank.memory_starts
    key_bank = exp.key_bank
    setting = setting_from_args(run_args)

    with torch.no_grad():
        for sample_id, sample in enumerate(samples):
            batch_x = sample['batch_x']
            batch_y = sample['batch_y']
            valid_mask = sample['valid_mask']
            query_start = sample['query_start']
            y_final, y_base, y_ret, beta, lam, _ = model(
                batch_x=batch_x,
                memory_y=memory_y,
                valid_mask=valid_mask,
                key_bank=key_bank,
                memory_x_last=exp.memory_x_last,
            )

            query_x_np = batch_x[0, :, target_idx].detach().cpu().numpy()
            query_y_np = batch_y[0, :, target_idx].detach().cpu().numpy()
            arrays[f'sample{sample_id}_query_x_{names[target_idx]}'] = query_x_np
            arrays[f'sample{sample_id}_query_y_{names[target_idx]}'] = query_y_np
            arrays[f'sample{sample_id}_y_base_{names[target_idx]}'] = y_base[0, :, target_idx].detach().cpu().numpy()
            arrays[f'sample{sample_id}_y_ret_{names[target_idx]}'] = y_ret[0, :, target_idx].detach().cpu().numpy()
            arrays[f'sample{sample_id}_y_final_{names[target_idx]}'] = y_final[0, :, target_idx].detach().cpu().numpy()

            for src_idx in src_indices:
                q_rel = model._relation_tensor(batch_x, target_idx, src_idx)
                z_q = model.stage1_encoder(q_rel)
                z_mem = key_bank[target_idx, src_idx].to(exp.device)
                memory_value_c, query_offset_c = model._memory_value(
                    batch_x, memory_y, exp.memory_x_last, target_idx
                )
                retrieved, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                    z_q=z_q,
                    z_mem=z_mem,
                    memory_value_c=memory_value_c,
                    valid_mask=valid_mask,
                    top_k=run_args.top_k,
                    tau_topk=run_args.tau_topk,
                )
                retrieved = model._restore_retrieved_value(retrieved, query_offset_c)

                top_idx_np = top_idx[0].detach().cpu().numpy().astype(np.int64)
                alpha_np = alpha[0].detach().cpu().numpy()
                scores_np = top_scores[0].detach().cpu().numpy()
                retrieved_np = retrieved[0].detach().cpu().numpy()
                v_top = model._restore_retrieved_value(ret_debug['v_top'].reshape(-1, run_args.pred_len), query_offset_c.repeat_interleave(ret_debug['v_top'].size(1)))
                v_top_np = v_top.reshape(ret_debug['v_top'].shape)[0].detach().cpu().numpy()
                future_mse_np = ((v_top_np - query_y_np[None, :]) ** 2).mean(axis=1)
                query_y_raw = to_raw_channel(query_y_np, scaler, target_idx)
                v_top_raw = to_raw_channel(v_top_np, scaler, target_idx)
                beta_weight = float(beta[0, target_idx, src_idx].detach().cpu().item())
                lambda_value = float(lam[0, target_idx].detach().cpu().item())

                prefix = f'sample{sample_id}_{names[target_idx]}_from_{names[src_idx]}'
                arrays[f'{prefix}_top_idx'] = top_idx_np
                arrays[f'{prefix}_alpha'] = alpha_np
                arrays[f'{prefix}_scores'] = scores_np
                arrays[f'{prefix}_retrieved'] = retrieved_np
                arrays[f'{prefix}_top_values'] = v_top_np

                for rank, mem_idx in enumerate(top_idx_np):
                    mem_start = int(memory_starts[mem_idx])
                    row = {
                        'setting': setting,
                        'data': run_args.data,
                        'seq_len': run_args.seq_len,
                        'pred_len': run_args.pred_len,
                        'target_channel': names[target_idx],
                        'source_channel': names[src_idx],
                        'sample_id': sample_id,
                        'test_row': sample['test_row'],
                        'query_start': query_start,
                        'query_start_date': date_at(df_raw, query_start),
                        'query_future_start': query_start + run_args.seq_len,
                        'query_future_start_date': date_at(df_raw, query_start + run_args.seq_len),
                        'valid_candidate_count': sample['valid_count'],
                        'rank': rank + 1,
                        'memory_idx': int(mem_idx),
                        'memory_start': mem_start,
                        'memory_start_date': date_at(df_raw, mem_start),
                        'memory_future_start': mem_start + run_args.seq_len,
                        'memory_future_start_date': date_at(df_raw, mem_start + run_args.seq_len),
                        'lag_to_query_start': query_start - mem_start,
                        'score': float(scores_np[rank]),
                        'alpha': float(alpha_np[rank]),
                        'future_mse': float(future_mse_np[rank]),
                        'future_mae': float(np.abs(v_top_np[rank] - query_y_np).mean()),
                        'memory_future_mean': float(v_top_np[rank].mean()),
                        'memory_future_std': float(v_top_np[rank].std()),
                        'query_future_mean': float(query_y_np.mean()),
                        'query_future_std': float(query_y_np.std()),
                        'memory_future_mean_raw': '' if v_top_raw is None else float(v_top_raw[rank].mean()),
                        'memory_future_std_raw': '' if v_top_raw is None else float(v_top_raw[rank].std()),
                        'query_future_mean_raw': '' if query_y_raw is None else float(query_y_raw.mean()),
                        'query_future_std_raw': '' if query_y_raw is None else float(query_y_raw.std()),
                        'beta_target_source': beta_weight,
                        'lambda_target': lambda_value,
                    }
                    summary_rows.append(row)

                if src_idx in plot_src_indices:
                    plot_values = np.array([
                        {
                            'memory_x': memory_x_np[mem_idx, :, target_idx],
                            'memory_y': memory_y_np[mem_idx, :, target_idx],
                        }
                        for mem_idx in top_idx_np
                    ], dtype=object)
                    plot_sample(
                        out_dir / f'sample{sample_id}_{names[target_idx]}_from_{names[src_idx]}.png',
                        sample_title=f'{run_args.data} seq{run_args.seq_len} pred{run_args.pred_len} sample={sample_id}',
                        query_x=query_x_np,
                        query_y=query_y_np,
                        retrieved=retrieved_np,
                        top_values=plot_values,
                        alpha=alpha_np,
                        top_scores=scores_np,
                        future_mse=future_mse_np,
                        source_name=names[src_idx],
                        target_name=names[target_idx],
                    )

    fields = [
        'setting', 'data', 'seq_len', 'pred_len', 'target_channel', 'source_channel',
        'sample_id', 'test_row', 'query_start', 'query_start_date',
        'query_future_start', 'query_future_start_date', 'valid_candidate_count',
        'rank', 'memory_idx', 'memory_start', 'memory_start_date',
        'memory_future_start', 'memory_future_start_date', 'lag_to_query_start',
        'score', 'alpha', 'future_mse', 'future_mae',
        'memory_future_mean', 'memory_future_std', 'query_future_mean', 'query_future_std',
        'memory_future_mean_raw', 'memory_future_std_raw', 'query_future_mean_raw', 'query_future_std_raw',
        'beta_target_source', 'lambda_target',
    ]
    write_csv(out_dir / 'topk_summary.csv', summary_rows, fields)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'topk_arrays.npz', **arrays)
    print(f'wrote {len(summary_rows)} top-k rows to {out_dir / "topk_summary.csv"}')
    print(f'wrote arrays to {out_dir / "topk_arrays.npz"}')


if __name__ == '__main__':
    main()
