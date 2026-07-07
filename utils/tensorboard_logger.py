import os


def build_summary_writer(args, stage_name, setting):
    if not bool(int(getattr(args, 'use_tensorboard', 1))):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print('[tensorboard] torch.utils.tensorboard is unavailable; skip TensorBoard logging')
        return None

    log_dir = os.path.join(
        getattr(args, 'tensorboard_dir', './runs'),
        stage_name,
        args.data,
        f'seq{args.seq_len}_pred{args.pred_len}',
        setting,
    )
    print(f'[tensorboard] log_dir={log_dir}')
    return SummaryWriter(log_dir=log_dir)


def write_metric_scalars(writer, prefix, metrics, step, keys):
    if writer is None:
        return
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if hasattr(value, 'item'):
            value = value.item()
        try:
            writer.add_scalar(f'{prefix}/{key}', float(value), step)
        except (TypeError, ValueError):
            continue
