import torch


class MetricAverager:
    def __init__(self):
        self.totals = {}
        self.count = 0

    def update(self, metrics):
        self.count += 1
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def average(self):
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


def format_metrics(prefix, metrics):
    parts = [prefix]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f'{key}: {value:.6f}')
        else:
            parts.append(f'{key}: {value}')
    return ' | '.join(parts)
