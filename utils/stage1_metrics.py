import torch


class MetricAverager:
    def __init__(self):
        self.totals = {}
        self.count = 0

    def update(self, metrics):
        self.count += 1
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
                current = self.totals.get(key)
                self.totals[key] = value if current is None else current + value
            else:
                current = self.totals.get(key, 0.0)
                self.totals[key] = current + float(value)

    def average(self):
        if self.count == 0:
            return {}
        return {
            key: (
                value.item() / self.count
                if isinstance(value, torch.Tensor)
                else value / self.count
            )
            for key, value in self.totals.items()
        }


def format_metrics(prefix, metrics):
    parts = [prefix]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f'{key}: {value:.6f}')
        else:
            parts.append(f'{key}: {value}')
    return ' | '.join(parts)
