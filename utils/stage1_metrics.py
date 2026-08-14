import torch


class MetricAverager:
    def __init__(self):
        self.totals = {}
        self.weights = {}

    def update(self, metrics, weight=1.0):
        weight = float(weight)
        if weight <= 0:
            return
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
                current = self.totals.get(key)
                weighted_value = value * weight
                self.totals[key] = (
                    weighted_value
                    if current is None
                    else current + weighted_value
                )
            else:
                current = self.totals.get(key, 0.0)
                self.totals[key] = current + float(value) * weight
            self.weights[key] = self.weights.get(key, 0.0) + weight

    def average(self):
        if not self.totals:
            return {}
        return {
            key: (
                value.item() / self.weights[key]
                if isinstance(value, torch.Tensor)
                else value / self.weights[key]
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
