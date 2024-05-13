from collections import defaultdict
from typing import Callable

import numpy as np


class Metric:
    def __init__(self, name: str, metric_fn: Callable) -> None:
        self.name = name
        self.metric_fn = metric_fn
        self.score = defaultdict(list)
        self.mean_score = []

    def update_metric(self, true_mask, predict_mask, classmap, device):
        _metric = self.metric_fn(true_mask, predict_mask, classmap, device)

        for class_name, class_score in _metric["byclass"].items():
            self.score[class_name].append(class_score)

        self.mean_score.append(_metric["mean"])

    def get_metric(self):
        output_score = {}

        for class_name, class_score in self.score.items():
            output_score[class_name] = np.mean(class_score)

        output_mean_score = np.mean(self.mean_score)

        # reset score and mean_score
        self.score = defaultdict(list)
        self.mean_score = []

        return {"byclass": output_score, "mean": output_mean_score}


class Metrics:
    def __init__(self, classmap, device, *_metrics: Metric) -> None:
        self.classmap = classmap
        self.device = device
        self.metrics = {*_metrics}

    def calculate_metrics(self, true_mask, predict_mask):
        for metric in self.metrics:
            metric.update_metric(true_mask, predict_mask, self.classmap, self.device)

    def get_metrics(self) -> dict:
        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] = metric.get_metric()

        return metrics


def test():
    import torch
    from src.model.metrics import iou

    NUM_CLASSES = 3
    DATA_DIM = 10
    BATCH_SIZE = 4
    SEED = 0
    DEVICE = "cpu"

    torch.manual_seed(SEED)

    sample_classmap = {0: "class 1", 1: "class 2", 2: "class 3"}
    size = (BATCH_SIZE, NUM_CLASSES, DATA_DIM, DATA_DIM)

    test_true_mask = torch.randint(0, 2, size=size, device=DEVICE)
    test_predict_mask = torch.randn(*size, device=DEVICE)

    metric1 = Metric(name="iou", metric_fn=iou)
    metrics_solver = Metrics(sample_classmap, DEVICE, metric1)

    EPOCHS = 3

    for epoch in range(EPOCHS):
        metrics_solver.calculate_metrics(test_true_mask, test_predict_mask)

    metrics = metrics_solver.get_metrics()

    EXPECTATIONS = {
        "iou": {
            "byclass": {
                "class 1": 0.2509225010871887,
                "class 2": 0.24902723729610443,
                "class 3": 0.23636363446712494,
            },
            "mean": 0.24543779095013937,
        }
    }

    assert metrics == EXPECTATIONS, "Wrong metric(s)."


if __name__ == "__main__":
    test()
