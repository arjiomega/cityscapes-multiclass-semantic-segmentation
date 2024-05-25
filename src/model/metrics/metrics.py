from collections import defaultdict
from typing import Callable

import numpy as np


class Metric:
    def __init__(self, name: str, metric_fn: Callable) -> None:
        self.name = name
        self.metric_fn = metric_fn
        self.score = defaultdict(list)
        self.mean_score = []

    # TODO DO NOT APPEND IF CLASS SCORE IS 0.0
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
