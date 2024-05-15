import json
from pathlib import Path
from typing import Literal

import mlflow
import torch
from tqdm import tqdm

from src.model.metrics.metrics import Metrics
from src.visualization import visualize


class Step:
    def __init__(
        self,
        step_mode: Literal["train", "validation"],
        device: Literal["cuda", "cpu"],
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        metrics_fn: Metrics,
        loss_fn: torch.nn.Module = None,
        optimizer: torch.optim = None,
        track_experiment: bool = False,
        reports_dir: str = None,
        view_step_info: bool = False,
    ) -> None:
        self.step_mode = step_mode
        self.model = model
        self.dataset = dataset
        self.device = device
        self.metrics_fn = metrics_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.view_step_info = view_step_info
        self.track_experiment = track_experiment
        self.reports_dir = reports_dir
        self.step_loss = 0.0
        self.metrics = None

    def step(self, epoch):
        if self.step_mode == "train":
            self.model.train()
        elif self.step_mode == "validation":
            self.model.eval()
        else:
            raise ValueError("Invalid step type. Input either 'train' or 'validation'.")

        self.step_loss = 0.0

        if self.step_mode == "validation":
            with torch.inference_mode():
                self._run_loop(epoch)

        else:
            self._run_loop(epoch)

        self.step_loss /= len(self.dataset)
        self.metrics = self.metrics_fn.get_metrics()

        # VISUALIZATION
        if self.step_mode == "validation":
            sample_batch_img, sample_batch_mask = next(iter(self.dataset))

            # (1, channels, data_dim, data_dim)
            sample_img = sample_batch_img[0].unsqueeze(0)
            sample_mask = sample_batch_mask[0].unsqueeze(0)

            visualize.from_prediction(
                model=self.model,
                image=sample_img,
                true_mask=sample_mask,
                device=self.device,
                save_as=f"epoch_{epoch}",
                save_dir=self.reports_dir,
            )

        if self.view_step_info:
            print(f"{self.step_mode} loss: {self.step_loss}")
            print(
                json.dumps(self.metrics, sort_keys=True, indent=4)
            )  # separators=(",", ": ")

        if self.track_experiment:
            self._log_metrics(epoch)

    def _run_loop(self, epoch):
        for images, masks in tqdm(
            self.dataset, desc=f"{self.step_mode} step: Epoch {epoch}"
        ):

            # (batch size, channels, height, width)
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Model Prediction
            # (batch size, channels, height, width)
            pred_logits = self.model(images)

            # Loss Calculation
            loss = self.loss_fn(pred_logits, masks)
            self.step_loss += loss.item()

            # Metric(s) Calculation
            self.metrics_fn.calculate_metrics(masks, pred_logits)

            # Optimizer
            if self.step_mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _log_metrics(self, epoch):
        mlflow.log_metric(f"{self.step_mode}_loss", f"{self.step_loss:3f}", step=epoch)

        for metric_name, metric_dict in self.metrics.items():
            for score_type, score in metric_dict.items():
                if score_type == "mean":
                    mlflow.log_metric(
                        key=f"{self.step_mode}_{metric_name}_{score_type}",
                        value=f"{score:3f}",
                        step=epoch,
                    )
                else:
                    for class_name, class_score in score.items():
                        mlflow.log_metric(
                            key=f"{self.step_mode}_{metric_name}_{score_type}_{class_name}",
                            value=f"{class_score:3f}",
                            step=epoch,
                        )


def step_initializer(
    device: Literal["cuda", "cpu"],
    model: torch.nn.Module,
    train_dataset: torch.utils.data.DataLoader,
    validation_dataset: torch.utils.data.DataLoader,
    metrics_fn: Metrics,
    loss_fn: torch.nn.Module = None,
    optimizer: torch.optim = None,
    track_experiment: bool = False,
    view_step_info: bool = False,
    reports_dir: str = None,
) -> tuple[Step, Step]:

    setup_ = {
        "device": device,
        "model": model,
        "metrics_fn": metrics_fn,
        "loss_fn": loss_fn,
        "track_experiment": track_experiment,
        "view_step_info": view_step_info,
        "reports_dir": reports_dir,
    }

    train_step = Step(
        step_mode="train", dataset=train_dataset, optimizer=optimizer, **setup_
    )
    validation_step = Step(step_mode="validation", dataset=validation_dataset, **setup_)

    return train_step, validation_step


if __name__ == "__main__":
    train_step = TrainStep()
