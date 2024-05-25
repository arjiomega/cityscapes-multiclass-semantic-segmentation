import json
from typing import Literal

import torch
from tqdm import tqdm

from src.visualization import visualize
from src.model.metrics.metrics import Metrics


class Step:
    """
    A class to perform training or validation steps for a given model.

    Attributes:
        step_mode (Literal["train", "validation"]): Mode of the step, either 'train' or 'validation'.
        device (Literal["cuda", "cpu"]): Device to run the model on.
        model (torch.nn.Module): The model to be trained or validated.
        dataset (torch.utils.data.DataLoader): DataLoader for the dataset.
        metrics_fn (Metrics): Metrics calculation function.
        loss_fn (torch.nn.Module): Loss function for training or validation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        view_step_info (bool): Whether to print step info.
        reports_dir (str): Directory to save reports.
        step_loss (float): Accumulated loss for the current step.
        metrics (dict): Calculated metrics for the current step.

    Methods:
        step(epoch, **run_extra_fns): Perform a training or validation step for the given epoch.
        _run_loop(epoch): Run the main loop for training or validation.
    """

    def __init__(
        self,
        step_mode: Literal["train", "validation"],
        device: Literal["cuda", "cpu"],
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        metrics_fn: Metrics,
        loss_fn: torch.nn.Module = None,
        optimizer: torch.optim = None,
        reports_dir: str = None,
        view_step_info: bool = False,
    ) -> None:
        """
        Initialize the Step class.

        Args:
            step_mode (Literal["train", "validation"]): Mode of the step, either 'train' or 'validation'.
            device (Literal["cuda", "cpu"]): Device to run the model on.
            model (torch.nn.Module): The model to be trained or validated.
            dataset (torch.utils.data.DataLoader): DataLoader for the dataset.
            metrics_fn (Metrics): Metrics calculation function.
            loss_fn (torch.nn.Module, optional): Loss function for training or validation. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to None.
            reports_dir (str, optional): Directory to save reports. Defaults to None.
            view_step_info (bool, optional): Whether to print step info. Defaults to False.
        """

        self.step_mode = step_mode
        self.model = model
        self.dataset = dataset
        self.device = device
        self.metrics_fn = metrics_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.view_step_info = view_step_info
        self.reports_dir = reports_dir
        self.step_loss = 0.0
        self.metrics = None

    def step(self, epoch, **run_extra_fns):
        """
        Perform a training or validation step for the given epoch.

        Args:
            epoch (int): The current epoch number.
            **run_extra_fns: Additional functions to run during the step, e.g., logging functions.

        Raises:
            ValueError: If step_mode is neither 'train' nor 'validation'.
        """

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

        if "step_logger" in run_extra_fns:
            run_extra_fns["step_logger"](
                epoch, self.step_mode, self.step_loss, self.metrics, self.model
            )

    def _run_loop(self, epoch):
        """
        Run the main loop for training or validation.

        Args:
            epoch (int): The current epoch number.
        """

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


def step_initializer(
    device: Literal["cuda", "cpu"],
    model: torch.nn.Module,
    train_dataset: torch.utils.data.DataLoader,
    validation_dataset: torch.utils.data.DataLoader,
    metrics_fn: Metrics,
    loss_fn: torch.nn.Module = None,
    optimizer: torch.optim = None,
    view_step_info: bool = False,
    reports_dir: str = None,
) -> tuple[Step, Step]:
    """
    Initialize the training and validation steps.

    Args:
        device (Literal["cuda", "cpu"]): Device to run the model on.
        model (torch.nn.Module): The model to be trained or validated.
        train_dataset (torch.utils.data.DataLoader): DataLoader for the training dataset.
        validation_dataset (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        metrics_fn (Metrics): Metrics calculation function.
        loss_fn (torch.nn.Module, optional): Loss function for training or validation. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to None.
        view_step_info (bool, optional): Whether to print step info. Defaults to False.
        reports_dir (str, optional): Directory to save reports. Defaults to None.

    Returns:
        tuple[Step, Step]: A tuple containing the training and validation steps.
    """

    setup_ = {
        "device": device,
        "model": model,
        "metrics_fn": metrics_fn,
        "loss_fn": loss_fn,
        "view_step_info": view_step_info,
        "reports_dir": reports_dir,
    }

    train_step = Step(
        step_mode="train", dataset=train_dataset, optimizer=optimizer, **setup_
    )
    validation_step = Step(step_mode="validation", dataset=validation_dataset, **setup_)

    return train_step, validation_step
