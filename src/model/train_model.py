import json
from typing import Callable


import torch
import mlflow
import torch.utils
from tqdm import tqdm
import torch.utils.data
import albumentations as A
from torchinfo import summary
from albumentations.pytorch import ToTensorV2


from src.data import load_dataset
from src.visualization import visualize
from src.model.models import model_loader
from src.model.metrics.metrics import Metrics
from src.data.labels import updated_class_dict
from src.model.loss_functions import loss_loader
from src.model.metrics import Metrics, Metric, iou


def train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    metrics_fn: Metrics,
    optimizer: torch.optim,
    train_dataset: torch.utils.data.DataLoader,
    epoch: int,
    device: str,
    track_step: bool = False,
    *args,
    **kwargs,
) -> float:
    """Training function that we iteratively run for N number of epochs.

    Args:
        model (torch.nn.Module): model that is going to be used for training.
        loss_fn (torch.nn.Module): loss function.
        optimizer (torch.optim): optimizer.
        train_dataset (torch.utils.data.DataLoader): training set.
        epoch (int): current epoch. Ex. 40 out of 100
    """

    train_loss = 0
    for images, masks in tqdm(train_dataset, desc=f"TRAINING STEP: Epoch {epoch}"):
        model.train()

        # (batch size, channels, height, width)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Model Prediction
        # (batch size, channels, height, width)
        pred_logits = model(images)

        # Loss Calculation
        loss = loss_fn(pred_logits, masks)
        train_loss += loss.item()

        # Metric(s) Calculation
        metrics_fn.calculate_metrics(masks, pred_logits)

        # Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataset)
    metrics = metrics_fn.get_metrics()

    if track_step:
        mlflow.log_metric(f"TRAIN loss", train_loss, step=epoch)

        for metric_name, metric_dict in metrics.items():
            for score_type, score in metric_dict.items():
                if score_type == "mean":
                    mlflow.log_metric(
                        key=f"TRAIN_{metric_name}_{score_type}", value=score, step=epoch
                    )
                else:
                    for class_name, class_score in score.items():
                        mlflow.log_metric(
                            key=f"TRAIN_{metric_name}_{score_type}_{class_name}",
                            value=class_score,
                            step=epoch,
                        )

    print(f"train loss: {train_loss}")
    print(json.dumps(metrics, sort_keys=True, indent=4, separators=(",", ": ")))


def validation_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    metrics_fn: Metrics,
    validation_dataset: torch.utils.data.DataLoader,
    epoch: int,
    device: str,
    track_step: bool = False,
    *args,
    **kwargs,
):
    val_loss = 0
    model.eval()
    with torch.inference_mode():
        for images, masks in tqdm(
            validation_dataset, desc=f"VALIDATION STEP: Epoch {epoch}"
        ):

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Model Prediction
            pred_logits = model(images)

            # Loss Calculation
            loss = loss_fn(pred_logits, masks)
            val_loss += loss.item()

            # Metric(s) Calculation
            metrics_fn.calculate_metrics(masks, pred_logits)

        val_loss /= len(validation_dataset)

    metrics = metrics_fn.get_metrics()

    if track_step:
        mlflow.log_metric("VALID_loss", val_loss, step=epoch)

        for metric_name, metric_dict in metrics.items():
            for score_type, score in metric_dict.items():
                if score_type == "mean":
                    mlflow.log_metric(
                        f"VALID_{metric_name}_{score_type}", score, step=epoch
                    )
                else:
                    for class_name, class_score in score.items():
                        mlflow.log_metric(
                            f"VALID_{metric_name}_{score_type}_{class_name}",
                            class_score,
                            step=epoch,
                        )

    print(f"validation loss: {val_loss}")
    print(json.dumps(metrics, sort_keys=True, indent=4, separators=(",", ": ")))


def track_experiment(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    def _track_experiment(train_fn):
        def wrapper(*args, **kwargs):
            with mlflow.start_run():
                train_fn()

        return wrapper

    return _track_experiment


def train(
    model_name: str,
    loss_name: str,
    metrics_fn: Callable,
    batch_size: int,
    data_dim: int,
    learning_rate: float,
    epochs: int,
    device: str,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
):
    NUM_CLASSES = 8

    data_transform = A.Compose(
        [
            A.Resize(height=data_dim, width=data_dim),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_dataset = load_dataset("train", batch_size, NUM_CLASSES, data_transform)
    validation_dataset = load_dataset("valid", batch_size, NUM_CLASSES, data_transform)

    model = model_loader(model_name)(in_channels=3, out_channels=NUM_CLASSES).to(device)
    loss_fn = loss_loader(loss_name)()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "loss_function": loss_name,
            "model": model_name,
            "metric_function": ["iou"],
            "optimizer": "adam",
        }

        # Log training hyperparameters
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            NO_PRINT = 0
            f.write(str(summary(model, verbose=NO_PRINT)))
            mlflow.log_artifact("model_summary.txt")

        for epoch in range(epochs):
            train_step(
                model=model,
                loss_fn=loss_fn,
                metrics_fn=metrics_fn,
                optimizer=optimizer,
                train_dataset=train_dataset,
                epoch=epoch,
                device=device,
                track_step=True,
            )

            validation_step(
                model=model,
                loss_fn=loss_fn,
                metrics_fn=metrics_fn,
                validation_dataset=validation_dataset,
                epoch=epoch,
                device=device,
                track_step=True,
            )

        mlflow.pytorch.log_model(model, "model")

    # VISUALIZATION
    sample_batch_img, sample_batch_mask = next(iter(validation_dataset))

    # (1, channels, data_dim, data_dim)
    sample_img = sample_batch_img[0].unsqueeze(0)
    sample_mask = sample_batch_mask[0].unsqueeze(0)

    visualize.from_prediction(
        model=model, image=sample_img, true_mask=sample_mask, device=device
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 0
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4
    DATA_DIM = 112
    EPOCHS = 5

    import os

    MLFLOW_CREDENTIALS_REQUIREMENTS = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]

    if all(
        [
            True if track_credentials in os.environ else False
            for track_credentials in MLFLOW_CREDENTIALS_REQUIREMENTS
        ]
    ):
        URI = os.environ["MLFLOW_TRACKING_URI"]
        USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"]
        PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"]
        TRACKING = f"""ENABLED
        USERNAME: {USERNAME}
        PASSWORD: {"*"*len(PASSWORD)}
        URI: {URI}
        """
    else:
        URI = None
        USERNAME = None
        PASSWORD = None
        TRACKING = "DISABLED"

    print(
        f"""
    TRAINING ON DEVICE: {device}

    SEED: {RANDOM_SEED}
    LEARNING RATE: {LEARNING_RATE}
    BATCH SIZE: {BATCH_SIZE}
    DATA DIMENSION (WIDTH x HEIGHT): {DATA_DIM} x {DATA_DIM}
    EPOCHS: {EPOCHS}

    TRACK EXPERIMENT: {TRACKING}
    """
    )

    if TRACKING == "DISABLED":
        user_input = input(
            "Continue training without tracking? [y/n] Currently models trained are not saved."
        )
        if user_input == "y":
            pass
        elif user_input == "n":
            exit()
        else:
            print("Wrong input. Exiting...")

    iou_metric = Metric(name="iou", metric_fn=iou)
    metrics_solver = Metrics(updated_class_dict, device, iou_metric)

    train(
        model_name="unet",
        loss_name="dice",
        metrics_fn=metrics_solver,
        batch_size=BATCH_SIZE,
        data_dim=DATA_DIM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        tracking_uri=URI,
        experiment_name="cityscapes",
        run_name="test run 1",
    )
