import json
from pathlib import Path
from typing import Callable, Literal


import torch
import mlflow
import torch.utils
from tqdm import tqdm
import torch.utils.data
import albumentations as A
from torchinfo import summary
from albumentations.pytorch import ToTensorV2


from config import config
from src.data import load_dataset
from src.model.step import step_initializer
from src.visualization import visualize
from src.model.models import model_loader
from src.model.metrics.metrics import Metrics
from src.data.labels import updated_class_dict
from src.model.loss_functions import loss_loader
from src.model.metrics import Metrics, Metric, iou


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
    num_classes: int,
    metrics_fn: Callable,
    batch_size: int,
    data_dim: int,
    learning_rate: float,
    epochs: int,
    device: str,
    view_step_info: bool = True,
    tracking_uri: str = None,
    experiment_name: str = None,
    run_name: str = None,
):

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

    if tracking_uri and experiment_name and run_name:
        track_experiment = True
    else:
        track_experiment = False

    train_dataset = load_dataset("train", batch_size, num_classes, data_transform)
    validation_dataset = load_dataset("valid", batch_size, num_classes, data_transform)

    model = model_loader(model_name)(in_channels=3, out_channels=num_classes).to(device)
    loss_fn = loss_loader(loss_name)()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    reports_dir = f"{experiment_name}_{run_name}"
    reports_path = Path(config.REPORTS_DIR, reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    train_step, validation_step = step_initializer(
        device=device,
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        metrics_fn=metrics_fn,
        loss_fn=loss_fn,
        optimizer=optimizer,
        track_experiment=track_experiment,
        view_step_info=view_step_info,
        reports_dir=reports_dir
    )

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

        with open(Path(reports_path, "model_summary.txt"), "w") as f:
            NO_PRINT = 0
            f.write(str(summary(model, verbose=NO_PRINT)))

        for epoch in range(epochs):
            train_step.step(epoch)
            validation_step.step(epoch)

        mlflow.log_artifacts(str(reports_path))
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 0
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    DATA_DIM = 112
    EPOCHS = 5
    NUM_CLASSES = 8

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
    NUM CLASSES: {NUM_CLASSES}
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
            exit()

    iou_metric = Metric(name="iou", metric_fn=iou)
    metrics_solver = Metrics(updated_class_dict, device, iou_metric)

    train(
        model_name="unet",
        loss_name="dice",
        num_classes=NUM_CLASSES,
        metrics_fn=metrics_solver,
        batch_size=BATCH_SIZE,
        data_dim=DATA_DIM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        tracking_uri=URI,
        experiment_name="cityscapes",
        run_name="test_run_3",
    )
