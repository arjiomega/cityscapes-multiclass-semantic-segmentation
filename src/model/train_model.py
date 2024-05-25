import os
import json
from pathlib import Path

import torch
import torch.utils
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2


from config import config
from src.data import load_dataset
from src.model.models import model_loader
from src.model.step import step_initializer
from src.model.metrics.metrics import Metrics
from src.data.labels import updated_class_dict
from src.model import TrainArgs, MLFlowTrackingArgs
from src.model.loss_functions import loss_loader
from src.model.metrics import Metrics, Metric, iou
from src.experiment_tracking import ExperimentTracking


class TrainInitializer:
    def __init__(
        self, train_args: TrainArgs, tracking_args: MLFlowTrackingArgs = None
    ) -> None:
        self.train_args = train_args
        self.tracking_args = tracking_args

        if tracking_args != None:
            self.track = ExperimentTracking(tracking_args)

    def _initialize_components(self):
        model_name = self.train_args.model_name
        loss_name = self.train_args.loss_function
        num_classes = self.train_args.num_classes
        learning_rate = self.train_args.learning_rate

        IMG_CHANNEL_NUM = 3

        self.model = model_loader(model_name)(
            in_channels=IMG_CHANNEL_NUM, out_channels=num_classes
        ).to(device)
        self.loss_fn = loss_loader(loss_name)()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate
        )

        iou_metric = Metric(name="iou", metric_fn=iou)
        self.metrics_fn = Metrics(updated_class_dict, device, iou_metric)

    @staticmethod
    def _run_epoch_loop(*_steppers, epochs):
        def wrapper(**run_extra_fns):
            for epoch in range(epochs):
                for stepper in _steppers:
                    stepper.step(epoch, **run_extra_fns)

        return wrapper

    def train(self, train_dataset, validation_dataset, *args, **kwargs):

        self._initialize_components()

        if self.tracking_args != None:
            track_experiment = True
        else:
            track_experiment = False

        # to be removed
        self.run_name = self.tracking_args.run_name
        self.experiment_name = self.tracking_args.experiment_name

        reports_dir = f"{self.experiment_name}_{self.run_name}"

        train_step, validation_step = step_initializer(
            device=device,
            model=self.model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            metrics_fn=self.metrics_fn,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            view_step_info=True,
            reports_dir=reports_dir,
        )

        train_loop_fn = self._run_epoch_loop(
            train_step, validation_step, epochs=self.train_args.epochs
        )

        if track_experiment:
            self.track.track_run(train_loop_fn, self.train_args, self.model)
        else:
            train_loop_fn()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(Path(config.CONFIG_DIR, "train_args.json")) as json_file:
        _data = json.load(json_file)
        print(type(_data))
        train_args = TrainArgs(**_data, device=device)

    print(
        f"""
    TRAINING ON DEVICE: {device}

    SEED: {train_args.random_seed}
    LEARNING RATE: {train_args.learning_rate}
    BATCH SIZE: {train_args.batch_size}
    NUM CLASSES: {train_args.num_classes}
    DATA DIMENSION (WIDTH x HEIGHT): {train_args.data_dim} x {train_args.data_dim}
    EPOCHS: {train_args.epochs}

    """
    )

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

        TRACKING_REQUIREMENTS = ["experiment_name", "run_name"]
        if all([True if req in _data else False for req in TRACKING_REQUIREMENTS]):
            EXPERIMENT_NAME = _data["experiment_name"]
            RUN_NAME = _data["run_name"]
        else:
            missing = ", ".join(
                [req for req in TRACKING_REQUIREMENTS if req not in _data]
            )
            raise ValueError(f"{missing} not found in config/train_args.json")

        TRACKING = f"""ENABLED
    USERNAME: {USERNAME}
    PASSWORD: {"*"*len(PASSWORD)}
    URI: {URI}
    
    EXPERIMENT NAME: {EXPERIMENT_NAME}
    RUN_NAME: {RUN_NAME}
    
    """

        mlflow_tracking_args = MLFlowTrackingArgs(
            URI, USERNAME, PASSWORD, experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME
        )

    else:
        URI = None
        USERNAME = None
        PASSWORD = None
        TRACKING = "DISABLED"

        mlflow_tracking_args = None

    print(
        f"""
    TRACK EXPERIMENT: {TRACKING}"""
    )

    if TRACKING == "DISABLED":
        user_input = input(
            "Continue training without tracking? [y/n] Currently, models trained are not saved."
        )
        if user_input == "y":
            pass
        elif user_input == "n":
            exit()
        else:
            print("Wrong input. Exiting...")
            exit()

    data_transform = A.Compose(
        [
            A.Resize(height=train_args.data_dim, width=train_args.data_dim),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_dataset = load_dataset(
        dataset_group="train", transform=data_transform, **train_args.__dict__
    )
    validation_dataset = load_dataset(
        dataset_group="valid", transform=data_transform, **train_args.__dict__
    )

    train = TrainInitializer(train_args=train_args, tracking_args=mlflow_tracking_args)
    train.train(train_dataset, validation_dataset)
