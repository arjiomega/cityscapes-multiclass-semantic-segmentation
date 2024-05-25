from typing import Literal

from pydantic import field_validator
from pydantic.dataclasses import dataclass


# Allow extra arguments from train_args.json for easy parsing
@dataclass(config=dict(extra="ignore"))
class TrainArgs:
    epochs: int
    random_seed: int
    learning_rate: float
    batch_size: int
    loss_function: str
    model_name: str
    metric_function: list[str]
    optimizer: str
    num_classes: int
    data_dim: int
    device: Literal["cuda", "cpu"]


@dataclass
class MLFlowTrackingArgs:
    tracking_uri: str
    tracking_username: str
    tracking_password: str
    experiment_name: str
    run_name: str

    @field_validator("tracking_uri")
    @classmethod
    def check_a_format(cls, value: str) -> str:
        if not value.startswith("https://dagshub.com/"):
            raise ValueError("String must start with 'MLFLOW'")
        if not value.endswith((".mlflow", ".mlflow/")):
            raise ValueError("String must end with '.mlflow'")
        return value
