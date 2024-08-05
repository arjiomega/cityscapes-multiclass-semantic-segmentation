import logging
from typing import Literal
from dotenv import load_dotenv
import mlflow
import torch

import os

from pathlib import Path

from config import config

load_dotenv()


class Inference:
    """Handles model inference operations."""

    def __init__(self, model, device: Literal["cuda", "cpu"] = "cpu") -> None:
        """
        Initializes an Inference object.

        Args:
            model: The model to use for inference.
            device (Literal["cuda", "cpu"], optional): The device to run inference on, defaults to "cpu".
        """

        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_mlflow(
        cls,
        model_name,
        model_version: str = "latest",
        device: Literal["cuda", "cpu"] = "cpu",
    ):
        """
        Creates an Inference object from an MLflow model.

        Args:
            model_name: Name of the MLflow model.
            model_version: Version of the MLflow model, defaults to "latest".
            device (Literal["cuda", "cpu"], optional): The device to run inference on, defaults to "cpu".

        Returns:
            Inference: An initialized Inference object.
        """

        if os.path.exists(Path(config.MODELS_DIR, "model.pt")):
            model = torch.load(Path(config.MODELS_DIR, "model.pt"))
        else:
            # Load the model from the Model Registry
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pytorch.load_model(model_uri)

            torch.save(model, Path(config.MODELS_DIR, "model.pt"))

        return cls(model, device)

    @classmethod
    def from_local_model(cls, model_name: str, device: Literal["cuda", "cpu"] = "cpu"):
        """
        Creates an Inference object from a locally stored model file.

        Args:
            model_name: Name of the local model file (without extension).
            device (Literal["cuda", "cpu"], optional): The device to run inference on, defaults to "cpu".

        Returns:
            Inference: An initialized Inference object.
        """

        if os.path.exists(Path(config.MODELS_DIR, f"{model_name}.pt")):
            model = torch.load(Path(config.MODELS_DIR, f"{model_name}.pt"))
            return cls(model, device)
        else:
            logging.error(f"model {model_name}.pt is not in model_store directory.")

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs inference on the provided image tensor.

        Args:
            image (torch.Tensor): Tensor representing the image to predict.
                Accepts both batch and single images. Format must be in:
                - (batch_size, n_channels, image_dim, image_dim)
                - (n_channels, image_dim, image_dim)
                - (image_dim, image_dim, n_channels)

        Returns:
            torch.Tensor: Predicted output tensor.
        """
        image = image.to(self.device)

        image_dim = list(image.shape)

        if len(image_dim) == 3:
            if image_dim[0] == 3:
                image = image.unsqueeze(0)
            elif image_dim[2] == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                logging.error(
                    """Wrong format. 
                        Format must be in (batch_size, n_channels, image_dim, image_dim) |
                        (n_channels, image_dim, image_dim) |
                        (image_dim, image_dim, n_channels)."""
                )

        self.model.eval()
        with torch.inference_mode():
            predict = self.model(image)

        return predict
