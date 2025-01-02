import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

from src.data.mask_updater import MaskUpdater
from src.utils.config_loader import ConfigLoader

class DataTransformer:
    """
    DataTransformer class to handle image and mask transformations for training and validation.

    Attributes:
        data_dim (int): Dimension to resize the images and masks.
        class_dict (dict): Dictionary mapping class names to their respective labels.
        encoder_name (str): Name of the encoder to use for preprocessing.
        weights_name (str): Name of the pretrained weights to use for the encoder.
        image_transformer (callable): Function to transform images.
        mask_transformer (callable): Function to transform masks.
    """
    def __init__(self, config: ConfigLoader):
        """
        Initialize the DataTransformer with the given configuration.

        Args:
            config (ConfigLoader): Configuration loader with necessary parameters.
        """
        self.data_dim = config.data_dim
        self.class_dict = config.class_map
        self.encoder_name = config.encoder_name
        self.weights_name = config.weights_name

        self.image_transformer, self.mask_transformer = self.load_transformer()

    def __call__(self, image: np.ndarray=None, mask: np.ndarray=None) -> dict[str, torch.Tensor]:
        """
        Transform the given image and/or mask.

        Args:
            image (np.ndarray, optional): Image to transform. Defaults to None.
            mask (np.ndarray, optional): Mask to transform. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the transformed image and mask.
        """
        return {
            "image": self.image_transformer(image) if image is not None else None,
            "mask": self.mask_transformer(mask) if mask is not None else None
        }

    def load_transformer(self):
        """
        Load the image and mask transformers.

        Returns:
            tuple: Tuple containing the image transformer and mask transformer functions.
        """
        mask_updater = MaskUpdater(self.class_dict)
        transform = A.Compose(
            [
                A.Resize(height=self.data_dim, width=self.data_dim),
                ToTensorV2(),
            ]
        )
        encoder_preprocess = get_preprocessing_fn(self.encoder_name, pretrained=self.weights_name)

        def _image_transformer(image: np.ndarray):
            """
            Transform the given image.

            Args:
                image (np.ndarray): Image to transform.

            Returns:
                torch.Tensor: Transformed image.
            """
            preprocessed_image: np.ndarray = encoder_preprocess(image)
            transformed_image: torch.Tensor = transform(image=preprocessed_image)["image"]

            return transformed_image.float()

        def _mask_transformer(mask: np.ndarray):
            """
            Transform the given mask.

            Args:
                mask (np.ndarray): Mask to transform.

            Returns:
                torch.Tensor: Transformed mask.
            """
            transformed_mask = transform(image=np.zeros_like(mask), mask=mask)["mask"]
        
            updated_mask = mask_updater(transformed_mask.long())
            onehot_mask: torch.Tensor = torch.nn.functional.one_hot(updated_mask, len(self.class_dict)).permute(2, 0, 1)

            return onehot_mask

        return _image_transformer, _mask_transformer