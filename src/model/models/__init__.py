import torch

from .unet import UNET
from .vgg16_unet import VGG16UNET

# For testing purposes only
_list_of_models = [UNET, VGG16UNET]


def model_loader(model_name: str) -> torch.nn.Module:
    return {"unet": UNET, "vgg16_unet": VGG16UNET}[model_name]
