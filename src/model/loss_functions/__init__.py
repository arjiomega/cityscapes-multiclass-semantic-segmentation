import torch

from .dice_loss import DiceLoss

_list_of_losses = [DiceLoss]


def loss_loader(loss_name: str) -> torch.nn.Module:
    return {"dice": DiceLoss}[loss_name]
