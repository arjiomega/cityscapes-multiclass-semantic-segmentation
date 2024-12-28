import torch
from torch import nn



class IoULoss(nn.Module):
    def __init__(self) -> None:
        super(IoULoss, self).__init__()

    @staticmethod
    def _iou_coeff(predict: torch.Tensor, target: torch.Tensor):
        T = predict.flatten()
        P = target.flatten()

        intersection = torch.sum(T * P)
        IoU = (intersection + 1.0) / (torch.sum(T) + torch.sum(P) - intersection + 1.0)

        return IoU

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return -self._iou_coeff(predict, target)