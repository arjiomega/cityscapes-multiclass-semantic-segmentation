import torch
from torch import nn

class DiceLoss(nn.Module):
    """Custom Dice Loss for PyTorch

    DICE LOSS = 1 - [2*p*t / (p^2 + t^2)]

    Squaring the denominator p^2 + t^2 results in faster convergence and works better for lower region of interest.
    src: https://stackoverflow.com/questions/66536963/dice-loss-working-only-when-probs-are-squared-at-denominator

    Args:
        target (torch.int64): the true mask in a format of (batch, classes, height, width)
            Example: (4, 8, 512, 512)
        predict (torch.float32): prediction after passing our features (X) in our model.
            Format looks like (batch, classes, height, width).
            WARNING: prediction is still in logits form.
                Example: (4, 8, 512, 512)

    Returns:
        torch.float32: Dice Loss. Shape (1,)
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predict = torch.softmax(predict, dim=1) # (batch_size, channel, height, width)
        target = target.type(torch.float32)

        # 2*p*t
        intersection = torch.sum(predict * target, dim=(2,3)) # (batch_size, channel)
        # p^2 + t^2
        union = torch.sum(predict.pow(2), dim=(2,3)) + torch.sum(target.pow(2), dim=(2,3)) # (batch_size, channel)

        # 2*p*t / (p^2 + t^2)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth) # (batch_size, channel)

        dice_loss = 1 - torch.mean(dice_coef)  # (1,)

        return dice_loss
