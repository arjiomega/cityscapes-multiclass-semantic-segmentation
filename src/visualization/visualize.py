import torch
import matplotlib.figure
import matplotlib.pyplot as plt


def _fix_dim(device: str, *args) -> tuple[torch.Tensor, ...]:
    """Fix dimensions for visualization purposes.

    Args:
        device (str): the device where we want to run the model. Either "cuda" or "cpu".
        *args (torch.Tensor): Take any number of torch tensor with a shape of (batch_size, channels, height, width).
        Also requires batch size of 1.

    Returns:
        tuple[torch.Tensor, ...]: Processed torch tensors (height, width, channels)
    """

    processed_list = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            raise TypeError("Input must be a torch Tensor")
        else:
            assert arg.shape[0] == 1, "Batch size of 1 of required."

            processed_arg = arg.squeeze(0).permute(1, 2, 0)

            print(processed_arg.shape)

            processed_list.append(processed_arg)

    return tuple(processed_list)


def from_prediction(
    model: torch.nn.Module, image: torch.Tensor, true_mask: torch.Tensor, device: str
) -> matplotlib.figure.Figure:
    """_summary_

    Args:
        image (torch.Tensor): Expected shape of (1, 3, height, width)
        true_mask (torch.Tensor): Expected shape of (1, num_classes, height, width)
    """

    model.eval()

    fig, arr = plt.subplots(1, 3)

    with torch.inference_mode():
        pred_logits = model(image.to(device))  # (1, num_classes, height, width)
        predict = torch.softmax(pred_logits, dim=1)  # (1, num_classes, height, width)

        true_mask = true_mask.argmax(dim=1).squeeze()  # (height, width)
        predict = predict.argmax(dim=1).squeeze()  # (height, width)

        (image,) = _fix_dim(
            device, image
        )  # image.squeeze(0).permute(1, 2, 0) # (height, width, num_classes)

        arr[0].imshow(image)
        arr[1].imshow(true_mask)
        arr[2].imshow(predict.to("cpu"))

        fig.savefig("full_figure.png")
