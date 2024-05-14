import torch
from torchmetrics import JaccardIndex


def iou(
    true_mask: torch.Tensor,
    predict_mask: torch.Tensor,
    classmap: dict[int:str],
    device: str,
) -> dict[str : float | dict[str:float]]:
    """IoU / Jaccard Index metric.

    NOTE: torchmetrics can solve mIOU directly but IoU per class is also needed to calculated,
    so mIOU will be solved by getting the mean of IoU per class.

    Args:
        true_mask (torch.Tensor): (batch, num_class, height, width)
        predict_mask (torch.Tensor): (batch, num_class, height, width)
        classmap (dict[int:str]): mapping for the mask. Ex. {0: "unlabeled", 1: "human", ...}
        device (str): the device we are going to perform the calculation on.

    Returns:
        dict[str:float|dict[str:float]] = {
            "IoU": {
                class_1: ...,
                class_2: ...,
                ...
            }
            "mIOU": ...,
        }
    """
    # CALCULATE IoU by class
    iou_fn = JaccardIndex(task="binary", zero_division=0).to(device)

    N, C, H, W = true_mask.shape

    # (batch, height, width) -> (batch, height, width, num_class)
    predict_mask = torch.softmax(predict_mask, dim=1)
    predict_mask = predict_mask.argmax(dim=1)

    # (batch, height, width, num_class) -> (batch, num_class, height, width)
    B, H, W, N = 0, 1, 2, 3
    predict_mask = torch.nn.functional.one_hot(predict_mask, num_classes=C).permute(
        B, N, H, W
    )

    sum_IoU = 0
    iou_by_class = {}

    for class_idx in range(C):
        class_name = classmap[class_idx]

        # (batch, height, width)
        true_mask_by_class = true_mask[:, class_idx, ...]
        predict_mask_by_class = predict_mask[:, class_idx, ...]

        # print(f"TRUE uniques for class {class_name} {torch.unique(true_mask_by_class)}")
        # print(f"PREDICT uniques for class {class_name} {torch.unique(predict_mask_by_class)}")

        iou_by_class[class_name] = iou_fn(
            predict_mask_by_class, true_mask_by_class
        ).item()

        # print(f"iou: {iou_by_class[class_name]:3f} | type {type(iou_by_class[class_name])}")
        # print("-------------------------")

        sum_IoU += iou_by_class[class_name]

    # x = input("One batch ended for IOU, continue?\n\n")

    return {"byclass": iou_by_class, "mean": sum_IoU / C}
