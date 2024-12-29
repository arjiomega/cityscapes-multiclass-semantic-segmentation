
import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex


class IoU:
    def __init__(self, n_classes, device: torch.device = torch.device("cpu")):
        self.device = device
        self.n_classes = n_classes
        self.binaryIOU = BinaryJaccardIndex().to(device)
        self.IOU = MulticlassJaccardIndex(num_classes=n_classes).to(device)
        self.class_scores = [[] for _ in range(n_classes)]
        self.score = []

    def __call__(self, pred_probs, masks, threshold=0.5):
        # pred_probs - softmax

        # calculate mIOU
        score_ = self.IOU(pred_probs.argmax(dim=1), masks.argmax(dim=1)).to("cpu")
        self.score.append(score_)

        # calculate IOU for each class
        for class_idx, class_score_list in enumerate(self.class_scores):
            class_pred_probs = pred_probs[:, class_idx, :, :]
            class_masks = masks[:, class_idx, :, :]

            # if value is >threshold turn to 1
            class_pred_probs = (class_pred_probs > threshold).long()

            class_score_ = self.binaryIOU(class_pred_probs, class_masks).to("cpu")

            if not torch.isnan(class_score_):
                class_score_list.append(class_score_)

    def get_scores(self) -> tuple[float, list[float]]:
        # get mean across batches
        class_scores_mean = [
            np.mean(class_score_list) if class_score_list else 0.0
            for class_score_list in self.class_scores
            if class_score_list
        ]
        score_mean = np.mean(self.score)

        # reset scores
        self.class_scores = [[] for _ in range(self.n_classes)]
        self.score = []

        return score_mean, class_scores_mean