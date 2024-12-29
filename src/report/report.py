import logging
import numpy as np
from io import BytesIO

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Report:
    def __init__(self, show_plots=False, save_plots=True):
        self.show_plots = show_plots
        self.save_plots = save_plots

    def plot_loss_history(self, train_history, valid_history, filepath="loss_history.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label="Train Loss")
        plt.plot(valid_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()

        if self.save_plots:
            plt.savefig(filepath)
        
        if self.show_plots:
            plt.show()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def plot_metric_history(self, train_history, valid_history, metric="mIOU", filepath="metric_history.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label=f"Train {metric}")
        plt.plot(valid_history, label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} History")
        plt.legend()

        if self.save_plots:
            plt.savefig(filepath)
        
        if self.show_plots:
            plt.show()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def plot_training_results(
        self, 
        images, 
        masks, 
        predict_probs_mask, 
        epoch, 
        iou, 
        filepath="image_plot.png"
        ):
        # transform back image
        image = (images.to("cpu")[0].permute(1, 2, 0).numpy() * self.std) + self.mean
        image = np.clip(image, 0, 1)

        # mask
        mask = masks.to("cpu").argmax(dim=1)[0]

        # predict mask
        pred_mask = predict_probs_mask.to('cpu').argmax(dim=1)[0]

        fig, arr = plt.subplots(1, 3, figsize=(10, 4))
        arr[0].imshow(image)
        arr[0].set_title("Image")
        arr[0].axis('off')
        arr[1].imshow(mask)
        arr[1].set_title("Mask")
        arr[1].axis('off')
        arr[2].imshow(pred_mask)
        arr[2].set_title("Predict Mask")
        arr[2].axis('off')

        # set plot title
        fig.suptitle(f"Training Results for Epoch {epoch} IoU: {iou:.4f}")

        if self.save_plots:
            plt.savefig(filepath)
        
        if self.show_plots:
            plt.show()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf