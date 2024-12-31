import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from src.train.metrics import IoU


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model: torch.nn.Module, device: torch.device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    @classmethod
    def from_pretrained(cls):
        pass

    def compile(self):
        pass

    def inference(self, image):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass

class ModelTrainer:
    def __init__(
            self, 
            model, 
            iou: IoU, 
            class_dict: dict[str, list[str]],
            device: torch.device = torch.device("cpu"),
        ):
        self.device = device
        self.notifier = None # discord webhook
        self.reporter = None # plots, etc.

        # model, loss, optimizer, metric
        self.iou = iou
        self.model = model
        self.loss = sigmoid_focal_loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.epoch_counter = 0
        
        self.train_history = {
            "loss": [], 
            "mIOU": [] , 
            "class_IOU": {class_name: [] for class_name in class_dict.keys()}
        }
        self.valid_history = {
            "loss": [], 
            "mIOU": [] , 
            "class_IOU": {class_name: [] for class_name in class_dict.keys()}
        }
     
    def add_notifier(self, notifier):
        """Used for discord webhook"""
        self.notifier = notifier
        return self

    def add_reporter(self, reporter):
        """Used for plots, etc."""
        self.reporter = reporter
        pass

    def train(self, train_loader, valid_loader, epochs):

        if self.notifier:
            self.notifier.send_start_notification(epochs)

        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch}/{epochs}")

            self.model.train()
            self.run_loop(train_loader)
      
            self.model.eval()
            with torch.inference_mode():
                self.run_loop(valid_loader, eval_mode=True)
  
            if self.notifier and self.reporter:
                self.reporter.plot_loss_history(
                    self.train_history["loss"], 
                    self.valid_history["loss"],
                    filepath="loss_history.png"
                )
                self.reporter.plot_metric_history(
                    self.train_history["mIOU"],
                    self.valid_history["mIOU"],
                    filepath="metric_history.png"
                )
                self.notifier.send_plots("loss_history.png", "metric_history.png")

            if self.notifier:
                self.notifier.send_epoch_notification(
                    epoch=self.epoch_counter,
                    train_loss=self.train_history["loss"][-1],
                    train_mIOU=self.train_history["mIOU"][-1],
                    val_loss=self.valid_history["loss"][-1],
                    val_mIOU=self.valid_history["mIOU"][-1],
                    train_class_IOU={
                        class_name: epoch_history[-1] 
                        for class_name, epoch_history in self.train_history["class_IOU"].items()
                        if epoch_history
                    },
                    valid_class_IOU={
                        class_name: epoch_history[-1] 
                        for class_name, epoch_history in self.valid_history["class_IOU"].items()
                        if epoch_history
                    }
                )

            
            self.epoch_counter += 1

    def run_loop(self, dataloader: DataLoader, eval_mode=False):
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            pred_logits = self.model(images)
            logits_normalized = pred_logits - pred_logits.mean(dim=1, keepdim=True)
            pred_probs = torch.softmax(logits_normalized, dim=1)

            # loss calculation
            loss_ = self.loss(pred_probs, masks.argmax(dim=1))
            epoch_loss += loss_.item()

            # metric(s) calculation
            self.iou(pred_probs, masks)

            if not eval_mode:
                # Backpropagation
                self.optimizer.zero_grad()
                loss_.backward()
                self.optimizer.step()

        epoch_loss /= len(dataloader)

        mIOU, class_IOU = self.iou.get_scores()

        logger.info(f"{'Validation' if eval_mode else 'Training'} - Loss: {epoch_loss:.4f}, mIOU: {mIOU:.4f}")

        history_dict = self.valid_history if eval_mode else self.train_history

        history_dict["mIOU"].append(mIOU)
        history_dict["loss"].append(epoch_loss)

        for class_score, (class_name, class_score_list) in zip(class_IOU, history_dict["class_IOU"].items()):
            logger.info(f"{class_name} IOU: {class_score:.4f}")
            class_score_list.append(class_score)

        return epoch_loss, mIOU
    
    def save(self, save_path, n_classes):
        save_path = save_path if save_path.endswith(".pth") else f"{save_path}.pth"
        torch.save({
            "epochs": self.epoch_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "valid_history": self.valid_history,
            "n_classes": n_classes
        }, save_path)