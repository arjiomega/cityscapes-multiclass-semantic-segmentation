import logging
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from config import config
from src.models.model_loader import ModelLoader
from src.train.metrics import IoU
from src.utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
            self, 
            model, 
            metric_fn, 
            loss_fn, 
            optimizer,
            class_dict: dict[str, list[str]],
            scheduler = None,
            training_config: ConfigLoader = None,
            device: torch.device = torch.device("cpu"),
        ):
        self.training_config = training_config

        self.class_dict = class_dict
        self.device = device
        self.notifier = None # discord webhook
        self.reporter = None # plots, etc.

        # model, loss, optimizer, metric
        self.metric_fn = metric_fn
        self.model = model.to(self.device)
        self.loss = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

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

        self.best_mIOU = 0.0 # valid mIOU
     
    @classmethod
    def from_config(
        cls, 
        config_path=Path(config.CONFIG_DIR, "train_args.json"), 
        config: ConfigLoader = None,
        device: torch.device = torch.device("cpu"),
        continue_from_checkpoint: str = None
        ):
        
        config = config if config else ConfigLoader.from_path(config_path)

        if config.metric_function != "IoU":
            raise ValueError(f"Metric function {config.metric_function} is not available. Try 'IoU' or setup ModelTrainer manually.")
        if config.loss_function != "focal_loss":
            raise ValueError(f"Loss function {config.loss_function} is not available. Try 'focal_loss' or setup ModelTrainer manually.")
        if config.optimizer != "Adam":
            raise ValueError(f"Optimizer {config.optimizer} is not available. Try 'Adam' or setup ModelTrainer manually.")
        
        class_dict = config.class_map

        model = ModelLoader.from_config(config=config).load_model().to(device)

        metric_fn = IoU(n_classes=len(class_dict), device=device)
        loss = smp.losses.FocalLoss(mode="multiclass", alpha=0.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)

        if continue_from_checkpoint:
            checkpoint = torch.load(continue_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Model Trainer created from config. Backup per epoch available.")

        return cls(
            model=model,
            metric_fn=metric_fn,
            loss_fn=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            class_dict=class_dict,
            training_config=config,
            device=device
        )

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
            self.notifier.send_start_notification(
                epochs, 
                training_config=self.training_config.to_dict() if self.training_config else None
            )

        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch}/{epochs}")

            self.model.train()
            self.run_epoch(train_loader)
      
            self.model.eval()
            with torch.inference_mode():
                self.run_epoch(valid_loader, eval_mode=True)
                self.save_best_model()

            self.report_and_notify()
            
            self.epoch_counter += 1

    def save_best_model(self):
        if self.training_config is None:
            logger.warning("No training config found. Skipping model save.")
            return
        if self.valid_history["mIOU"][-1] > self.best_mIOU:
            self.best_mIOU = self.valid_history["mIOU"][-1]
            self.save(f"best_model_{self.training_config.run_name}", self.training_config.to_dict())
            logger.info(f"Best model saved with mIOU: {self.best_mIOU:.4f} | Epoch {self.epoch_counter}")

    def run_epoch(self, dataloader: DataLoader, eval_mode=False):
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader):
            logger.debug(f"Image type: {type(images)}, Mask type: {type(masks)}")
            images = images.to(self.device)
            masks = masks.to(self.device)

            pred_logits = self.model(images)
            logits_normalized = pred_logits - pred_logits.mean(dim=1, keepdim=True)
            pred_probs = torch.softmax(logits_normalized, dim=1)

            # loss calculation
            loss_ = self.loss(pred_probs, masks.argmax(dim=1))
            epoch_loss += loss_.item()

            # metric(s) calculation
            self.metric_fn(pred_probs, masks)

            if not eval_mode:
                # Backpropagation
                self.optimizer.zero_grad()
                loss_.backward()
                self.optimizer.step()

        epoch_loss /= len(dataloader)

        mIOU, class_IOU = self.metric_fn.get_scores()

        if self.scheduler:
            self.scheduler.step(mIOU)

        logger.info(f"{'Validation' if eval_mode else 'Training'} - Loss: {epoch_loss:.4f}, mIOU: {mIOU:.4f}")

        history = self.valid_history if eval_mode else self.train_history
        self.update_history(history, epoch_loss, mIOU, class_IOU)
    
    def update_history(self, history: dict, loss: float, mIOU: float, class_IOU: list):
        """Update the training or validation history."""
        history["loss"].append(loss)
        history["mIOU"].append(mIOU)
        for class_score, (class_name, class_score_list) in zip(class_IOU, history["class_IOU"].items()):
            logger.info(f"{class_name} IOU: {class_score:.4f}")
            class_score_list.append(class_score)

    def report_and_notify(self):
        """Generate reports and send notifications."""
        if self.reporter:
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
            if self.notifier:
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

    def save(self, save_path: str, training_config: dict):
        """Save the model and training state."""
        save_path = save_path if save_path.endswith(".pth") else f"{save_path}.pth"
        torch.save({
            "epochs": self.epoch_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "valid_history": self.valid_history,
            "n_classes": len(self.class_dict),
            "training_config": training_config
        }, save_path)

        logger.info(f"Model saved to {save_path}.")