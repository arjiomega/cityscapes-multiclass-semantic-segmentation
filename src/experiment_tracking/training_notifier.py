import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.experiment_tracking.discord_webhook import DiscordWebhook

class TrainingNotifier:
    def __init__(self, webhook_url, username="Training Bot", avatar_url=None):
        self.webhook = DiscordWebhook(webhook_url, username, avatar_url)
        self.username = username
        self.avatar_url = avatar_url

    def send_start_notification(self, epochs):
        message = f"Training started for {epochs} epochs."
        self.webhook.send_message(content=message)

    def send_epoch_notification(
            self, 
            epoch, 
            train_loss, 
            train_mIOU, 
            val_loss, 
            val_mIOU, 
            train_class_IOU: dict[str, float], 
            valid_class_IOU: dict[str, float]
        ):
    
        train_fields = [
            {"name": "Training Loss", "value": f"{train_loss:.4f}", "inline": True},
            {"name": "Training mIOU", "value": f"{train_mIOU:.4f}", "inline": True}
        ]

        valid_fields = [
            {"name": "Validation Loss", "value": f"{val_loss:.4f}", "inline": True},
            {"name": "Validation mIOU", "value": f"{val_mIOU:.4f}", "inline": True},
        ]

        for class_name, iou in train_class_IOU.items():
            train_fields.append({"name": f"Train Class {class_name} IOU", "value": f"{iou:.4f}", "inline": True})

        for class_name, iou in valid_class_IOU.items():
            valid_fields.append({"name": f"Valid Class {class_name} IOU", "value": f"{iou:.4f}", "inline": True})

        self.webhook.send_embed(
            title=f"Epoch {epoch} Train Results",
            description="Training results for the current epoch.",
            color=0x00ff00,
            fields=train_fields,
            footer_text=self.username,
            timestamp=datetime.now(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"
        )
        self.webhook.send_embed(
            title=f"Epoch {epoch} Validation Results",
            description="Validation results for the current epoch.",
            color=0x00ff00,
            fields=valid_fields,
            footer_text=self.username,
            timestamp=datetime.now(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"
        )


    def send_plots(self, loss_plot_filepath, metric_plot_filepath):
        with open(loss_plot_filepath, "rb") as loss_file, open(metric_plot_filepath, "rb") as metric_file:    
            files = {
                "loss_plot": loss_file,
                "miou_plot": metric_file,
            }
            self.webhook.send_message(content="", files=files)

    def send_completion_notification(self, epochs, final_train_loss, final_train_mIOU, final_val_loss, final_val_mIOU):
        embed = {
            "title": "Training Complete",
            "description": f"Training completed for {epochs} epochs.",
            "color": 0x00ff00,
            "fields": [
                {"name": "Final Training Loss", "value": f"{final_train_loss:.4f}", "inline": True},
                {"name": "Final Training mIOU", "value": f"{final_train_mIOU:.4f}", "inline": True},
                {"name": "Final Validation Loss", "value": f"{final_val_loss:.4f}", "inline": True},
                {"name": "Final Validation mIOU", "value": f"{final_val_mIOU:.4f}", "inline": True}
            ],
            "footer": {"text": self.username}
        }
        self.webhook.send_message(content=None, embeds=[embed])
