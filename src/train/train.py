import argparse
import logging
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from config import config
from src.train.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.data.data_transformer import DataTransformer
from src.data.load_dataset import SubsetLoader, LoadDataset

from src.report.report import Report
from src.experiment_tracking.training_notifier import TrainingNotifier
    

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def main(path: str, train_from: Literal["config", "checkpoint"]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_from == "config":
        training_config = ConfigLoader.from_path(config_path)
    elif train_from == "checkpoint":
        checkpoint = torch.load(path, map_location=device)
        training_config = ConfigLoader.from_dict(checkpoint["training_config"])

    epochs = training_config.epochs
    batch_size = training_config.batch_size

    # Model Trainer
    model_trainer = ModelTrainer.from_config(config=training_config,device=device)

    # Load Checkpoint if available
    if train_from == "checkpoint":
        model_trainer.model.load_state_dict(checkpoint["model_state_dict"])
        model_trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Checkpoint successfully loaded.")

    # Transformations
    data_transformer = DataTransformer(training_config)

    # Load Dataset
    subset_loader = SubsetLoader(config.RAW_IMG_DIR, config.RAW_MASK_DIR)
    train_dataset = LoadDataset("train", subset_loader, data_transformer=data_transformer)
    valid_dataset = LoadDataset("valid", subset_loader, data_transformer=data_transformer)

    batch_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    batch_valid = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    webhook_url = os.getenv("WEBHOOK_URL", None)
    avatar_url = os.getenv("AVATAR_URL", None)
    
    if webhook_url and avatar_url:
        logger.info("Webhook URL and Avatar URL found. Adding Notifier and Reporter.")
        notifier = TrainingNotifier(webhook_url, avatar_url=avatar_url)
        reporter = Report(show_plots=False)
        model_trainer.add_notifier(notifier)
        model_trainer.add_reporter(reporter)
    else:
        logger.info("Webhook URL and Avatar URL not found. Not adding Notifier and Reporter.")

    model_trainer.train(batch_train, batch_valid, epochs)
    model_trainer.save(training_config.run_name, training_config.to_dict())

if __name__ == "__main__":
    config_path = Path(config.CONFIG_DIR, "train_args.json")

    parser = argparse.ArgumentParser(description="Script to train a machine learning model.")

    # Add arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training from."
    )

    # Add arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the training configuration file to resume training from."
    )

    args = parser.parse_args()

    path_, train_from = (args.checkpoint, "checkpoint") if args.checkpoint else (config_path, "config")

    main(path_, train_from)