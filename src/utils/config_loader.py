import json
import logging
from pathlib import Path
from src.models import architectures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigLoader:
    REQUIRED_KEYS = [
        "architecture",
        "encoder_family",
        "encoder_name",
        "weights_name",
        "loss_function",
        "metric_function",
        "optimizer",
        "epochs",
        "learning_rate",
        "batch_size",
        "data_dim",
        "random_seed",
        "class_map"
    ]
    MLFLOW_KEYS = [
        "experiment_name",
        "run_name",
    ]
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.architecture = self.config["architecture"]
        self.encoder_family = self.config["encoder_family"]
        self.encoder_name = self.config["encoder_name"]
        self.weights_name = self.config["weights_name"]
        self.loss_function = self.config["loss_function"]
        self.metric_function = self.config["metric_function"]
        self.optimizer = self.config["optimizer"]
        self.epochs = self.config["epochs"]
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.data_dim = self.config["data_dim"]
        self.random_seed = self.config["random_seed"]
        self.class_map = self.config["class_map"]
        self.experiment_name = self.config.get("experiment_name", None)
        self.run_name = self.config.get("run_name", None)


    def load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            missing_keys = [key for key in self.REQUIRED_KEYS if key not in config]
            if missing_keys:
                raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
            mlflow_keys = [key for key in self.MLFLOW_KEYS if key in config]
            if mlflow_keys:
                logger.info(f"MLFlow Keys Found.")
            return config


if __name__ == "__main__":
    config_path = "/home/rjomega-linux/projects/cityscapes-multiclass-semantic-segmentation/config/train_args.json"
    config_loader = ConfigLoader(config_path)
    architecture = config_loader.config["architecture"]
    print(architecture)