import json
import logging
from pathlib import Path

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
    def __init__(
            self, 
            architecture,
            encoder_family,
            encoder_name,
            weights_name,
            loss_function,
            metric_function,
            optimizer,
            epochs,
            learning_rate,
            batch_size,
            data_dim,
            random_seed,
            class_map,
            experiment_name = None,
            run_name = None,
        ):
        self.architecture = architecture
        self.encoder_family = encoder_family
        self.encoder_name = encoder_name
        self.weights_name = weights_name
        self.loss_function = loss_function
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.random_seed = random_seed
        self.class_map = class_map
        self.experiment_name = experiment_name
        self.run_name = run_name

    @classmethod
    def from_path(cls, config_path: str):
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
            missing_keys = [key for key in cls.REQUIRED_KEYS if key not in config]
            if missing_keys:
                raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
            mlflow_keys = [key for key in cls.MLFLOW_KEYS if key in config]
            if mlflow_keys:
                logger.info(f"MLFlow Keys Found.")
                    
        return cls(
            config["architecture"],
            config["encoder_family"],
            config["encoder_name"],
            config["weights_name"],
            config["loss_function"],
            config["metric_function"],
            config["optimizer"],
            config["epochs"],
            config["learning_rate"],
            config["batch_size"],
            config["data_dim"],
            config["random_seed"],
            config["class_map"],
            config.get("experiment_name", None),
            config.get("run_name", None)
        )

    @classmethod
    def from_dict(cls, config: dict):
        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
        mlflow_keys = [key for key in cls.MLFLOW_KEYS if key in config]
        if mlflow_keys:
            logger.info(f"MLFlow Keys Found.")
        return cls(
            config["architecture"],
            config["encoder_family"],
            config["encoder_name"],
            config["weights_name"],
            config["loss_function"],
            config["metric_function"],
            config["optimizer"],
            config["epochs"],
            config["learning_rate"],
            config["batch_size"],
            config["data_dim"],
            config["random_seed"],
            config["class_map"],
            config.get("experiment_name", None),
            config.get("run_name", None)
        )


if __name__ == "__main__":
    config_path = "/home/rjomega-linux/projects/cityscapes-multiclass-semantic-segmentation/config/train_args.json"
    config_loader = ConfigLoader.from_path(config_path)
    architecture = config_loader.config["architecture"]
    print(architecture)