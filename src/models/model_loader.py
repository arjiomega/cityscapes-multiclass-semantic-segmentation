from pathlib import Path

from config import config
from src.models import architectures, encoders
from src.utils.config_loader import ConfigLoader

class ModelLoader:
    def __init__(self, model):
        self.model = model
        self.is_from_segmentation_models = False

    def load_model(self):
        return self.model

    @classmethod
    def from_segmentation_models(
        cls, 
        architecture_name: str, 
        encoder_family: str, 
        encoder_name: str, 
        weights_name: str, 
        n_classes:int,
        in_channels: int=3,
        ):
        if architecture_name not in architectures.keys():
            raise ValueError(f"Architecture {architecture_name} is not available.")
        if encoder_family not in encoders.keys():
            raise ValueError(f"Encoder Family {encoder_family} is not available.")
        if encoder_name not in encoders[encoder_family].keys():
            raise ValueError(f"Encoder {encoder_name} is not available in the encoder family {encoder_family}.")
        if weights_name not in encoders[encoder_family][encoder_name]['weights']:
            raise ValueError(f"Weights {weights_name} are not available for encoder {encoder_name} in the encoder family {encoder_family}.")

        architecture = architectures[architecture_name]
        model = architecture(encoder_name=encoder_name, encoder_weights=weights_name, in_channels=in_channels, classes=n_classes)
        cls.is_from_segmentation_models = True
        return cls(model)
    
    @classmethod
    def from_config(cls, config_path=Path(config.CONFIG_DIR, "train_args.json"), config: ConfigLoader = None):
      
        config = config if config else ConfigLoader.from_path(config_path)

        return cls.from_segmentation_models(
            architecture_name=config.architecture,
            encoder_family=config.encoder_family,
            encoder_name=config.encoder_name,
            weights_name=config.weights_name,
            n_classes=len(config.class_map)
        )

if __name__ == "__main__":
    model_loader = ModelLoader.from_segmentation_models("DeepLabV3Plus", "efficientnet", "efficientnet-b3", "imagenet", 5)
    model = model_loader.load_model()
    print(model)