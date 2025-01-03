import numpy as np
import torch

from src.data.data_transformer import DataTransformer
from src.models.model_loader import ModelLoader
from src.utils.config_loader import ConfigLoader


# Requires a trained model using this new structure
class Inference:
    def __init__(self, model: torch.nn.Module, data_transformer: DataTransformer, device: torch.device):
        self.model = model.to(device)
        self.data_transformer = data_transformer
        self.device = device

    @classmethod
    def load_model_from_checkpoint(cls, checkpoint_path: str, device: torch.device = torch.device("cpu")):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        training_config = ConfigLoader.from_dict(checkpoint["training_config"])

        model = ModelLoader.from_config(config=training_config).load_model()

        model.load_state_dict(checkpoint['model_state_dict'])
    
        data_transformer = DataTransformer(training_config)

        return cls(model, data_transformer, device)

    def __call__(self, image: np.ndarray):
        transformed_image: torch.Tensor = self.data_transformer(image=image)["image"] # (num_classes, width, height)
        transformed_image = transformed_image.unsqueeze(0).float().to(self.device) # (1, num_classes, width, height)

        self.model.eval()
        with torch.inference_mode():
            pred_logits = self.model(transformed_image)
            logits_normalized = pred_logits - pred_logits.mean(dim=1, keepdim=True)
            pred_probs = torch.softmax(logits_normalized, dim=1)
            pred_argmax = pred_probs.argmax(dim=1)[0].to("cpu").numpy()

        return pred_argmax


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference.load_model_from_checkpoint("model_deeplabv3plus_efficientnetb4-part2.pth", device)
