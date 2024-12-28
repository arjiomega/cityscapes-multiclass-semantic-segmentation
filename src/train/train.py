import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import config
from src.models.VGG16UNET import VGG16UNET
from src.data.mask_updater import MaskUpdater
from src.data.load_dataset import LoadDataset, SubsetLoader

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
    def __init__(self, model, device=None):
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.loss = sigmoid_focal_loss
        self.metric = None
        self.model_modifier = None
        self.train_history = {"loss": [], "iou": []}
        self.valid_history = {"loss": [], "iou": []}
     
    def train(self, train_loader, valid_loader, epochs):

        for epoch in range(epochs):

            self.model.train()
            train_mIOU, train_loss = self.run_loop(train_loader)
            self.train_history["loss"].append(train_loss)
            self.train_history["iou"].append(train_mIOU)

            self.model.eval()
            with torch.inference_mode():
                val_mIOU, val_loss = self.run_loop(valid_loader, eval_mode=True)
                self.valid_history["loss"].append(val_loss)
                self.valid_history["iou"].append(val_mIOU)
            
            if self.model_modifier is not None:
                self.model = self.model_modifier(self.model)


    def run_loop(self, dataloader: DataLoader, eval_mode=False):
        iou_list = []
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            pred_logits = self.model(images)
            logits_normalized = pred_logits - pred_logits.mean(dim=1, keepdim=True)
            pred_probs = torch.softmax(logits_normalized, dim=1)

            # loss calculation
            loss_ = self.loss(pred_probs, masks.float(), reduction='mean')

            if not eval_mode:
                # Backpropagation
                self.optimizer.zero_grad()
                loss_.backward()
                self.optimizer.step()

        epoch_loss /= len(dataloader)
        mIOU = np.mean(iou_list)

        return epoch_loss, mIOU

if __name__ == "__main__":
    updated_class_dict = {
        "unlabeled": ["unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "dynamic", "ground"],
        "road": ["road", "parking", "rail track"],
        "vehicles": ["car", "motorcycle", "bus", "truck", "train", "caravan", "trailer"],
        "bike": ["bicycle"],
        "person": ["person"],
        "rider": ["rider"],
        "sky": ["sky"],
        "vegetation": ["vegetation", "terrain"],
        "objects": ["traffic light", "traffic sign", "pole", "polegroup"],
        "walls": ["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
        "sidewalk": ["sidewalk"],
        "license plate": ["license plate"],
    }

    mask_updater = MaskUpdater(updated_class_dict)

    data_transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    subset_loader = SubsetLoader(config.RAW_IMG_DIR, config.RAW_MASK_DIR)
    train_dataset = LoadDataset("train", subset_loader, mask_updater, transform=data_transform)
    valid_dataset = LoadDataset("valid", subset_loader, mask_updater, transform=data_transform)
    batch_train = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    batch_valid = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    model = VGG16UNET()
    model_trainer = ModelTrainer(model=VGG16UNET(mask_updater.num_classes))
    # model_trainer.train(batch_train, batch_valid, epochs=1)
    
