from typing import Literal

import torch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from src.data.utils import CityLoader

# Subset -> Train, Valid, Test
class SubsetLoader:
    def __init__(self, image_path, mask_path):
        self.train = CityLoader("train", image_path, mask_path)
        self.valid = CityLoader("valid", image_path, mask_path)
        self.test = CityLoader("test", image_path, mask_path)
    def __getitem__(self, subset: Literal["train", "valid", "test"]) -> CityLoader:
        if subset == "train":
            return self.train
        elif subset == "valid":
            return self.valid
        elif subset == "test":
            return self.test
        else:
            raise ValueError("Subset not available.")

class LoadDataset(Dataset):
    def __init__(self, 
                 dataset: Literal["train", "valid", "test"], 
                 subset_loader: SubsetLoader, 
                 mask_updater,
                 randomize=True, 
                 transform=None
        ):
        self.dataset = subset_loader[dataset].get_img_mask_pairs(randomize=randomize)
        self.transform = transform if transform is not None else self._transform()
        self.mask_updater = mask_updater
        self.num_classes = self.mask_updater.num_classes

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = np.array(Image.open(img).convert("RGB"))
        mask = np.array(Image.open(mask).convert("L"))

        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented["image"].float(), augmented["mask"].long()

        mask = self.mask_updater(mask)

        # exclude the ignore class        
        mask = torch.nn.functional.one_hot(mask, self.num_classes).permute(2, 0, 1)

        return img, mask

    def __len__(self):
        return len(self.dataset)

    def _transform(self):
        transform = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        return transform