from typing import Literal

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.data_transformer import DataTransformer
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
                 data_transformer: DataTransformer,
                 randomize=True, 
        ):
        self.dataset = subset_loader[dataset].get_img_mask_pairs(randomize=randomize)
        self.data_transformer = data_transformer
   

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = np.array(Image.open(img).convert("RGB"))
        mask = np.array(Image.open(mask).convert("L"))

        transformed_data = self.data_transformer(img, mask)
        transformed_img, transformed_mask = transformed_data["image"], transformed_data["mask"]

        return transformed_img, transformed_mask

    def __len__(self):
        return len(self.dataset)