import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from config import config
from src.data.labels import updated_class_dict
from src.data.data_utils import extract_data_name


def _get_filename_info(filename: str) -> list[str]:
    """Get the city, sequence, and frame from the filename.

    Ex.
        'strasbourg_000001_063385' -> ['strasbourg', '000001', '063385']
    """
    return extract_data_name(filename).split("_")


def _get_dataset_path(dataset_group: str) -> tuple[Path, Path]:

    return {
        "train": (config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR),
        "valid": (config.VALID_IMG_DIR, config.VALID_MASK_DIR),
        "test": (config.TEST_IMG_DIR, config.TEST_MASK_DIR),
    }[dataset_group]


class Data:
    def __init__(
        self,
        city,
        seq,
        frame,
        dataset,
        img_type: str = config.IMG_TYPE,
        img_format: str = config.IMG_FORMAT,
        mask_type: str = config.MASK_TYPE,
        mask_format: str = config.MASK_FORMAT,
        raw_img_dir: str | Path = config.RAW_IMG_DIR,
        raw_mask_dir: str | Path = config.RAW_MASK_DIR,
        processed_data_dir: str | Path = config.PROCESSED_DATA_DIR,
    ) -> None:
        self.city = city
        self.dataset = dataset
        self.name = city + "_" + seq + "_" + frame

        self.image_prefix = img_type + img_format
        self.mask_prefix = mask_type + mask_format

        self.raw_img_dir = raw_img_dir
        self.raw_mask_dir = raw_mask_dir
        self.processed_data_dir = processed_data_dir

    def get_filename(self, type: str, load_raw: bool = False):
        match type:
            case "image":
                filename = self.name + "_" + self.image_prefix
                if load_raw:
                    modified_dir = Path(self.raw_img_dir, self.dataset, self.city)
                    return str(Path(modified_dir, filename))
                else:
                    return str(
                        Path(self.processed_data_dir, self.dataset, "images", filename)
                    )
            case "mask":
                filename = self.name + "_" + self.mask_prefix
                if load_raw:
                    modified_dir = Path(self.raw_mask_dir, self.dataset, self.city)
                    return str(Path(modified_dir, filename))
                else:
                    return str(
                        Path(self.processed_data_dir, self.dataset, "masks", filename)
                    )
            case _:
                raise ValueError(f"Invalid type: {type}. Must be either image or mask.")

    def get_classes(self):
        mask_array = self.load_array("mask")

        return {
            class_idx: updated_class_dict[class_idx]
            for class_idx in set(mask_array.flatten())
        }

    def load_array(self, type: str, load_raw: bool = False):
        match type:
            case "image":
                filename = self.get_filename("image", load_raw)
                img_array = cv2.imread(filename)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                return img_array
            case "mask":
                filename = self.get_filename("mask", load_raw)
                mask_array = cv2.imread(filename, 0)
                return mask_array
            case _:
                raise ValueError(f"Invalid type: {type}. Must be either image or mask.")


class LoadDataset(Dataset):
    def __init__(
        self,
        dataset_group: str,
        n_classes: int,
        transform=None,
        dataset_img_dir: Path = None,
        dataset_mask_dir: Path = None,
    ) -> None:
        """Load cityscapes dataset.

        Args:
            dataset_group (str): "train", "valid", or "test"
            n_classes (int): Number of classes in the mask.
            transform : Tranformation function that is going to be used for the data.
            Defaults to None.
            dataset_img_dir (Path, optional): directory containing images for a certain
            dataset_group. Defaults to None.
            dataset_mask_dir (Path, optional): directory containing masks for a certain
            dataset_group. Defaults to None.
        """
        if dataset_img_dir and dataset_mask_dir:
            img_path = dataset_img_dir
            mask_path = dataset_mask_dir
        else:
            img_path, mask_path = _get_dataset_path(dataset_group)

        img_names = sorted(
            [extract_data_name(filename) for filename in os.listdir(img_path)]
        )
        mask_names = sorted(
            [extract_data_name(filename) for filename in os.listdir(mask_path)]
        )

        assert len(img_names) == len(
            mask_names
        ), "images and masks filenames must be the same"

        # check if images and masks filenames are the same
        # strasbourg_000001_063385 == strasbourg_000001_063385
        assert img_names == mask_names, "images and masks cities must be the same"

        self.data = [
            Data(*_get_filename_info(path), dataset_group)
            for path in os.listdir(img_path)
        ]

        self.n_classes = n_classes

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index].load_array("image")
        mask = self.data[index].load_array("mask")

        if self.transform:
            augmented_data = self.transform(image=image, mask=mask)
            image: torch.Tensor = augmented_data["image"].float()
            mask: torch.Tensor = augmented_data["mask"].long()
        else:
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()

        onehot_mask = torch.nn.functional.one_hot(mask, self.n_classes).permute(2, 0, 1)

        return image, onehot_mask


def load_dataset(
    dataset_group: str,
    batch_size: int,
    n_classes: int,
    transform,
    num_workers: int = 4,
    *args,
    **kwargs,
) -> DataLoader:

    failed_dataset_group_response = f"{dataset_group} is an invalid dataset_group. \
                                    must be 'train', 'valid', 'test'"
    assert dataset_group in ["train", "valid", "test"], failed_dataset_group_response

    dataset = LoadDataset(dataset_group, transform=transform, n_classes=n_classes)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return dataloader
