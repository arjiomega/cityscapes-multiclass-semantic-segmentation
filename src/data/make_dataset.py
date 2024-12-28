import os
import shutil
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import config
from src.data.load_dataset import Data
from src.data.update_mask import update_mask
from src.data.data_utils import extract_data_name

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)


def _cities_similarity_rule(img_dataset_path: Path, mask_dataset_path: Path) -> None:
    cities_similarity_rule = os.listdir(img_dataset_path) == os.listdir(
        mask_dataset_path
    )
    assert cities_similarity_rule, "images and masks cities must be the same"


def _filename_similarity_rule(img_city_path: Path, mask_city_path: Path) -> None:
    img_city_set = set(
        [extract_data_name(filename) for filename in os.listdir(img_city_path)]
    )
    mask_city_set = set(
        [extract_data_name(filename) for filename in os.listdir(mask_city_path)]
    )

    filename_similarity_rule = img_city_set == mask_city_set
    assert filename_similarity_rule, "images and masks filenames must be the same"


def _update_dir_name() -> None:

    if os.path.exists(Path(config.RAW_IMG_DIR, "val")):
        # update val to valid dir
        img_val_dir = str(Path(config.RAW_IMG_DIR, "val"))
        updated_img_val_dir = str(Path(config.RAW_IMG_DIR, "valid"))
        os.rename(img_val_dir, updated_img_val_dir)
    if os.path.exists(Path(config.RAW_MASK_DIR, "val")):
        # update val to valid dir
        mask_val_dir = str(Path(config.RAW_MASK_DIR, "val"))
        updated_mask_val_dir = str(Path(config.RAW_MASK_DIR, "valid"))
        os.rename(mask_val_dir, updated_mask_val_dir)


if __name__ == "__main__":
    dataset_names = ["train", "valid", "test"]

    _update_dir_name()

    for dataset in dataset_names:
        img_dataset_path = Path(config.RAW_IMG_DIR, dataset)
        mask_dataset_path = Path(config.RAW_MASK_DIR, dataset)

        _cities_similarity_rule(img_dataset_path, mask_dataset_path)

        list_of_cities = os.listdir(img_dataset_path)

        for city in list_of_cities:
            img_city_path = Path(img_dataset_path, city)
            mask_city_path = Path(mask_dataset_path, city)
            _filename_similarity_rule(img_city_path, mask_city_path)

            filenames_no_ext = set(
                [extract_data_name(filename) for filename in os.listdir(mask_city_path)]
            )

            logging.info(f"Loading {dataset} dataset for city {city}")

            for filename in tqdm(filenames_no_ext):
                img_filename = filename + "_" + config.IMG_TYPE + config.IMG_FORMAT
                mask_filename = filename + "_" + config.MASK_TYPE + config.MASK_FORMAT
                img_source_path = Path(img_city_path, img_filename)
                mask_source_path = Path(mask_city_path, mask_filename)

                img_destination_path = Path(
                    config.PROCESSED_DATA_DIR, dataset, "images"
                )
                mask_destination_path = Path(
                    config.PROCESSED_DATA_DIR, dataset, "masks"
                )
                img_destination_path.mkdir(parents=True, exist_ok=True)
                mask_destination_path.mkdir(parents=True, exist_ok=True)

                shutil.copy(img_source_path, img_destination_path)

                filename_split = filename.split("_")
                data = Data(*filename_split, dataset)
                mask = data.load_array("mask", load_raw=True)
                updated_mask = update_mask(mask)

                mask_as_image = Image.fromarray(updated_mask)
                mask_as_image.save(Path(mask_destination_path, mask_filename))
