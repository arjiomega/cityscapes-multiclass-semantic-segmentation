from pathlib import Path

import pytest

from config import config
from src.data.dataloader import Data

CITY = "aachen"
SEQ = "000000"
FRAME = "000019"
DATASET_GROUP = "train"

IMG_FILENAME = "_".join([CITY, SEQ, FRAME, config.IMG_TYPE + config.IMG_FORMAT])
MASK_FILENAME = "_".join([CITY, SEQ, FRAME, config.MASK_TYPE + config.MASK_FORMAT])


class TestGetFileName:

    @pytest.fixture
    def raw_data(self):
        return Data(CITY, SEQ, FRAME, DATASET_GROUP)

    def test_load_raw_img_data(self, raw_data):
        CORRECT_RAW_IMG_PATH = str(
            Path(config.RAW_IMG_DIR, DATASET_GROUP, CITY, IMG_FILENAME)
        )

        assert (
            raw_data.get_filename("image", load_raw=True) == CORRECT_RAW_IMG_PATH
        ), f"Expected {CORRECT_RAW_IMG_PATH}, got {raw_data.get_filename('image', load_raw=True)}"

    def test_load_raw_mask_data(self, raw_data):
        CORRECT_RAW_MASK_PATH = str(
            Path(config.RAW_MASK_DIR, DATASET_GROUP, CITY, MASK_FILENAME)
        )

        assert (
            raw_data.get_filename("mask", load_raw=True) == CORRECT_RAW_MASK_PATH
        ), f"Expected {CORRECT_RAW_MASK_PATH}, got {raw_data.get_filename('mask', load_raw=True)}"

    def test_load_processed_img_data(self, raw_data):
        CORRECT_PROCESSED_IMG_PATH = str(
            Path(config.PROCESSED_DATA_DIR, DATASET_GROUP, "images", IMG_FILENAME)
        )

        assert (
            raw_data.get_filename("image") == CORRECT_PROCESSED_IMG_PATH
        ), f"Expected {CORRECT_PROCESSED_IMG_PATH}, got {raw_data.get_filename('image')}"

    def test_load_processed_mask_data(self, raw_data):
        CORRECT_PROCESSED_MASK_PATH = str(
            Path(config.PROCESSED_DATA_DIR, DATASET_GROUP, "masks", MASK_FILENAME)
        )

        assert (
            raw_data.get_filename("mask") == CORRECT_PROCESSED_MASK_PATH
        ), f"Expected {CORRECT_PROCESSED_MASK_PATH}, got {raw_data.get_filename('mask')}"
