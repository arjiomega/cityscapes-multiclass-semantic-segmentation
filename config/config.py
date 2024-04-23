"""Configuration module for the Cityscapes multiclass semantic segmentation project.

Contains the file system paths for the project's data directories.

Attributes:
    PROJECT_DIR (str): The path to the project's root directory.
    DATA_DIR (pathlib.Path): The path to the project's data directory.
    RAW_DATA_DIR (pathlib.Path): The path to the directory containing the raw
        Cityscapes dataset files.
    PROCESSED_DATA_DIR (pathlib.Path): The path to the directory containing
        preprocessed data for training and testing the network.
    RAW_IMG_DIR (pathlib.Path): The path to the directory containing the raw
        left images.
    PROCESSED_IMG_DIR (pathlib.Path): The path to the directory containing the
        preprocessed left images.
    RAW_MASK_DIR (pathlib.Path): The path to the directory containing the raw
        segmentation masks.
    IMG_TYPE (str): The type of the raw left images.
    IMG_FORMAT (str): The format of the raw left images.
    MASK_TYPE (str): The type of the raw segmentation masks.
    MASK_FORMAT (str): The format of the raw segmentation masks.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = CONFIG_DIR.parent
DATA_DIR = Path(PROJECT_DIR, "data")

RAW_DATA_DIR = Path(DATA_DIR, "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")

RAW_IMG_DIR = Path(RAW_DATA_DIR, "leftImg8bit_trainvaltest", "leftImg8bit")
RAW_MASK_DIR = Path(RAW_DATA_DIR, "gtFine_trainvaltest", "gtFine")

TRAIN_DIR = Path(PROCESSED_DATA_DIR, "train")
VALID_DIR = Path(PROCESSED_DATA_DIR, "valid")
TEST_DIR = Path(PROCESSED_DATA_DIR, "test")

TRAIN_IMG_DIR = Path(TRAIN_DIR, "images")
TRAIN_MASK_DIR = Path(TRAIN_DIR, "masks")

VALID_IMG_DIR = Path(VALID_DIR, "images")
VALID_MASK_DIR = Path(VALID_DIR, "masks")

TEST_IMG_DIR = Path(TEST_DIR, "images")
TEST_MASK_DIR = Path(TEST_DIR, "masks")

IMG_TYPE = "leftImg8bit"
IMG_FORMAT = ".png"
MASK_TYPE = "gtFine_labelIds"
MASK_FORMAT = ".png"
