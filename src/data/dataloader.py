from pathlib import Path

import cv2

from config import config
from src.data.labels import updated_class_dict


class Data:
    def __init__(self, city, seq, frame, dataset) -> None:
        self.city = city
        self.dataset = dataset
        self.name = city + "_" + seq + "_" + frame

    def get_filename(self, type: str, load_raw: bool = False):
        match type:
            case "image":
                filename = self.name + "_" + config.IMG_TYPE + config.IMG_FORMAT
                if load_raw:
                    modified_dir = Path(config.RAW_IMG_DIR, self.dataset, self.city)
                    return str(Path(modified_dir, filename))
                else:
                    return str(
                        Path(
                            config.PROCESSED_DATA_DIR, self.dataset, "images", filename
                        )
                    )
            case "mask":
                filename = self.name + "_" + config.MASK_TYPE + config.MASK_FORMAT
                if load_raw:
                    modified_dir = Path(config.RAW_MASK_DIR, self.dataset, self.city)
                    return str(Path(modified_dir, filename))
                else:
                    return str(
                        Path(config.PROCESSED_DATA_DIR, self.dataset, "masks", filename)
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
