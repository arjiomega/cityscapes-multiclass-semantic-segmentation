from pathlib import Path

import cv2

from config import config

class Data:
    def __init__(self,city,seq,frame, dataset) -> None:   
        self.city = city
        self.dataset = dataset
        self.name = city + "_" + seq + "_" + frame
        
    def get_filename(self, type:str):
        match type:
            case "image":
                filename = self.name + "_" + config.IMG_TYPE + config.IMG_FORMAT
                return str(Path(config.RAW_IMG_DIR, self.dataset, self.city, filename))
            case "mask":
                filename = self.name + "_" + config.MASK_TYPE + config.MASK_FORMAT
                return str(Path(config.RAW_MASK_DIR, self.dataset, self.city, filename))
            case _:
                raise ValueError(f"Invalid type: {type}. Must be either image or mask.")
    
    def load_array(self, type:str):
        match type:
            case "image":
                filename = self.get_filename("image")
                img_array = cv2.imread(filename)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                return img_array
            case "mask":
                filename = self.get_filename("mask")
                mask_array = cv2.imread(filename, 0)
                return mask_array
            case _:
                raise ValueError(f"Invalid type: {type}. Must be either image or mask.")