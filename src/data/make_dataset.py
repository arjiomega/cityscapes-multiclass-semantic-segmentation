import os
import shutil
from pathlib import Path

from PIL import Image

from config import config
from src.data.load_data import Data
from src.data.update_mask import update_mask

def _extract_data_name(filename: str) -> str:
    str2list = filename.split("_")[:3]
    return "_".join(str2list)

def _cities_similarity_rule(img_dataset_path: Path, mask_dataset_path: Path) -> None:
    cities_similarity_rule = os.listdir(img_dataset_path) == os.listdir(mask_dataset_path)
    assert cities_similarity_rule, "images and masks cities must be the same"
    
def _filename_similarity_rule(img_city_path: Path, mask_city_path: Path) -> None:
    img_city_set = set([_extract_data_name(filename) 
        for filename in os.listdir(img_city_path)
    ])
    mask_city_set = set([_extract_data_name(filename) 
        for filename in os.listdir(mask_city_path)
    ])
    
    filename_similarity_rule = img_city_set == mask_city_set
    assert filename_similarity_rule, "images and masks filenames must be the same"

if __name__ == "__main__":
    dataset_names = ["train", "val", "test"]
    
    for dataset in dataset_names:
        img_dataset_path = Path(config.RAW_IMG_DIR, dataset)
        mask_dataset_path = Path(config.RAW_MASK_DIR, dataset)
        
        _cities_similarity_rule(img_dataset_path, mask_dataset_path)
    
        list_of_cities = os.listdir(img_dataset_path)
        
        for city in list_of_cities:
            img_city_path = Path(img_dataset_path, city)
            mask_city_path = Path(mask_dataset_path, city)
            _filename_similarity_rule(img_city_path, mask_city_path)
            
            filenames_no_ext = set([_extract_data_name(filename) 
                for filename in os.listdir(mask_city_path)
            ])
            
            for filename in filenames_no_ext:
                img_filename = filename + "_" + config.IMG_TYPE + config.IMG_FORMAT
                mask_filename = filename + "_" + config.MASK_TYPE + config.MASK_FORMAT
                img_source_path = Path(img_city_path, img_filename)
                mask_source_path = Path(mask_city_path, mask_filename)
                
                img_destination_path = Path(config.PROCESSED_IMG_DIR, dataset)
                mask_destination_path = Path(config.PROCESSED_MASK_DIR, dataset)
                img_destination_path.mkdir(parents=True, exist_ok=True)
                mask_destination_path.mkdir(parents=True, exist_ok=True)
                
                shutil.copy(img_source_path,img_destination_path)
                
                filename_split = filename.split("_")
                data = Data(*filename_split, dataset)
                mask = data.load_array("mask")
                updated_mask = update_mask(mask)
                
                mask_as_image = Image.fromarray(updated_mask)
                mask_as_image.save(Path(mask_destination_path, mask_filename))