import torch

class_dict = {
    "unlabeled": 0,
    "ego vehicle": 1,
    "rectification border": 2,
    "out of roi": 3,
    "static": 4,
    "dynamic": 5,
    "ground": 6,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "rail track": 10,
    "building": 11,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    "bridge": 15,
    "tunnel": 16,
    "pole": 17,
    "polegroup": 18,
    "traffic light": 19,
    "traffic sign": 20,
    "vegetation": 21,
    "terrain": 22,
    "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
    "license plate": -1,
}

class MaskUpdater:
    """
    A class to update segmentation masks based on a new class mapping.

    Attributes:
    -----------
    updated_class_dict : dict[str, int]
        A dictionary where keys are new class names and values are their corresponding class IDs.
    mapping : dict[int, int]
        A dictionary mapping old class IDs to new class IDs.
    num_classes : int
        The number of classes in the updated class dictionary.

    Methods:
    --------
    __call__(mask: torch.Tensor) -> torch.Tensor
        Updates the given segmentation mask based on the new class mapping.
    """
    def __init__(self, updated_class_dict: dict[str, int]):
        self.updated_class_dict = updated_class_dict
        self.mapping = {
            old_class_id: -1 if class_name == "ignore" else class_id
            for old_class_name, old_class_id in class_dict.items()
            for class_id, (class_name, class_list) in enumerate(updated_class_dict.items()) 
            if old_class_name in class_list
        }
        self.num_classes = len(updated_class_dict)

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        updated_mask = mask.clone()
        for old_class_id, new_class_id in self.mapping.items():
            updated_mask[mask == old_class_id] = new_class_id
        return updated_mask

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
        "ignore": ["license plate"],
    }

    mask_updater = MaskUpdater(updated_class_dict)

    test_mask = torch.randint(-1, 34, (512, 512))

    print(test_mask.unique())
    updated_mask = mask_updater(test_mask)
    print(updated_mask.unique())