import numpy as np

from src.data import labels

def _get_updated_class_id(class_name: str) -> int:
    if class_name in labels.human:
        return 1
    elif class_name in labels.vehicles:
        return 2
    elif class_name in labels.other_classes:
        class_idx = labels.other_classes.index(class_name)
        return 3 + class_idx
    else:
        return 0

def update_mask(mask):
    updated_mask = np.array(mask)
    for class_name, class_id in labels.class_dict.items():
        updated_mask[mask == class_id] = _get_updated_class_id(class_name)
    return updated_mask